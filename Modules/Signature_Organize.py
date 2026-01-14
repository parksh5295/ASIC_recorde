import logging
from Signature_tool.performance import calculate_signatures_performance
from Signature_tool.subsumption import find_redundant_signatures
from Signature_tool.merging import merge_similar_signatures
import pandas as pd
import os
import pickle
import gc  # For explicit garbage collection in memory-heavy organization


logger = logging.getLogger(__name__)

def organize_signatures(
    all_signatures: dict, 
    data_provider,
    num_processes: int,
    run_dir: str, # For storing temp files
    turn_counter: int, # To create unique temp filenames
    coverage_threshold: float = 0.95,
    enable_merging: bool = False,
    merge_infrequent_threshold: int = 5,
    data_batch_size: int = 50000
) -> dict:
    """
    Organizes a set of signatures by calculating performance, merging, and pruning.
    Includes intermediate saving to handle crashes during the expensive performance calculation step.
    """
    data_is_empty = False
    if isinstance(data_provider, pd.DataFrame):
        data_is_empty = data_provider.empty
    elif data_provider is None:
        data_is_empty = True
    elif hasattr(data_provider, "is_empty"):
        data_is_empty = data_provider.is_empty()

    if not all_signatures or data_is_empty:
        logger.warning("[Organizer] No signatures or data provided to organize. Returning original set.")
        return all_signatures

    initial_count = len(all_signatures)
    logger.info(f"--- Starting Signature Organization for {initial_count} signatures (Turn: {turn_counter}) ---")

    # --- NEW: Intermediate Caching Logic ---
    # Define a temporary file path for the performance results
    cache_dir = os.path.join(run_dir, "organizer_cache")
    os.makedirs(cache_dir, exist_ok=True)
    perf_cache_filename = f"perf_results_turn_{turn_counter}.pkl"
    perf_cache_path = os.path.join(cache_dir, perf_cache_filename)

    performance_results = None

    try:
        # 1. Calculate or Load Performance Results
        if os.path.exists(perf_cache_path):
            try:
                logger.info(f"[Organizer] Found performance cache file. Loading from: {perf_cache_path}")
                with open(perf_cache_path, 'rb') as f:
                    performance_results = pickle.load(f)
                logger.info("[Organizer] Successfully loaded performance results from cache.")
            except Exception as e:
                logger.warning(f"[Organizer] Failed to load from cache file ({e}). Recalculating...")

        if performance_results is None:
            signatures_for_perf_calc = [
                {'id': sig_id, 'name': f'Sig_{sig_id}', 'rule_dict': rule}
                for sig_id, rule in all_signatures.items()
            ]
            performance_results = calculate_signatures_performance(
                signatures_for_perf_calc,
                data_source=data_provider,
                num_processes=num_processes,
                data_batch_size=data_batch_size
            )
            
            try:
                logger.info(f"[Organizer] Saving performance results to cache: {perf_cache_path}")
                with open(perf_cache_path, 'wb') as f:
                    pickle.dump(performance_results, f)
            except Exception as e:
                logger.error(f"[Organizer] Failed to save performance results to cache: {e}")

        current_signatures = all_signatures.copy()

        # 1.a Initial pruning before merging to guarantee monotonic reduction
        logger.info("[Organizer] Running initial pruning (pre-merge) to remove redundant signatures.")
        redundant_ids_initial = find_redundant_signatures(current_signatures, performance_results, coverage_threshold)
        if redundant_ids_initial:
            logger.info(f"[Organizer] Initial pruning removed {len(redundant_ids_initial)} signatures before merging.")
            for sig_id in redundant_ids_initial:
                current_signatures.pop(sig_id, None)
                performance_results.pop(sig_id, None)

        # 2. Merging Step
        if enable_merging:
            newly_merged_rules, ids_to_remove_from_merge = merge_similar_signatures(
                current_signatures,
                performance_results,
                merge_infrequent_threshold
            )

            if ids_to_remove_from_merge:
                logger.info(f"[Organizer] Removing {len(ids_to_remove_from_merge)} original signatures that were merged.")
                # Remove the old rules that were merged
                for sig_id in ids_to_remove_from_merge:
                    current_signatures.pop(sig_id, None)
                    performance_results.pop(sig_id, None)  # Also remove their performance data

            if newly_merged_rules:
                logger.info(f"[Organizer] Adding {len(newly_merged_rules)} new merged signatures.")
                # Add the new merged rules
                current_signatures.update(newly_merged_rules)

                # We must calculate performance for these new rules before proceeding to pruning
                logger.info("[Organizer] Recalculating performance for new merged signatures...")
                new_sigs_for_perf = [{'id': sid, 'name': f'Sig_{sid}', 'rule_dict': r} for sid, r in newly_merged_rules.items()]
                new_perf_results = calculate_signatures_performance(
                    new_sigs_for_perf,
                    data_source=data_provider,
                    num_processes=num_processes,
                    data_batch_size=data_batch_size
                )
                performance_results.update(new_perf_results)

        # 3. Pruning (Subsumption) Step - run AFTER merging to ensure counts never grow
        redundant_ids = find_redundant_signatures(current_signatures, performance_results, coverage_threshold)

        # 4. Create the new, pruned dictionary of signatures by removing the redundant ones
        pruned_signatures = {
            sig_id: rule
            for sig_id, rule in current_signatures.items()
            if sig_id not in redundant_ids
        }
        
        final_count = len(pruned_signatures)
        pruned_count = initial_count - final_count
        
        logger.info(f"--- Signature Organization Complete ---")
        logger.info(f"Final signature count: {final_count} (Initial: {initial_count}, Pruned: {pruned_count}).")

        # Small GC hint for very large workloads
        try:
            performance_results = None
            current_signatures = None
            newly_merged_rules = None
            new_sigs_for_perf = None
            new_perf_results = None
        except UnboundLocalError:
            pass
        gc.collect()

        return pruned_signatures

    finally:
        # --- NEW: Cleanup Logic ---
        # On successful completion of this function, remove the temporary cache file.
        # If the script crashes, this file will remain for the next run.
        if os.path.exists(perf_cache_path):
            try:
                logger.info(f"[Organizer] Cleaning up temporary cache file: {perf_cache_path}")
                os.remove(perf_cache_path)
            except OSError as e:
                logger.error(f"[Organizer] Error removing cache file: {e}")
