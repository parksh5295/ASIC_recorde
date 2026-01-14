import pandas as pd
import argparse
import time
import os
import sys
import multiprocessing
from datetime import datetime
import copy # --- copy module for deepcopy ---
import numpy as np # --- numpy for direct mapping ---
import re # --- re for robust interval parsing ---
import signal # --- Import signal to handle BrokenPipeError gracefully ---
import shutil # --- Import for directory deletion ---
import gc  # --- Import gc to allow explicit garbage collection ---

import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


LEVEL_LIMITS_BY_FILE_TYPE = {
    'MiraiBotnet': 5,
    'NSL-KDD': 3,
    'NSL_KDD': 3,
    'DARPA98': 10,
    'DARPA': 10,
    'CICIDS2017': 3,
    'CICIDS': 3,
    'CICModbus23': None,
    'CICModbus': None,
    'IoTID20': None,
    'IoTID': None,
    'CICIoT': 3,
    'CICIoT2023': 3,
    'netML': 3,
    'Kitsune': 3,
    'default': None
}

# === START: Project Root Path Correction ===
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    if '..' not in sys.path:
        sys.path.insert(0, '..')
# === END: Project Root Path Correction ===

from Dataset_Choose_Rule.dtype_optimize import load_csv_safely
from utils.remove_rare_columns import remove_rare_columns # Import the function
from Heterogeneous_Method.Feature_Encoding import Heterogeneous_Feature_named_featrues # --- Import for protection list ---
from Dataset_Choose_Rule.isv_save_log import get_params_str # --- Import for logging (itemset, rule saving) ---
from Dataset_Choose_Rule.isv_checkpoint_handler import save_checkpoint, load_checkpoint # --- Import for checkpointing ---
from Dataset_Choose_Rule.dataset_window_preset import get_temporal_chunk_size
from Modules.Signature_Organize import organize_signatures # --- Import for signature pruning ---
from utils.chunk_cache import ChunkCache, _yield_batches_from_source
from utils.rule_spooler import RuleSpooler
from utils.isv_filtering import (
    resolve_rule_spool_settings,
    calculate_and_log_support_stats,
    calculate_support_for_itemset,
    init_worker_filter_batch,
    is_rule_valid_for_filtering_batch,
    init_worker_filter,
    is_rule_valid_for_filtering
)
from utils.support_dealing import get_dominant_columns, build_rules_from_dominant
from utils.class_row import get_label_columns_to_exclude
from Signature_tool.signature_reduction import reduce_signatures_by_subsets
try:
    from Evaluation.calculate_signature import calculate_signature
except ImportError:
    logger.warning("Could not import calculate_signature. Individual signature metrics will not be available.")
    def calculate_signature(data, signatures):
        return []

try:
    from Dataset_Choose_Rule.association_data_choose import get_clustered_data_path # MODIFIED
    from Dataset_Choose_Rule.choose_amount_dataset import file_cut
    from definition.Anomal_Judgment import anomal_judgment_label, anomal_judgment_nonlabel
    from utils.time_transfer import time_scalar_transfer
    from Modules.Heterogeneous_module import choose_heterogeneous_method
    from Heterogeneous_Method.separate_group_mapping import map_intervals_to_groups
    from Modules.Association_module import association_module
    from Modules.Difference_sets import dict_list_difference
    from Rebuild_Method.FalsePositive_Check import apply_signatures_to_dataset, evaluate_false_positives, summarize_fp_results
except ImportError as e:
    print(f"Warning: Could not import all project modules: {e}. Some functionalities might be limited.")
    def association_module(df, *args, **kwargs):
        print("WARNING: Using dummy 'association_module'.")
        if not df.empty:
            rule = {col: val for col, val in df.iloc[0].items() if pd.notna(val)}
            return [rule] if rule else []
        return []
    def apply_signatures_to_dataset(df, sigs):
        print("WARNING: Using dummy 'apply_signatures_to_dataset'.")
        return pd.DataFrame()
    def load_csv_safely(file_type, path):
        print("WARNING: Using dummy 'load_csv_safely'. Attempting pd.read_csv directly.")
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            return None

try:
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.set_loglevel("warning")
except ImportError:
    print("Warning: matplotlib is not installed. Plotting functionality will be disabled.")
    plt = None


# --- Logger Setup ---
logger = logging.getLogger(__name__)
# MODIFIED: Set level to DEBUG to see detailed logs
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def _suppress_broken_pipe_tracebacks():
    """Prevent BrokenPipeError tracebacks from noisy worker exits."""
    def _handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, BrokenPipeError):
            return
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    sys.excepthook = _handler

def log_dataframe_debug_info(df, name="DataFrame"):
    """Logs detailed debug information about a DataFrame."""
    if df is None or df.empty:
        logger.debug(f"--- {name} Info: DataFrame is empty or None. ---")
        return
        
    logger.debug(f"--- START: {name} Info ---")
    logger.debug(f"Shape: {df.shape}")
    logger.debug(f"Columns: {df.columns.tolist()}")
    logger.debug(f"Data Types:\n{df.dtypes.to_string()}")
    logger.debug(f"Head:\n{df.head().to_string()}")
    logger.debug(f"--- END: {name} Info ---")

def calculate_performance_in_batches(data_source, signatures, batch_size):
    """
    Calculates performance metrics by streaming batches from the provided data source.
    Returns alert results to identify which signatures triggered alerts.
    """
    if not signatures or data_source is None:
        return 0, 0, 0, 0, 0, 0, pd.DataFrame() # tp, fp, recall, precision, f1, accuracy, alerts_df

    all_alerted_indices = set()
    actual_positives_indices = set()
    actual_negatives_indices = set()
    all_alerts = []  # Collect all alerts to track which signatures triggered them

    total_rows = 0
    batch_counter = 0

    for batch_df in _yield_batches_from_source(data_source, batch_size):
        batch_counter += 1
        total_rows += len(batch_df)

        alerts_batch = apply_signatures_to_dataset(batch_df, signatures)
        if not alerts_batch.empty:
            all_alerted_indices.update(alerts_batch['alert_index'].unique())
            all_alerts.append(alerts_batch)

        actual_positives_indices.update(batch_df[batch_df['label'] == 1].index)
        actual_negatives_indices.update(batch_df[batch_df['label'] == 0].index)

    if total_rows == 0:
        return 0, 0, 0, 0, 0, 0, pd.DataFrame()

    logger.info(f"  [Perf Eval] Processed {total_rows} rows across {batch_counter} streamed batch(es).")

    # Combine all alerts
    alerts_df = pd.concat(all_alerts, ignore_index=True) if all_alerts else pd.DataFrame()

    tp = len(all_alerted_indices.intersection(actual_positives_indices))
    fp = len(all_alerted_indices.intersection(actual_negatives_indices))
    total_negatives = len(actual_negatives_indices)
    tn = total_negatives - fp
    
    total_anomalies = len(actual_positives_indices)
    total_alerts = len(all_alerted_indices)

    recall = tp / total_anomalies if total_anomalies > 0 else 0
    precision = tp / total_alerts if total_alerts > 0 else 0
    f1 = 0
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    accuracy = (tp + tn) / total_rows if total_rows > 0 else 0
        
    return tp, fp, recall, precision, f1, accuracy, alerts_df


# --- START: New Helper function for parallel filtering ---
# This global tuple will hold the data needed by the worker processes.
# It's a workaround to avoid passing large dataframes to each process,
# which can be slow due to serialization (pickling).

def preprocess_and_map_chunk(chunk_df, file_type, category_mapping, data_list):
    """
    Applies robust labeling and then maps a data chunk.
    NOTE: Time-based feature conversion should be done BEFORE calling this function.
    """
    # MODIFIED: The labeling logic is removed. The loaded data already has 'label' and 'cluster'.
    # This function now only applies the mapping.

    # --- START: Replace mapping logic with Direct Mapping from ISV_ex.py ---
    mapped_chunk = chunk_df.copy()
    
    # Apply interval mapping directly for each column that has a rule
    if 'interval' in category_mapping and isinstance(category_mapping['interval'], pd.DataFrame):
        interval_rules = category_mapping['interval']
        for col in mapped_chunk.columns:
            if col in interval_rules.columns:
                # Create a temporary series for mapping to avoid chained assignment warnings
                mapped_col = pd.Series(np.nan, index=mapped_chunk.index)
                
                # --- NEW: Robust Interval Parsing Logic ---
                # This logic parses the mapping rules and applies them correctly
                intervals = []
                groups = []
                for rule_str in interval_rules[col].dropna():
                    try:
                        interval_part, group_str = rule_str.split('=')
                        group = int(group_str)
                        
                        # Use regex to find all numbers in the interval string
                        nums = [float(n) for n in re.findall(r'-?\\d+\\.?\\d*', interval_part)]
                        if len(nums) != 2: continue
                        
                        # Determine if the interval is closed on the left/right
                        closed_left = interval_part.strip().startswith('[')
                        closed_right = interval_part.strip().endswith(']')
                        
                        intervals.append((nums[0], nums[1], closed_left, closed_right))
                        groups.append(group)
                    except (ValueError, IndexError):
                        continue

                # Apply all parsed rules at once using vectorization for performance
                numeric_col = pd.to_numeric(mapped_chunk[col], errors='coerce').fillna(0)
                
                # Default value if no rule matches
                # Find the max group number to determine a safe default
                default_group = max(groups) + 1 if groups else 0

                # Iterate backwards to ensure correct priority for overlapping intervals (though unlikely)
                for i in range(len(intervals) - 1, -1, -1):
                    lower, upper, closed_l, closed_r = intervals[i]
                    group = groups[i]
                    
                    # Build boolean mask based on interval boundaries
                    if closed_l and closed_r:
                        mask = (numeric_col >= lower) & (numeric_col <= upper)
                    elif closed_l:
                        mask = (numeric_col >= lower) & (numeric_col < upper)
                    elif closed_r:
                        mask = (numeric_col > lower) & (numeric_col <= upper)
                    else:
                        mask = (numeric_col > lower) & (numeric_col < upper)
                    
                    mapped_col.loc[mask] = group

                # Fill any remaining NaNs with the default group and update the DataFrame column
                mapped_chunk[col] = mapped_col.fillna(default_group).astype(int)

    # The label is already in mapped_chunk from the copy, so no need to re-add
    return mapped_chunk

def main(args):
    # --- NEW: Top-level try-except block to catch the root cause of process failures ---
    try:
        start_time = time.time()
        logger.info("--- Initial Setup: Generating Mapping On-the-fly ---")

        rule_spool_chunk_size_runtime, rule_spool_force_flush_threshold = resolve_rule_spool_settings(args.file_type, args.rule_spool_chunk_size)
        logger.info(f"[RuleSpool] chunk_size={rule_spool_chunk_size_runtime}, flush_threshold={rule_spool_force_flush_threshold}")

        negative_filtering_enabled = bool(args.negative_filtering)
        negative_filter_threshold = max(0.0, args.negative_filter_threshold)
        normal_min_support = args.normal_min_support if args.normal_min_support is not None else args.min_support
        if negative_filtering_enabled:
            if negative_filter_threshold >= normal_min_support:
                logger.warning(f"[NegativeFilter] Threshold ({negative_filter_threshold:.4f}) >= normal_min_support ({normal_min_support:.4f}). Consider lowering for stricter filtering.")
            logger.info(f"[NegativeFilter] Enabled with threshold {negative_filter_threshold:.4f}.")

        # --- Temporal window preset for chunk_size ---
        if args.cstemporal:
            preset_cs = get_temporal_chunk_size(args.file_type)
            if preset_cs:
                logger.info(f"[TemporalWindow] Using preset chunk_size={preset_cs} for {args.file_type}")
                args.chunk_size = preset_cs
                args.chunk_size_label = "tem"
            else:
                logger.warning(f"[TemporalWindow] No preset found for {args.file_type}; keeping chunk_size={args.chunk_size}")
                args.chunk_size_label = None
        
        # --- NEW: Get max_level for filename ---
        max_level = LEVEL_LIMITS_BY_FILE_TYPE.get(args.file_type, LEVEL_LIMITS_BY_FILE_TYPE['default'])

        # --- Build turn range tag for filename ---
        start_label = args.run_turn_start if args.run_turn_start is not None else "NA"
        end_label = args.run_turn_end if args.run_turn_end is not None else "NA"
        turn_range_tag = f"ts{start_label}-te{end_label}"

        # --- Parameter and Path Setup ---
        params_str_base = get_params_str(args, max_level)
        dom_parts = []
        if args.dominant_min_support is not None:
            dom_parts.append(f"ms={args.dominant_min_support}")
        if args.dominant_min_confidence is not None:
            dom_parts.append(f"mc={args.dominant_min_confidence}")
        if args.dominant_normal_min_support is not None:
            dom_parts.append(f"nms={args.dominant_normal_min_support}")
        if args.dominant_level is not None:
            dom_parts.append(f"lvl={args.dominant_level}")
        dom_suffix = ""
        if dom_parts:
            dom_suffix = "_dom(" + "_".join(dom_parts) + ")"
        params_str = f"{params_str_base}_n{args.n_splits}_dom{args.dominant_freq_threshold}{dom_suffix}_turneval_tr{turn_range_tag}"
        run_dir = os.path.join("../Dataset_ISV", args.file_type, params_str) # Used for artifacts and checkpoints
        chunk_cache_dir = os.path.join(run_dir, "chunk_cache")
        chunk_cache = ChunkCache(chunk_cache_dir)
        
        # --- Reset Logic ---
        if args.reset:
            if os.path.isdir(run_dir):
                logger.info(f"--reset flag is set. Deleting directory: {run_dir}")
                try:
                    shutil.rmtree(run_dir)
                    logger.info(f"Successfully deleted directory.")
                except OSError as e:
                    logger.error(f"Error deleting directory {run_dir}: {e}")
            else:
                logger.info(f"--reset flag is set, but the target directory does not exist: {run_dir}")
            
            # Exit the script gracefully after handling the reset operation.
            sys.exit(0)
        
        # --- Load Checkpoint ---
        resume_from_turn = 0
        all_valid_signatures = {}
        signatures_to_remove = set()
        history = []
        signature_turn_created = {}
        
        checkpoint_data = load_checkpoint(run_dir)
        if checkpoint_data:
            resume_from_turn = checkpoint_data['resume_from_turn']
            all_valid_signatures = checkpoint_data['all_valid_signatures']
            signatures_to_remove = checkpoint_data['signatures_to_remove']
            history = checkpoint_data['history']
            signature_turn_created = checkpoint_data.get('signature_turn_created', {})

        # --- Data Loading and Initial Mapping (ALWAYS RUNS) ---
        file_path, total_rows = get_clustered_data_path(args.file_type, args.file_number)
        
        try:
            # We need a fresh iterator for the main loop, so we only read the first chunk here
            first_chunk_df = pd.read_csv(file_path, nrows=args.chunk_size, low_memory=False)
            if first_chunk_df.empty:
                logger.error(f"File {file_path} appears to be empty.")
                return
        except (StopIteration, FileNotFoundError, pd.errors.EmptyDataError) as e:
            logger.error(f"Could not read the first chunk from {file_path}: {e}")
            return

        processed_first_chunk = time_scalar_transfer(first_chunk_df.copy(), args.file_type)
        
        # MODIFIED: The extensive labeling logic is removed. 
        # The 'label' for evaluation and 'cluster' for association are already present in the loaded data.
        logger.info("Labeling is skipped as 'label' and 'cluster' columns are pre-loaded.")

        # Generate mapping rules from the fully preprocessed first chunk
        # MODIFIED: Drop all label-related columns before creating mapping rules
        label_cols_to_exclude = get_label_columns_to_exclude(args.file_type)
        mapping_features = processed_first_chunk.drop(columns=label_cols_to_exclude, errors='ignore')
        _, _, category_mapping, _ = choose_heterogeneous_method(
            mapping_features,
            args.file_type, 'Interval_inverse', 'N', n_splits_override=args.n_splits
        )
        
        if not category_mapping or 'interval' not in category_mapping or category_mapping['interval'].empty:
            logger.error("FATAL: Failed to generate a valid category mapping.")
            return

        logger.info(f"Successfully generated new mapping with {len(category_mapping['interval'].columns)} features.")
        
        # This data_list is for the old hunter/prey logic, which is now simplified.
        # We keep it for preprocess_and_map_chunk's signature but it's not used for state.
        data_list = [pd.DataFrame(), pd.DataFrame()]

        # --- State Re-creation and Fast-Forward ---
        #processed_data_so_far = pd.DataFrame()
        main_chunk_iterator = pd.read_csv(file_path, chunksize=args.chunk_size, low_memory=False)
        
        if resume_from_turn > 0:
            logger.info(f"Fast-forwarding to turn {resume_from_turn + 1}. Rebuilding cumulative data...")
            ff_turn_counter = 0
            for chunk in main_chunk_iterator:
                ff_turn_counter += 1
                if ff_turn_counter > resume_from_turn:
                    # We've reached the chunk where we need to resume processing.
                    # Break the loop so the main loop can start with this chunk.
                    # To do this, we need to restructure the loop.
                    break 

                # Apply the necessary preprocessing to rebuild the streamed cache state
                processed_chunk = time_scalar_transfer(chunk, args.file_type)
                mapped_chunk = preprocess_and_map_chunk(processed_chunk, args.file_type, category_mapping, data_list)
                chunk_cache.register_chunk(mapped_chunk, ff_turn_counter)
            
            logger.info(f"Fast-forward complete. {chunk_cache.total_rows} rows of prior data rebuilt.")
        
        # --- NEW: Initial Pruning after Checkpoint Load ---
        if args.signature_organize and all_valid_signatures and not chunk_cache.is_empty():
            logger.info("--- Performing Initial Signature Organization after loading checkpoint ---")
            
            all_valid_signatures = organize_signatures(
                all_signatures=all_valid_signatures,
                data_provider=chunk_cache,
                data_batch_size=args.evaluation_batch_size,
                num_processes=args.num_processes,
                run_dir=run_dir,
                turn_counter=0, # Use 0 for the initial organization step
                coverage_threshold=args.prune_coverage_threshold,
                enable_merging=args.merge_signatures,
                merge_infrequent_threshold=args.merge_infrequent_threshold
            )
            
            # Update the blacklist to remove any pruned signatures that were also blacklisted
            blacklisted_and_pruned = signatures_to_remove - set(all_valid_signatures.keys())
            if blacklisted_and_pruned:
                signatures_to_remove = signatures_to_remove.intersection(set(all_valid_signatures.keys()))
                logger.info(f"Removed {len(blacklisted_and_pruned)} signatures from the blacklist as they were pruned.")

        # --- Main Loop ---
        # Re-initialize iterator to start from the beginning for the main loop logic
        main_chunk_iterator = pd.read_csv(file_path, chunksize=args.chunk_size, low_memory=False)
        turn_counter = 0

        for chunk in main_chunk_iterator:
            turn_counter += 1

            # --- TURN RANGE CONTROL (for stopping) ---
            if args.run_turn_end is not None and turn_counter > args.run_turn_end:
                logger.info(f"Reached run_turn_end={args.run_turn_end}. Stop experiment.")
                break
            
            # --- Preprocessing for the Current Chunk (ALWAYS for state) ---
            processed_chunk = time_scalar_transfer(chunk, args.file_type)
            mapped_chunk = preprocess_and_map_chunk(processed_chunk, args.file_type, category_mapping, data_list)

            # Persist the mapped chunk for future streaming-based evaluations (ALWAYS needed for state)
            mapped_chunk = chunk_cache.register_chunk(mapped_chunk, turn_counter)
            
            # --- TURN RANGE CONTROL (for processing) ---
            # Skip actual processing if outside turn range, but keep data in cache for state
            if args.run_turn_start is not None and turn_counter < args.run_turn_start:
                continue

            # --- 1. Skip or Process (checkpoint resume) ---
            if turn_counter <= resume_from_turn:
                # Rebuild state (already done in fast-forward, but need to concat the first chunk if turn 1 is skipped)
                # This is getting complicated. Let's simplify the loop structure.
                continue # Skip processing for turns already completed.

            # The loop starts from the next turn to process
            logger.info(f"--- Processing Turn {turn_counter} (Rows {(turn_counter-1)*args.chunk_size + 1} - {turn_counter*args.chunk_size}) ---")
            # NEW: keep an in-memory view for per-turn entry/exit evaluation
            turn_eval_data = mapped_chunk.copy()
        
            if turn_counter == 1:
                log_dataframe_debug_info(mapped_chunk, "First Chunk (After Mapping)")

            #processed_data_so_far = pd.concat([processed_data_so_far, mapped_chunk], ignore_index=True)
            
            # ... (The rest of the main loop logic for processing `mapped_chunk` remains the same) ...
            # [This includes blacklist reset, performance evaluation, validation, hunter, generation, etc.]
            # --- Blacklist Reset Logic ---
            if turn_counter > 1 and (turn_counter - 1) % 2 == 0:
                if signatures_to_remove:
                    logger.warning(f"*** Resetting blacklist at the start of Turn {turn_counter}. "
                                   f"Removing {len(signatures_to_remove)} signatures from the blacklist. ***")
                    signatures_to_remove.clear()
                else:
                    logger.info(f"*** Blacklist reset point at Turn {turn_counter}, but it was already empty. ***")

            # --- 0. (NEW) Entry Performance Evaluation ---
            entry_alerts = pd.DataFrame() # FIX: Initialize to prevent UnboundLocalError
            entry_recall, entry_precision, entry_f1, entry_accuracy = 0, 0, 0, 0
            rules_at_turn_start = {sig_id: rule for sig_id, rule in all_valid_signatures.items() if sig_id not in signatures_to_remove}
            entry_sig_count = len(rules_at_turn_start)
            if rules_at_turn_start and turn_eval_data is not None and not turn_eval_data.empty:
                # Sort by signature ID to ensure consistent ordering (for "first match wins" logic)
                sorted_entry_sigs = sorted(rules_at_turn_start.items(), key=lambda x: x[0])
                formatted_entry_sigs = [{'id': sid, 'name': f'Sig_{sid}', 'rule_dict': r} for sid, r in sorted_entry_sigs]
                
                # --- MODIFIED: Use batched performance calculation ---
                _, _, entry_recall, entry_precision, entry_f1, entry_accuracy, _ = calculate_performance_in_batches(
                    turn_eval_data, 
                    formatted_entry_sigs, 
                    args.evaluation_batch_size
                )
                
                logger.info(f"Turn {turn_counter} ENTRY Performance - Recall: {entry_recall:.4f}, Precision: {entry_precision:.4f}, F1: {entry_f1:.4f}, Accuracy: {entry_accuracy:.4f}")

            # MODIFIED: Split chunk data based on the 'cluster' column
            # Get label columns to exclude for this dataset
            label_cols_to_exclude = get_label_columns_to_exclude(args.file_type)
            normal_data_in_chunk = mapped_chunk[mapped_chunk['cluster'] == 0].copy().drop(columns=label_cols_to_exclude, errors='ignore')
            # IMPORTANT: Drop label columns immediately to prevent label leakage into signatures
            anomalous_data_in_chunk = mapped_chunk[mapped_chunk['cluster'] == 1].copy().drop(columns=label_cols_to_exclude, errors='ignore')
            # Release the mapped chunk reference early to reduce peak memory per turn
            del mapped_chunk
            gc.collect()

            # --- 1. Validation Step (FP Detection) ---
            newly_removed_count = 0
            newly_flagged_for_removal = set() # Store IDs of signatures removed THIS turn
            if not args.disable_fp_removal and all_valid_signatures and not normal_data_in_chunk.empty:
                signatures_to_test = [ {'id': sig_id, 'name': f'Sig_{sig_id}', 'rule_dict': rule} for sig_id, rule in all_valid_signatures.items() if sig_id not in signatures_to_remove ]
                
                if signatures_to_test:
                    if args.use_fp_metrics:
                        # --- NRA/UFP/HAF based FP removal ---
                        logger.info(f"[FP Metrics] Validating {len(signatures_to_test)} existing signatures using NRA/UFP/HAF metrics against {len(normal_data_in_chunk)} normal data rows...")
                        
                        # Batch process signatures to generate alerts
                        all_fp_alerts = []
                        batch_size = args.signature_batch_size
                        num_batches = (len(signatures_to_test) + batch_size - 1) // batch_size
                        
                        logger.info(f"[FP Metrics] Processing signatures in {num_batches} batches of size {batch_size} to generate alerts.")
                        
                        for i in range(0, len(signatures_to_test), batch_size):
                            batch_signatures = signatures_to_test[i:i+batch_size]
                            logger.debug(f"  [FP Metrics] Validating batch {i//batch_size + 1}/{num_batches}...")
                            
                            fp_alerts_batch = apply_signatures_to_dataset(normal_data_in_chunk, batch_signatures)
                            
                            if not fp_alerts_batch.empty:
                                all_fp_alerts.append(fp_alerts_batch)
                        
                        if all_fp_alerts:
                            fp_alerts = pd.concat(all_fp_alerts, ignore_index=True)
                            
                            # Build current_signatures_map for evaluate_false_positives
                            current_signatures_map = {sig['id']: sig for sig in signatures_to_test}
                            
                            # Evaluate FP scores using NRA/UFP/HAF metrics
                            logger.info(f"[FP Metrics] Calculating NRA/UFP/HAF scores for {len(fp_alerts)} alerts...")
                            detailed_fp_results = evaluate_false_positives(
                                alerts_df=fp_alerts,
                                current_signatures_map=current_signatures_map,
                                known_fp_sig_dicts=None,  # Can be extended to support known FP list
                                attack_free_df=normal_data_in_chunk,
                                t0_nra=args.fp_t0_nra,
                                n0_nra=args.fp_n0_nra,
                                lambda_haf=args.fp_lambda_haf,
                                lambda_ufp=args.fp_lambda_ufp,
                                combine_method=args.fp_combine_method,
                                belief_threshold=args.fp_belief_threshold,
                                superset_strictness=args.fp_superset_strictness,
                                file_type=args.file_type
                            )
                            
                            # Summarize FP results to get final FP decision per signature
                            logger.info(f"[FP Metrics] Summarizing FP results...")
                            fp_summary = summarize_fp_results(detailed_fp_results)
                            
                            if not fp_summary.empty:
                                # Get signatures marked as likely FP (final_likely_fp == True)
                                high_fp_signatures = fp_summary[fp_summary['final_likely_fp'] == True]
                                flagged_for_removal = set(high_fp_signatures['signature_id'].unique())
                                
                                newly_flagged_for_removal = flagged_for_removal - signatures_to_remove
                                newly_removed_count = len(newly_flagged_for_removal)
                                
                                if newly_removed_count > 0:
                                    logger.warning(f"[FP Metrics] Found {newly_removed_count} new signatures marked as FP by NRA/UFP/HAF metrics. Flagging for removal.")
                                    logger.info(f"[FP Metrics] FP Summary: {len(high_fp_signatures)} signatures marked as FP out of {len(fp_summary)} total evaluated.")
                                    signatures_to_remove.update(newly_flagged_for_removal)
                                else:
                                    logger.info(f"[FP Metrics] No new FP signatures detected. All existing signatures passed NRA/UFP/HAF evaluation.")
                            
                            # Explicitly free temporary validation structures
                            try:
                                del all_fp_alerts
                                del fp_alerts
                                del detailed_fp_results
                                del fp_summary
                            except NameError:
                                pass
                            gc.collect()
                        else:
                            logger.info(f"[FP Metrics] No alerts generated on normal data. All signatures passed initial check.")
                    else:
                        # --- Simple FP removal (original method) ---
                        logger.info(f"Validating {len(signatures_to_test)} existing signatures against {len(normal_data_in_chunk)} normal data rows (simple method)...")
                        
                        all_fp_alerts = []
                        batch_size = args.signature_batch_size
                        num_batches = (len(signatures_to_test) + batch_size - 1) // batch_size
                        
                        logger.info(f"Processing signatures in {num_batches} batches of size {batch_size}.")

                        for i in range(0, len(signatures_to_test), batch_size):
                            batch_signatures = signatures_to_test[i:i+batch_size]
                            logger.debug(f"  Validating batch {i//batch_size + 1}/{num_batches}...")
                            
                            fp_alerts_batch = apply_signatures_to_dataset(normal_data_in_chunk, batch_signatures)
                            
                            if not fp_alerts_batch.empty:
                                all_fp_alerts.append(fp_alerts_batch)

                        if all_fp_alerts:
                            fp_alerts = pd.concat(all_fp_alerts, ignore_index=True)
                            flagged_for_removal = set(fp_alerts['signature_id'].unique())
                            newly_flagged_for_removal = flagged_for_removal - signatures_to_remove
                            newly_removed_count = len(newly_flagged_for_removal)
                            if newly_removed_count > 0:
                                logger.warning(f"Found {newly_removed_count} new signatures causing FPs. Flagging for removal.")
                                signatures_to_remove.update(newly_flagged_for_removal)

                        # Explicitly free temporary validation structures
                        try:
                            del all_fp_alerts
                        except NameError:
                            pass
                        try:
                            del fp_alerts
                        except NameError:
                            pass
                        gc.collect()
            elif args.disable_fp_removal:
                logger.info(f"[FP Removal] FP removal is disabled. Skipping validation step.")

            rule_spooler = RuleSpooler(run_dir, turn_counter, chunk_size=rule_spool_chunk_size_runtime)
            new_signatures_found = 0
            try:
                # --- 2. Generation Step (Spooling) ---
                if not anomalous_data_in_chunk.empty:
                    logger.info(f"Generating new candidate rules from {len(anomalous_data_in_chunk)} anomalous data rows...")
                    
                    # Identify dominant (near-constant) columns and mask them for support counting
                    # NOTE: anomalous_data_in_chunk already has label columns excluded at line 526
                    dominant_cols = {}
                    mask_skipped = False
                    if args.mask_dominant_cols:
                        dominant_cols = get_dominant_columns(anomalous_data_in_chunk, freq_threshold=args.dominant_freq_threshold)
                        if dominant_cols:
                            logger.info(f"[DominantCols] Masking {len(dominant_cols)} near-constant columns for support counting: {list(dominant_cols.keys())}")

                    anomalous_for_mining = anomalous_data_in_chunk.drop(columns=dominant_cols.keys(), errors='ignore') if dominant_cols else anomalous_data_in_chunk
                    # label_cols_to_exclude already defined above (line 523)
                    
                    # Guard: if masking removed everything (or nearly everything), skip masking
                    # NOTE: anomalous_data_in_chunk and anomalous_for_mining already have label columns excluded
                    if args.mask_dominant_cols and dominant_cols and anomalous_for_mining.shape[1] == 0:
                        # Check if all columns are truly constant (support=1). If yes, keep masking.
                        cols_for_check = list(anomalous_data_in_chunk.columns)
                        all_constant = True
                        for c in cols_for_check:
                            if anomalous_data_in_chunk[c].nunique(dropna=False) > 1:
                                all_constant = False
                                break
                        if all_constant:
                            logger.warning("[DominantCols] All features are constant (support=1); keeping masking despite empty remainder.")
                        else:
                            logger.warning("[DominantCols] Masking removed all feature columns; skipping masking for this turn.")
                            dominant_cols = {}
                            anomalous_for_mining = anomalous_data_in_chunk
                            mask_skipped = True
                        '''
                        else:
                            logger.warning("[DominantCols] All features masked for Kitsune; keeping masking (no skip).")
                        '''

                    # Adjust thresholds when masking is skipped
                    min_support_eff = args.min_support
                    min_conf_eff = args.min_confidence
                    normal_min_support_effective = normal_min_support
                    if mask_skipped:
                        if args.dominant_min_support is not None:
                            min_support_eff = args.dominant_min_support
                        if args.dominant_min_confidence is not None:
                            min_conf_eff = args.dominant_min_confidence
                        if args.dominant_normal_min_support is not None:
                            normal_min_support_effective = args.dominant_normal_min_support
                    
                    # NOTE: anomalous_for_mining already has label columns excluded
                    calculate_and_log_support_stats(
                        anomalous_for_mining,
                        min_support_eff,
                        turn_counter
                    )

                    logger.debug(f"  [Association Params] Turn: {turn_counter}, "
                                 f"Anomalous Rows: {len(anomalous_data_in_chunk)}, "
                                 f"min_support: {min_support_eff}, "
                                 f"min_confidence: {min_conf_eff}")

                    max_level = LEVEL_LIMITS_BY_FILE_TYPE.get(args.file_type, LEVEL_LIMITS_BY_FILE_TYPE['default'])
                    if mask_skipped and args.dominant_level is not None:
                        max_level = args.dominant_level
                    
                    assoc_args = {
                        'association_rule_choose': args.association_method,
                        'min_support': min_support_eff,
                        'min_confidence': min_conf_eff,
                        'association_metric': 'confidence',
                        'num_processes': args.num_processes,
                        'file_type_for_limit': args.file_type,
                        'max_level_limit': max_level,
                        'itemset_limit': args.itemset_limit
                    }
                    if args.save_artifacts:
                        assoc_args['turn_counter'] = turn_counter
                        assoc_args['params_str'] = params_str
                    
                    if args.dynamic_support:
                        assoc_args['enable_dynamic_support'] = True
                        assoc_args['dynamic_support_threshold'] = args.itemset_count_threshold
                        assoc_args['support_increment_factor'] = args.support_increment_factor

                    # NOTE: anomalous_for_mining already has label columns excluded at line 526
                    standard_rules, _ = association_module(
                        anomalous_for_mining,
                        **assoc_args
                    )
                    # Re-attach dominant columns (near-constant) so rules include them
                    if standard_rules and dominant_cols:
                        for rule in standard_rules:
                            for d_col, d_val in dominant_cols.items():
                                rule.setdefault(d_col, d_val)
                    # Fallback: if no rules were generated, create minimal rules from dominant cols
                    is_fallback_rule = False
                    if (not standard_rules or len(standard_rules) == 0) and dominant_cols:
                        standard_rules = build_rules_from_dominant(dominant_cols)
                        is_fallback_rule = True
                        logger.info(f"[DominantCols] Created {len(standard_rules)} fallback rule(s) from dominant columns. These will bypass normal_min_support filtering.")
                    if standard_rules:
                        rule_spooler.add_rules(standard_rules)
                        logger.info(f"[RuleSpool] Queued {len(standard_rules)} rules for filtering.")
                        if rule_spool_force_flush_threshold and rule_spooler.buffer_length() >= rule_spool_force_flush_threshold:
                            logger.debug(f"[RuleSpool] Buffer reached {rule_spooler.buffer_length()} rules. Forcing flush to disk.")
                            rule_spooler.force_flush()
                else:
                    logger.info(f"No anomalous data found in the current chunk. Skipping new rule generation for Turn {turn_counter}.")

                # --- Filtering and Adding new rules from spool ---
                total_spooled = rule_spooler.rule_count()
                if rule_spooler.has_rules():
                    logger.info(f"Filtering {total_spooled} spooled rules in disk-backed batches...")
                    normal_data_empty = normal_data_in_chunk.empty
                    # Prepare memory-only batching, but compute support globally across all chunks
                    if args.normal_data_batch_size and len(normal_data_in_chunk) > args.normal_data_batch_size:
                        logger.info(f"Splitting normal data of size {len(normal_data_in_chunk)} into batches of {args.normal_data_batch_size} (memory only, global support).")
                        normal_data_chunks = [
                            normal_data_in_chunk.iloc[i:i + args.normal_data_batch_size]
                            for i in range(0, len(normal_data_in_chunk), args.normal_data_batch_size)
                        ]
                    else:
                        normal_data_chunks = [normal_data_in_chunk] if not normal_data_empty else []
                    
                    def rule_passes_global(rule, is_fallback=False):
                        """
                        Evaluate a rule against all normal-data chunks, accumulating global support.
                        For fallback rules, also check support in anomaly data to determine if the rule
                        can distinguish between normal and anomaly.
                        
                        Args:
                            rule: The rule dictionary to evaluate
                            is_fallback: If True, this is a fallback rule from dominant columns only.
                        """
                        if normal_data_empty:
                            return True
                        
                        # Calculate support in normal data
                        normal_total_rows = 0
                        normal_total_matches = 0
                        for chunk in normal_data_chunks:
                            if chunk is None or chunk.empty:
                                continue
                            normal_total_rows += len(chunk)
                            
                            rule_keys_in_chunk = [k for k in rule.keys() if k in chunk.columns]
                            if not rule_keys_in_chunk:
                                # No relevant columns in this chunk -> zero matches in this chunk
                                continue
                            
                            rule_series = pd.Series({k: rule[k] for k in rule_keys_in_chunk})
                            match_mask = chunk[rule_keys_in_chunk].eq(rule_series).all(axis=1)
                            normal_total_matches += match_mask.sum()
                        
                        if normal_total_rows == 0:
                            return True  # No normal data present
                        
                        normal_support = normal_total_matches / normal_total_rows
                        
                        # For fallback rules: Check if they can distinguish between normal and anomaly
                        # If support in both normal and anomaly is nearly identical (both close to 1.0),
                        # the rule is useless for detection and should be rejected.
                        if is_fallback:
                            # Calculate support in anomaly data for comparison
                            if not anomalous_data_in_chunk.empty:
                                anomaly_total_rows = len(anomalous_data_in_chunk)
                                rule_keys_in_anomaly = [k for k in rule.keys() if k in anomalous_data_in_chunk.columns]
                                if rule_keys_in_anomaly:
                                    rule_series_anomaly = pd.Series({k: rule[k] for k in rule_keys_in_anomaly})
                                    anomaly_match_mask = anomalous_data_in_chunk[rule_keys_in_anomaly].eq(rule_series_anomaly).all(axis=1)
                                    anomaly_total_matches = anomaly_match_mask.sum()
                                    anomaly_support = anomaly_total_matches / anomaly_total_rows if anomaly_total_rows > 0 else 0.0
                                else:
                                    anomaly_support = 0.0
                            else:
                                anomaly_support = 0.0
                            
                            # Check if the rule can distinguish between normal and anomaly
                            # If both supports are very high (>= 0.95), the rule matches almost everything
                            # and cannot distinguish normal from anomaly
                            if normal_support >= 0.95 and anomaly_support >= 0.95:
                                logger.warning(f"[FallbackRule] Rejected: Cannot distinguish normal from anomaly. "
                                             f"Normal support: {normal_support:.4f}, Anomaly support: {anomaly_support:.4f}. "
                                             f"This rule matches almost all data and is not useful for detection.")
                                return False
                            
                            logger.info(f"[FallbackRule] Accepted: Normal support: {normal_support:.4f}, Anomaly support: {anomaly_support:.4f}. "
                                       f"Rule can distinguish between normal and anomaly.")
                            return True
                        
                        # Negative filtering: applies only to non-fallback rules
                        if negative_filtering_enabled and normal_support > negative_filter_threshold:
                            return False
                        
                        # Standard normal_min_support filtering (only for non-fallback rules)
                        if normal_support > normal_min_support_effective:
                            return False
                        
                        return True
                    
                    def filter_rule_batch(rule_batch, check_fallback=False):
                        if not rule_batch:
                            return []
                        if normal_data_empty:
                            return list(rule_batch)
                        
                        filtered = []
                        for r in rule_batch:
                            # Check if this rule is a fallback rule (matches dominant_cols exactly)
                            is_fallback = False
                            if check_fallback and dominant_cols:
                                # A fallback rule contains only dominant columns
                                rule_cols = set(r.keys())
                                dominant_cols_set = set(dominant_cols.keys())
                                # Check if rule contains only dominant columns (or is a subset)
                                if rule_cols.issubset(dominant_cols_set):
                                    # Verify values match
                                    if all(r.get(col) == dominant_cols.get(col) for col in rule_cols):
                                        is_fallback = True
                            
                            if rule_passes_global(r, is_fallback=is_fallback):
                                filtered.append(r)
                        return filtered

                    total_filtered = 0
                    # Check for fallback rules only if we created fallback rules this turn
                    check_for_fallback = is_fallback_rule
                    for rules_batch in rule_spooler.consume_chunks():
                        filtered_rules = filter_rule_batch(rules_batch, check_fallback=check_for_fallback)
                        total_filtered += len(filtered_rules)
                        for rule in filtered_rules:
                            rule_id = hash(frozenset(rule.items()))
                            if rule_id not in all_valid_signatures and rule_id not in signatures_to_remove:
                                all_valid_signatures[rule_id] = rule
                                signature_turn_created[rule_id] = turn_counter  # Record when this signature was created
                                new_signatures_found += 1
                    logger.info(f"{total_filtered} rules passed filter.")
                    logger.info(f"Added {new_signatures_found} new unique signatures.")
                else:
                    logger.info("No new anomalous rules were generated in this turn.")

            finally:
                rule_spooler.cleanup()
                gc.collect()

            # --- Pruning Step (Signature Organization) ---
            if args.prune_signatures and all_valid_signatures:
                initial_sig_count = len(all_valid_signatures)
                
                # The organize_signatures function will prune based on redundancy against the full dataset so far.
                all_valid_signatures = organize_signatures(
                    all_signatures=all_valid_signatures,
                    data_provider=chunk_cache,
                    data_batch_size=args.evaluation_batch_size,
                    num_processes=args.num_processes,
                    run_dir=run_dir,
                    turn_counter=turn_counter,
                    coverage_threshold=args.prune_coverage_threshold,
                    enable_merging=args.merge_signatures,
                    merge_infrequent_threshold=args.merge_infrequent_threshold
                )
                
                final_sig_count = len(all_valid_signatures)
                # The number of rules removed by pruning, for logging purposes.
                num_pruned_this_turn = initial_sig_count - final_sig_count
                if num_pruned_this_turn > 0:
                    # Add to the turn's removed count for the history log
                    newly_removed_count += num_pruned_this_turn

                # Encourage GC to reclaim any large temporary structures created during organization
                gc.collect()

            # --- 3. (MODIFIED) Calculate Reduction/Inactive for Recording (NOT applied to actual signature set) ---
            # NOTE: We calculate reduction/inactive removal counts for history recording,
            # but do NOT actually apply these removals to all_valid_signatures
            # The actual signature set continues to the next turn without reduction/inactive removal
            
            # Create a copy for reduction/inactive calculation (for recording only)
            signatures_for_recording = {sig_id: rule for sig_id, rule in all_valid_signatures.items() if sig_id not in signatures_to_remove}
            
            # --- Calculate Inactive Removal (for recording) ---
            # Track which signatures were created THIS turn to calculate inactive_removed_from_new
            new_signatures_before_inactive = {sig_id for sig_id in signatures_for_recording.keys() 
                                             if signature_turn_created.get(sig_id) == turn_counter}
            inactive_removed_count = 0
            inactive_removed_from_new = 0
            if signatures_for_recording and turn_eval_data is not None and not turn_eval_data.empty:
                sorted_sigs_for_inactive = sorted(signatures_for_recording.items(), key=lambda x: x[0])
                formatted_sigs_for_inactive = [{'id': sid, 'name': f'Sig_{sid}', 'rule_dict': r} 
                                               for sid, r in sorted_sigs_for_inactive]
                
                # Use batched performance calculation to get alerts
                _, _, _, _, _, _, inactive_eval_alerts = calculate_performance_in_batches(
                    turn_eval_data, 
                    formatted_sigs_for_inactive, 
                    args.evaluation_batch_size
                )
                
                if not inactive_eval_alerts.empty and 'signature_id' in inactive_eval_alerts.columns:
                    # Get signature IDs that triggered alerts
                    active_sig_ids = set(inactive_eval_alerts['signature_id'].dropna().unique())
                    # Find inactive signatures (signatures that never triggered alerts)
                    all_sig_ids = set(signatures_for_recording.keys())
                    inactive_ids = all_sig_ids - active_sig_ids
                    inactive_removed_count = len(inactive_ids)
                    
                    if inactive_removed_count > 0:
                        # Remove from the copy (for recording only, not actual all_valid_signatures)
                        for inactive_id in inactive_ids:
                            if inactive_id in signatures_for_recording:
                                del signatures_for_recording[inactive_id]
                                # Count if this was a newly created signature this turn
                                if inactive_id in new_signatures_before_inactive:
                                    inactive_removed_from_new += 1
            
            # --- Calculate Reduction (for recording) ---
            # Track which signatures were created THIS turn to calculate reduction_removed_from_new
            new_signatures_before_reduction = {sig_id for sig_id in signatures_for_recording.keys() 
                                              if signature_turn_created.get(sig_id) == turn_counter}
            reduction_removed_count = 0
            reduction_removed_from_new = 0
            if signatures_for_recording:
                reduction_count_before = len(signatures_for_recording)
                ids_before_reduction = set(signatures_for_recording.keys())
                signatures_for_recording = reduce_signatures_by_subsets(signatures_for_recording)
                reduction_count_after = len(signatures_for_recording)
                reduction_removed_count = reduction_count_before - reduction_count_after
                
                if reduction_removed_count > 0:
                    # Count how many of the removed signatures were from this turn
                    ids_after_reduction = set(signatures_for_recording.keys())
                    removed_ids = ids_before_reduction - ids_after_reduction
                    for removed_id in removed_ids:
                        if removed_id in new_signatures_before_reduction:
                            reduction_removed_from_new += 1
            
            # --- Exit Performance Evaluation (on reduced/inactive-removed set for recording) ---
            # Use the reduced/inactive-removed version for exit evaluation recording
            exit_sig_count = len(signatures_for_recording)
            exit_recall, exit_precision, exit_f1, exit_accuracy = 0, 0, 0, 0
            if turn_eval_data is not None and not turn_eval_data.empty:
                if signatures_for_recording:
                    # Sort by signature ID to ensure consistent ordering (for "first match wins" logic)
                    sorted_exit_sigs = sorted(signatures_for_recording.items(), key=lambda x: x[0])
                    formatted_exit_sigs = [{'id': sid, 'name': f'Sig_{sid}', 'rule_dict': r} for sid, r in sorted_exit_sigs]
                    
                    # --- MODIFIED: Use batched performance calculation ---
                    _, _, exit_recall, exit_precision, exit_f1, exit_accuracy, _ = calculate_performance_in_batches(
                        turn_eval_data, 
                        formatted_exit_sigs, 
                        args.evaluation_batch_size
                    )
                else:
                    # No signatures: calculate accuracy based on true negatives (no alerts = all normal correctly identified)
                    # TP=0, FP=0, TN=normal_count, FN=anomaly_count
                    if 'label' in turn_eval_data.columns:
                        total_rows = len(turn_eval_data)
                        normal_count = (turn_eval_data['label'] == 0).sum()
                        anomaly_count = (turn_eval_data['label'] == 1).sum()
                        # With no signatures: TP=0, FP=0, TN=normal_count, FN=anomaly_count
                        # Accuracy = (TP + TN) / total = TN / total = normal_count / total_rows
                        exit_accuracy = normal_count / total_rows if total_rows > 0 else 0.0
                        # Recall = TP / (TP + FN) = 0 / (0 + anomaly_count) = 0
                        exit_recall = 0.0
                        # Precision = TP / (TP + FP) = 0 / (0 + 0) = 0 (undefined, set to 0)
                        exit_precision = 0.0
                        exit_f1 = 0.0
                        logger.debug(f"[NoSignatures] Calculated accuracy with 0 signatures: {exit_accuracy:.4f} (normal: {normal_count}/{total_rows})")
            
            # Calculate net generated count (new signatures minus those removed by reduction/inactive from new ones)
            # This is what will be displayed in the graph as "generated"
            net_generated = new_signatures_found - reduction_removed_from_new - inactive_removed_from_new
            
            # Calculate actual signature count change in the actual signature set (not recording copy)
            # This is the real change: what signatures are actually in the set after this turn
            actual_exit_sig_count = len(all_valid_signatures) - len(signatures_to_remove)
            actual_net_change = actual_exit_sig_count - entry_sig_count
            
            # Calculate balanced removed count to ensure: generated - removed = actual_net_change
            # This ensures the accounting is correct: sum(generated) - sum(removed) = final signature count
            # removed = generated - actual_net_change
            balanced_removed = max(0, net_generated - actual_net_change)

            logger.info(f"End of Turn {turn_counter}. Signatures (actual): {entry_sig_count} -> {actual_exit_sig_count} (after reduction/inactive removal for recording: {exit_sig_count}). "
                       f"EXIT Recall: {exit_recall:.4f}. EXIT Precision: {exit_precision:.4f}. EXIT F1: {exit_f1:.4f}. EXIT Accuracy: {exit_accuracy:.4f}")
            logger.info(f"[Recording] Inactive removed: {inactive_removed_count} (from new: {inactive_removed_from_new}), Reduction removed: {reduction_removed_count} (from new: {reduction_removed_from_new}) (for history recording only, not applied to actual set)")
            logger.info(f"[Balance] Actual net change: {actual_net_change} = exit({actual_exit_sig_count}) - entry({entry_sig_count}), Generated: {net_generated}, Removed (FP+Pruning): {newly_removed_count}, Removed (balanced): {balanced_removed}")
            
            history.append({
                'turn': turn_counter, 
                'entry_signature_count': entry_sig_count, 
                'generated': net_generated,  # Net generated (after reduction/inactive removal from new signatures)
                'removed': balanced_removed,  # Balanced to ensure: generated - removed = actual_net_change (so accounting sums correctly)
                'inactive_removed': inactive_removed_count,  # Total inactive removals (for logging)
                'inactive_removed_from_new': inactive_removed_from_new,  # Inactive from new signatures (for reference)
                'reduction_removed': reduction_removed_count,  # Total reduction removals (for logging)
                'reduction_removed_from_new': reduction_removed_from_new,  # Reduction from new signatures (for reference)
                'exit_signature_count': exit_sig_count,  # After reduction/inactive removal for recording
                'entry_recall': entry_recall, 
                'entry_precision': entry_precision, 
                'entry_f1': entry_f1, 
                'entry_accuracy': entry_accuracy, 
                'exit_recall': exit_recall, 
                'exit_precision': exit_precision, 
                'exit_f1': exit_f1, 
                'exit_accuracy': exit_accuracy
            })
            
            # --- NEW: Save Checkpoint ---
            save_checkpoint(run_dir, turn_counter, all_valid_signatures, signatures_to_remove, history, signature_turn_created)

            # Turn-level GC to clean up any remaining temporary objects before next chunk
            try:
                del turn_eval_data
            except NameError:
                pass
            gc.collect()

        # --- Finalization ---
        # NOTE: Apply reduction and inactive removal to final signatures for saving
        # (but the actual signature set during turns was not affected)
        final_signatures_before_reduction = {sig_id: rule for sig_id, rule in all_valid_signatures.items() if sig_id not in signatures_to_remove}
        
        logger.info("--- Applying final reduction and inactive removal to signatures for saving ---")
        
        # Final inactive removal: remove signatures that never triggered alerts
        if final_signatures_before_reduction and chunk_cache is not None and not chunk_cache.is_empty():
            logger.info(f"Evaluating {len(final_signatures_before_reduction)} final signatures to identify inactive ones...")
            # Sort by signature ID to ensure consistent ordering (for "first match wins" logic)
            sorted_final_sigs = sorted(final_signatures_before_reduction.items(), key=lambda x: x[0])
            formatted_final_sigs = [{'id': sid, 'name': f'Sig_{sid}', 'rule_dict': r} 
                                    #for sid, r in final_signatures_before_reduction.items()]
                                   for sid, r in sorted_final_sigs]
            
            # Evaluate against entire dataset to find which signatures triggered alerts
            _, _, _, _, _, _, final_eval_alerts = calculate_performance_in_batches(
                chunk_cache, 
                formatted_final_sigs, 
                args.evaluation_batch_size
            )
            
            inactive_count_before_final = len(final_signatures_before_reduction)
            if not final_eval_alerts.empty and 'signature_id' in final_eval_alerts.columns:
                # Get signature IDs that triggered alerts
                active_sig_ids = set(final_eval_alerts['signature_id'].dropna().unique())
                # Find inactive signatures
                all_sig_ids = set(final_signatures_before_reduction.keys())
                inactive_ids = all_sig_ids - active_sig_ids
                
                if inactive_ids:
                    # Remove inactive signatures
                    for inactive_id in inactive_ids:
                        if inactive_id in final_signatures_before_reduction:
                            del final_signatures_before_reduction[inactive_id]
                            # Also remove from signature_turn_created for consistency
                            if inactive_id in signature_turn_created:
                                del signature_turn_created[inactive_id]
                    
                    inactive_count_after_final = len(final_signatures_before_reduction)
                    final_inactive_removed = inactive_count_before_final - inactive_count_after_final
                    logger.info(f"[Final Inactive Removal] Removed {final_inactive_removed} inactive signatures. "
                               f"Signatures: {inactive_count_before_final} -> {inactive_count_after_final}")
        
        # Final reduction: remove supersets when subsets exist
        reduction_count_before_final = len(final_signatures_before_reduction)
        final_signatures = reduce_signatures_by_subsets(final_signatures_before_reduction)
        reduction_count_after_final = len(final_signatures)
        final_reduction_removed = reduction_count_before_final - reduction_count_after_final
        if final_reduction_removed > 0:
            logger.info(f"[Final Reduction] Removed {final_reduction_removed} superset signatures. "
                       f"Signatures: {reduction_count_before_final} -> {reduction_count_after_final}")

        logger.info("--- Process Complete ---")
        logger.info(f"Initial unique signatures generated: {len(all_valid_signatures)}")
        logger.info(f"Signatures removed due to FPs: {len(signatures_to_remove)}")
        logger.info(f"Final count of validated signatures (after reduction/inactive removal): {len(final_signatures)}")

        # Build signature metadata: created turn and attack types
        def get_attack_columns(ftype):
            if ftype in ['CICIDS2017']:
                return ['Label']
            if ftype in ['CICIoT2023', 'CICIoT']:
                return ['attack_name']
            if ftype in ['DARPA98', 'DARPA']:
                return ['Class']
            if ftype in ['MiraiBotnet']:
                return ['reconnaissance', 'infection', 'action']
            if ftype in ['netML']:
                return ['Label']
            if ftype in ['NSL-KDD', 'NSL_KDD']:
                return ['class']
            return []

        def extract_attack_types_for_signature(sig, chunk_cache_obj, attack_cols, batch_size):
            attacks = set()
            if not attack_cols:
                attacks.add('-')
                return attacks

            # stream through cached chunks
            for batch in chunk_cache_obj.iter_batches(batch_size=batch_size):
                if batch is None or batch.empty:
                    continue
                rule_keys = [k for k in sig.keys() if k in batch.columns]
                if not rule_keys:
                    continue
                match_mask = batch[rule_keys].eq(pd.Series({k: sig[k] for k in rule_keys})).all(axis=1)
                if not match_mask.any():
                    continue
                matched = batch.loc[match_mask]

                # Mirai one-hot special handling
                if attack_cols == ['reconnaissance', 'infection', 'action']:
                    for col in attack_cols:
                        if col in matched.columns:
                            vals = matched[col]
                            if (vals == 1).any():
                                attacks.add(col)
                    continue

                for col in attack_cols:
                    if col in matched.columns:
                        vals = matched[col].astype(str).unique().tolist()
                        attacks.update(vals)

            if not attacks:
                attacks.add('-')
            return attacks

        attack_cols = get_attack_columns(args.file_type)
        signature_records = []
        #for sig_id, rule in final_signatures.items():
        # Sort by signature ID to ensure consistent CSV output order
        sorted_final_signatures = sorted(final_signatures.items(), key=lambda x: x[0])
        total_sigs = len(final_signatures)
        
        # Calculate individual signature performance metrics (TP, FP, TN, FN, F1)
        logger.info(f"Calculating individual performance metrics for {total_sigs} signatures...")
        individual_metrics = {}
        if chunk_cache is not None and not chunk_cache.is_empty() and sorted_final_signatures:
            # Prepare signatures for calculate_signature (expects list of dicts)
            sig_list_for_calc = [rule for sig_id, rule in sorted_final_signatures]
            
            # Collect all data from chunk_cache for evaluation
            all_eval_data = []
            for batch in chunk_cache.iter_batches(batch_size=args.evaluation_batch_size):
                if batch is not None and not batch.empty:
                    all_eval_data.append(batch)
            
            if all_eval_data:
                eval_df = pd.concat(all_eval_data, ignore_index=True)
                logger.info(f"Evaluating {total_sigs} signatures against {len(eval_df)} rows for individual metrics...")
                
                # Calculate individual signature metrics
                sig_metrics_results = calculate_signature(eval_df, sig_list_for_calc)
                
                # Map results back to signature IDs
                for idx, (sig_id, rule) in enumerate(sorted_final_signatures):
                    if idx < len(sig_metrics_results):
                        result = sig_metrics_results[idx]
                        tp = result.get('TP', 0)
                        fp = result.get('FP', 0)
                        tn = result.get('TN', 0)
                        fn = result.get('FN', 0)
                        
                        # Calculate precision, recall, F1
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                        
                        individual_metrics[sig_id] = {
                            'tp': tp,
                            'fp': fp,
                            'tn': tn,
                            'fn': fn,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1_score
                        }
                
                logger.info(f"Completed individual performance calculation for {len(individual_metrics)} signatures.")
            else:
                logger.warning("No evaluation data available for individual signature metrics.")
        else:
            logger.warning("Chunk cache is empty or no signatures. Skipping individual performance calculation.")
        
        logger.info(f"Extracting attack types for {total_sigs} signatures...")
        #for sig_idx, (sig_id, rule) in enumerate(final_signatures.items(), 1):
        for sig_idx, (sig_id, rule) in enumerate(sorted_final_signatures, 1):
            if sig_idx % 100 == 0 or sig_idx == 1 or sig_idx == total_sigs:
                logger.info(f"  Processing signature {sig_idx}/{total_sigs}...")
            created_turn = signature_turn_created.get(sig_id, None)
            attacks = extract_attack_types_for_signature(rule, chunk_cache, attack_cols, args.evaluation_batch_size)
            
            # Get individual metrics for this signature
            metrics = individual_metrics.get(sig_id, {})
            
            signature_records.append({
                'signature_rule': str(rule),
                'created_turn': created_turn,
                'attack_types': "|".join(sorted(attacks)),
                'tp': metrics.get('tp', 0),
                'fp': metrics.get('fp', 0),
                'tn': metrics.get('tn', 0),
                'fn': metrics.get('fn', 0),
                'precision': metrics.get('precision', 0.0),
                'recall': metrics.get('recall', 0.0),
                'f1_score': metrics.get('f1_score', 0.0)
            })
        logger.info(f"Finished extracting attack types and metrics for all {total_sigs} signatures.")

        final_signatures_df = pd.DataFrame(signature_records)

        # The output directory is now the same as the run_dir for artifacts/checkpoints
        output_dir = run_dir

        # UPDATED: Add parameters to filenames for clarity
        param_str = params_str  # already includes _turneval

        output_filename = f"{args.file_type}_{args.file_number}_{param_str}_incremental_signatures_eex.csv"
        output_path = os.path.join(output_dir, output_filename)

        final_signatures_df.to_csv(output_path, index=False)
        logger.info(f"Final signatures saved to: {output_path}")

        # --- PLOTTING and HISTORY CSV---
        if history:
            history_df = pd.DataFrame(history)

            performance_filename = f"{args.file_type}_{args.file_number}_{param_str}_performance_history_eex.csv"
            performance_path = os.path.join(output_dir, performance_filename)
            history_df.to_csv(performance_path, index=False)
            logger.info(f"Performance history saved to: {performance_path}")

            if plt:
                logger.info("Generating performance graph...")
                
                # Create a single figure and a primary axis for performance metrics
                fig, ax_perf = plt.subplots(figsize=(18, 8)) # Wider figure for better readability
                fig.suptitle(f'Incremental Signature Performance for {args.file_type}\n(support={args.min_support}, confidence={args.min_confidence})', fontsize=16)

                # Create a secondary axis for signature counts that shares the same x-axis
                ax_counts = ax_perf.twinx()

                x_labels = []
                x_ticks = []
                bar_width = 0.35

                # --- Plotting Loop for both lines and bars ---
                for i, row in history_df.iterrows():
                    turn = row['turn']
                    x_entry = i * 2
                    x_exit = i * 2 + 1

                    # --- 1. Plot Performance Lines on the primary axis (ax_perf) ---
                    # Plot Learning phase (solid line)
                    ax_perf.plot([x_entry, x_exit], [row['entry_recall'], row['exit_recall']], 'o-', color='blue', label='Recall (Learning)' if i == 0 else "")
                    ax_perf.plot([x_entry, x_exit], [row['entry_precision'], row['exit_precision']], 'x-', color='purple', label='Precision (Learning)' if i == 0 else "")
                    ax_perf.plot([x_entry, x_exit], [row['entry_f1'], row['exit_f1']], 's-', color='orange', label='F1-Score (Learning)' if i == 0 else "")
                    ax_perf.plot([x_entry, x_exit], [row['entry_accuracy'], row['exit_accuracy']], 'd-', color='green', label='Accuracy (Learning)' if i == 0 else "")
                    
                    # Plot Adaptation phase (dotted line)
                    if i < len(history_df) - 1:
                        next_row = history_df.iloc[i+1]
                        ax_perf.plot([x_exit, x_exit + 1], [row['exit_recall'], next_row['entry_recall']], 'o--', color='blue', alpha=0.5, label='Recall (Adaptation)' if i == 0 else "")
                        ax_perf.plot([x_exit, x_exit + 1], [row['exit_precision'], next_row['entry_precision']], 'x--', color='purple', alpha=0.5, label='Precision (Adaptation)' if i == 0 else "")
                        ax_perf.plot([x_exit, x_exit + 1], [row['exit_f1'], next_row['entry_f1']], 's--', color='orange', alpha=0.5, label='F1-Score (Adaptation)' if i == 0 else "")
                        ax_perf.plot([x_exit, x_exit + 1], [row['exit_accuracy'], next_row['entry_accuracy']], 'd--', color='green', alpha=0.5, label='Accuracy (Adaptation)' if i == 0 else "")

                    # --- 2. Plot Count Bars on the secondary axis (ax_counts) ---
                    # Position the bars in the middle of the entry-exit gap
                    bar_center = x_entry + 0.5
                    ax_counts.bar(bar_center - bar_width/2, row['generated'], bar_width, label='Generated' if i == 0 else "", color='green', alpha=0.6)
                    ax_counts.bar(bar_center + bar_width/2, row['removed'], bar_width, label='Removed' if i == 0 else "", color='red', alpha=0.6)

                    x_ticks.extend([x_entry, x_exit])
                    x_labels.extend([f"{turn}-entry", f"{turn}-exit"])

                # --- Formatting and Labels ---
                ax_perf.set_xticks(x_ticks)
                ax_perf.set_xticklabels(x_labels, rotation=45, ha='right')
                ax_perf.set_xlabel(f'Turn ({args.chunk_size}-row chunks)')
                ax_perf.set_ylabel('Metric Value (Recall, Precision, F1)')
                ax_perf.set_ylim(0, 1.05)
                ax_perf.grid(True, linestyle='--')

                ax_counts.set_ylabel('Signature Count (Generated/Removed)', color='gray')
                ax_counts.tick_params(axis='y', labelcolor='gray')
                # Ensure the bottom of the bar chart is at 0
                ax_counts.set_ylim(bottom=0)

                # Combine legends from both axes
                handles_perf, labels_perf = ax_perf.get_legend_handles_labels()
                handles_counts, labels_counts = ax_counts.get_legend_handles_labels()
                ax_perf.legend(handles_perf + handles_counts, labels_perf + labels_counts, loc='best')

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                
                graph_dir = "../isv_graph/"
                if not os.path.exists(graph_dir):
                    try:
                        os.makedirs(graph_dir)
                    except OSError as e:
                        logger.error(f"Could not create graph directory {graph_dir}: {e}")
                        graph_dir = "."
                
                graph_filename = f"{args.file_type}_{args.file_number}_{param_str}_metrics_eex.jpg"
                graph_path = os.path.join(graph_dir, graph_filename)
                
                try:
                    plt.savefig(graph_path, format='jpg', dpi=150)
                    logger.info(f"Performance graph saved to: {graph_path}")
                except Exception as e:
                    logger.error(f"Failed to save graph: {e}")
            
            # --- NEW: Final Summary Printout ---
            final_stats = history_df.iloc[-1]
            logger.info("--- Final Summary ---")
            logger.info(f"Total Validated Signatures: {len(final_signatures)}")
            logger.info(f"Final Recall (at Turn {int(final_stats['turn'])}): {final_stats['exit_recall']:.4f}")
            logger.info(f"Final Precision (at Turn {int(final_stats['turn'])}): {final_stats['exit_precision']:.4f}")
            logger.info(f"Final F1-Score (at Turn {int(final_stats['turn'])}): {final_stats['exit_f1']:.4f}")
            logger.info(f"Final Accuracy (at Turn {int(final_stats['turn'])}): {final_stats.get('exit_accuracy', 0):.4f}")
            logger.info("--------------------")
        else:
            final_stats = None

        end_time = time.time()

        # --- NEW: Save concise final metrics CSV ---
        try:
            total_generation_time = end_time - start_time
            num_final_signatures = len(final_signatures)
            avg_conditions = float(np.mean([len(rule) for rule in final_signatures.values()])) if final_signatures else 0.0
            final_recall = final_stats['exit_recall'] if final_stats is not None else 0.0

            summary_record = {
                "time_to_generate_signatures_sec": total_generation_time,
                "num_signatures_final": num_final_signatures,
                "avg_conditions_per_signature": avg_conditions,
                "total_recall_final_turn_exit": final_recall
            }

            summary_df = pd.DataFrame([summary_record])
            summary_filename = f"{args.file_type}_{args.file_number}_{param_str}_summary_metrics.csv"
            summary_path = os.path.join(output_dir, summary_filename)
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"Summary metrics saved to: {summary_path}")
        except Exception as e:
            logger.error(f"Failed to save summary metrics CSV: {e}")

        #end_time = time.time()
        logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")

    except Exception as e:
        # This is a critical logging block. If the main process fails for any reason
        # (including potential memory issues leading to other errors), this will be the
        # last thing logged before worker processes might start failing with BrokenPipeError.
        logger.error("="*80)
        logger.error("! AN UNHANDLED EXCEPTION OCCURRED IN THE MAIN PROCESS !")
        logger.error(f"Error Type: {type(e).__name__}")
        logger.error(f"Error Details: {e}")
        logger.error("This is very likely the ROOT CAUSE of the process terminating unexpectedly.")
        logger.error("If you see a flood of 'BrokenPipeError' messages after this, they are a SYMPTOM, not the cause.")
        logger.error("The main process died, and the worker processes could no longer communicate with it.")
        logger.error("Please check the traceback below for the actual error.")
        logger.error("="*80)
        
        # It's crucial to re-raise the exception to see the full traceback of the root cause.
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Incrementally generate and validate signatures from a dataset.")
    parser.add_argument('--file_type', type=str, default="MiraiBotnet", help="Type of the dataset file.")
    parser.add_argument('--file_number', type=int, default=1, help="Number of the dataset file.")
    parser.add_argument('--association_method', type=str, default='rarm', help="Association rule algorithm to use.")
    parser.add_argument('--min_support', type=float, default=0.3, help="Minimum support for association rule mining.")
    parser.add_argument('--normal_min_support', type=float, default=None, help="Optional override for normal-data filtering support threshold (defaults to min_support).")
    parser.add_argument('--min_confidence', type=float, default=0.8, help="Minimum confidence for association rule mining.")
    # MODIFIED: Default to None to detect if the user has provided a value.
    parser.add_argument('--num_processes', type=int, default=None, help="Number of processes to use for parallel tasks. Defaults to all available cores.")
    parser.add_argument('--chunk_size', type=int, default=500, help="Number of rows to process in each incremental turn.")
    parser.add_argument('--cstemporal', action='store_true', help="Use dataset-specific temporal window preset instead of numeric chunk_size. File names will use 'tem' for cs.")
    parser.add_argument('--itemset_limit', type=int, default=10000000, help="Safety limit for frequent itemsets to prevent memory overflow before rule generation.")
    parser.add_argument('--n_splits', type=int, default=40, help="Number of splits to use for dynamic interval mapping. Default is 40.")
    parser.add_argument('--signature_batch_size', type=int, default=20000, help="Batch size for validating signatures to conserve memory.")
    parser.add_argument('--normal_data_batch_size', type=int, default=30000, help="Batch size for splitting the turn's normal data to conserve memory during filtering. If not set, normal data is not batched.")
    parser.add_argument('--evaluation_batch_size', type=int, default=20000, help="Batch size for processing the full dataset during final performance evaluation to conserve memory.")
    parser.add_argument('--rule_spool_chunk_size', type=int, default=None, help="Number of rules to store per chunk when spooling to disk before filtering (auto if omitted).")
    parser.add_argument('--prune_signatures', action='store_true', help="If set, enables the signature pruning (subsumption) process to remove redundant rules.")
    parser.add_argument('--prune_coverage_threshold', type=float, default=0.9, help="Coverage threshold for the signature pruning process.")
    parser.add_argument('--merge_signatures', action='store_true', help="If set, enables merging of similar, infrequent signatures. (Requires --prune_signatures)")
    parser.add_argument('--merge_infrequent_threshold', type=int, default=5, help="TP count at or below which a rule is considered infrequent and eligible for merging.")
    parser.add_argument('--signature_organize', action='store_true', help="A shorthand to enable both --prune_signatures and --merge_signatures.")
    parser.add_argument('--save_artifacts', action='store_true', help="If set, save intermediate itemsets and rules for debugging.")
    # --- Dynamic Support Arguments ---
    parser.add_argument('--dynamic_support', action='store_true', help="Enable adaptive support thresholding to prevent memory overflow.")
    parser.add_argument('--itemset_count_threshold', type=int, default=500000, help="Itemset count limit at any level that triggers dynamic support adjustment.")
    parser.add_argument('--support_increment_factor', type=float, default=1.2, help="Factor by which to multiply min_support when the threshold is exceeded (e.g., 1.2 for a 20% increase).")
    parser.add_argument('--negative_filtering', action='store_true', help="Enable negative-aware filtering using P(rule|normal) thresholds.")
    parser.add_argument('--negative_filter_threshold', type=float, default=0.05, help="Maximum allowed P(rule|normal) when negative-aware filtering is enabled.")
    parser.add_argument('--reset', action='store_true', help="If set, deletes the checkpoint and artifact directory for the given parameters and exits.")
    parser.add_argument('--mask_dominant_cols', action='store_true', default=True, help="Mask near-constant columns (freq>0.99) from support counting and re-attach them to generated rules.")
    parser.add_argument('--dominant_freq_threshold', type=float, default=0.99, help="Frequency threshold to detect dominant (near-constant) columns for masking.")
    parser.add_argument('--dominant_min_support', type=float, default=None, help="Optional min_support override when masking is skipped due to all features being dominant.")
    parser.add_argument('--dominant_min_confidence', type=float, default=None, help="Optional min_confidence override when masking is skipped due to all features being dominant.")
    parser.add_argument('--dominant_normal_min_support', type=float, default=None, help="Optional normal_min_support override when masking is skipped due to all features being dominant.")
    parser.add_argument('--dominant_level', type=int, default=None, help="Optional max_level override when masking is skipped due to all features being dominant.")
    parser.add_argument('--run_turn_start', type=int, default=None, help="Start turn (inclusive) to run experiment")
    parser.add_argument('--run_turn_end', type=int, default=None, help="End turn (inclusive) to run experiment")
    # --- NRA/UFP/HAF based FP removal parameters ---
    parser.add_argument('--use_fp_metrics', action='store_true', help="Use NRA/UFP/HAF based FP removal instead of simple normal alert detection")
    parser.add_argument('--fp_t0_nra', type=int, default=60, help='Time window (seconds) for NRA calculation')
    parser.add_argument('--fp_n0_nra', type=int, default=20, help='Normalization factor for NRA calculation')
    parser.add_argument('--fp_lambda_haf', type=float, default=100.0, help='Lambda parameter for HAF score calculation')
    parser.add_argument('--fp_lambda_ufp', type=float, default=10.0, help='Lambda parameter for UFP score calculation')
    parser.add_argument('--fp_combine_method', type=str, default='max', choices=['max', 'weighted_sum'], help='Method to combine NRA, HAF, UFP scores')
    parser.add_argument('--fp_belief_threshold', type=float, default=0.5, help='Threshold for FP belief score to classify a signature as FP')
    parser.add_argument('--fp_superset_strictness', type=float, default=0.9, help='Strictness multiplier for superset FP detection')
    parser.add_argument('--disable_fp_removal', action='store_true', help='Disable FP removal entirely. Signatures will not be removed due to false positives.')

    cli_args = parser.parse_args()

    # --- NEW: Handle the --signature_organize shorthand ---
    if cli_args.signature_organize:
        cli_args.prune_signatures = True
        cli_args.merge_signatures = True

    # If --num_processes is not provided by the user (i.e., it's None), default to all available cores.
    if cli_args.num_processes is None:
        try:
            # Use os.cpu_count() which is recommended for getting the number of CPUs.
            cpu_count = os.cpu_count()
            cli_args.num_processes = cpu_count
            logger.info(f"--num_processes not set, defaulting to all available cores: {cpu_count}")
        except NotImplementedError:
            logger.warning("os.cpu_count() is not implemented. Defaulting to 4 processes.")
            cli_args.num_processes = 4 # Fallback

    main(cli_args) 
