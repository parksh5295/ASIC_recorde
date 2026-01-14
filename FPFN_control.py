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
from Rebuild_Method.fp_fn_control import (
    inject_fp_signatures,
    remove_fn_signatures,
    check_fn_recovery,
    summarize_fn_recovery
)

try:
    from Dataset_Choose_Rule.association_data_choose import get_clustered_data_path # MODIFIED
    from Dataset_Choose_Rule.choose_amount_dataset import file_cut
    from definition.Anomal_Judgment import anomal_judgment_label, anomal_judgment_nonlabel
    from utils.time_transfer import time_scalar_transfer
    from Modules.Heterogeneous_module import choose_heterogeneous_method
    from Heterogeneous_Method.separate_group_mapping import map_intervals_to_groups
    from Modules.Association_module import association_module
    from Modules.Difference_sets import dict_list_difference
    from Rebuild_Method.FalsePositive_Check import apply_signatures_to_dataset
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
    """
    if not signatures or data_source is None:
        return 0, 0, 0, 0, 0, 0 # tp, fp, recall, precision, f1, accuracy

    all_alerted_indices = set()
    actual_positives_indices = set()
    actual_negatives_indices = set()

    total_rows = 0
    batch_counter = 0

    for batch_df in _yield_batches_from_source(data_source, batch_size):
        batch_counter += 1
        total_rows += len(batch_df)

        alerts_batch = apply_signatures_to_dataset(batch_df, signatures)
        if not alerts_batch.empty:
            all_alerted_indices.update(alerts_batch['alert_index'].unique())

        actual_positives_indices.update(batch_df[batch_df['label'] == 1].index)
        actual_negatives_indices.update(batch_df[batch_df['label'] == 0].index)

    if total_rows == 0:
        return 0, 0, 0, 0, 0, 0

    logger.info(f"  [Perf Eval] Processed {total_rows} rows across {batch_counter} streamed batch(es).")

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
        
    return tp, fp, recall, precision, f1, accuracy


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

        # --- FP/FN Control Setup ---
        fn_removed_history = {}
        fp_injected_history = {}
        fp_removed_history = {}
        fp_alive = set()
        fn_recovered_cumulative = set()
        fpfn_events = []
        recovered_info_latest = {}

        fp_turns = set(map(int, args.FP_turns.split(','))) if args.FP_turns else set()
        fn_turns = set(map(int, args.FN_turns.split(','))) if args.FN_turns else set()

        # --- Build FP/FN and turn-range tag for isolation of outputs ---
        def _turns_to_label(turn_set):
            return "NA" if not turn_set else "-".join(str(t) for t in sorted(turn_set))

        start_label = args.run_turn_start if args.run_turn_start is not None else "NA"
        end_label = args.run_turn_end if args.run_turn_end is not None else "NA"
        fp_label = _turns_to_label(fp_turns) if args.FP_control else "NA"
        fn_label = _turns_to_label(fn_turns) if args.FN_control else "NA"
        fpfn_tag = f"ts{start_label}-te{end_label}_fp{fp_label}_fn{fn_label}"

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
        params_str = f"{params_str_base}_n{args.n_splits}_dom{args.dominant_freq_threshold}{dom_suffix}_turneval_fpfn-{fpfn_tag}"
        # Isolate FP/FN runs under a dedicated root to avoid clashing with regular runs
        base_output_root = os.path.join("../Dataset_ex", "FPFN")
        run_dir = os.path.join(base_output_root, args.file_type, params_str) # Used for artifacts and checkpoints
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
        # MODIFIED: Drop both 'label' and 'cluster' before creating mapping rules
        mapping_features = processed_first_chunk.drop(columns=['label', 'cluster'], errors='ignore')
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

            # --- TURN RANGE CONTROL ---
            if args.run_turn_start is not None and turn_counter < args.run_turn_start:
                continue

            if args.run_turn_end is not None and turn_counter > args.run_turn_end:
                logger.info(f"Reached run_turn_end={args.run_turn_end}. Stop experiment.")
                break

            # --- 1. Skip or Process ---
            if turn_counter <= resume_from_turn:
                # Rebuild state (already done in fast-forward, but need to concat the first chunk if turn 1 is skipped)
                # This is getting complicated. Let's simplify the loop structure.
                continue # Skip processing for turns already completed.

            # The loop starts from the next turn to process
            logger.info(f"--- Processing Turn {turn_counter} (Rows {(turn_counter-1)*args.chunk_size + 1} - {turn_counter*args.chunk_size}) ---")
            
            # --- Preprocessing for the Current Chunk ---
            processed_chunk = time_scalar_transfer(chunk, args.file_type)
            mapped_chunk = preprocess_and_map_chunk(processed_chunk, args.file_type, category_mapping, data_list)

            # Persist the mapped chunk for future streaming-based evaluations
            mapped_chunk = chunk_cache.register_chunk(mapped_chunk, turn_counter)
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
                formatted_entry_sigs = [{'id': sid, 'name': f'Sig_{sid}', 'rule_dict': r} for sid, r in rules_at_turn_start.items()]
                
                # --- MODIFIED: Use batched performance calculation ---
                _, _, entry_recall, entry_precision, entry_f1, entry_accuracy = calculate_performance_in_batches(
                    turn_eval_data, 
                    formatted_entry_sigs, 
                    args.evaluation_batch_size
                )
                
                logger.info(f"Turn {turn_counter} ENTRY Performance - Recall: {entry_recall:.4f}, Precision: {entry_precision:.4f}, F1: {entry_f1:.4f}, Accuracy: {entry_accuracy:.4f}")

            # MODIFIED: Split chunk data based on the 'cluster' column
            normal_data_in_chunk = mapped_chunk[mapped_chunk['cluster'] == 0].copy().drop(columns=['label', 'cluster'], errors='ignore')
            # Keep label for other potential uses, but it will be dropped before association
            anomalous_data_in_chunk = mapped_chunk[mapped_chunk['cluster'] == 1].copy()

            # --- FP CONTROL HOOK ---
            if args.FP_control and turn_counter in fp_turns:
                injected = inject_fp_signatures(
                    all_valid_signatures,
                    normal_data_in_chunk
                )
                logger.warning(f"[FP_CONTROL] Injected {len(injected)} fake FP signatures at turn {turn_counter}")
                fp_injected_history[turn_counter] = set(injected)
                fp_alive.update(injected)
            else:
                injected = set()
                fp_injected_history[turn_counter] = set()

            # Release the mapped chunk reference early to reduce peak memory per turn
            del mapped_chunk
            gc.collect()

            # --- 1. Validation Step ---
            newly_removed_count = 0
            newly_flagged_for_removal = set() # Store IDs of signatures removed THIS turn
            if all_valid_signatures and not normal_data_in_chunk.empty:
                logger.info(f"Validating {len(all_valid_signatures) - len(signatures_to_remove)} existing signatures against {len(normal_data_in_chunk)} normal data rows...")
                signatures_to_test = [ {'id': sig_id, 'name': f'Sig_{sig_id}', 'rule_dict': rule} for sig_id, rule in all_valid_signatures.items() if sig_id not in signatures_to_remove ]
                
                if signatures_to_test:
                    #fp_alerts = apply_signatures_to_dataset(normal_data_in_chunk, signatures_to_test)
                    #if not fp_alerts.empty:
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

            rule_spooler = RuleSpooler(run_dir, turn_counter, chunk_size=rule_spool_chunk_size_runtime)
            new_signatures_found = 0
            try:
                # --- 2. Generation Step (Spooling) ---
                if not anomalous_data_in_chunk.empty:
                    logger.info(f"Generating new candidate rules from {len(anomalous_data_in_chunk)} anomalous data rows...")
                    
                    # Identify dominant (near-constant) columns and mask them for support counting
                    dominant_cols = {}
                    mask_skipped = False
                    if args.mask_dominant_cols:
                        dominant_cols = get_dominant_columns(anomalous_data_in_chunk, freq_threshold=args.dominant_freq_threshold)
                        if dominant_cols:
                            logger.info(f"[DominantCols] Masking {len(dominant_cols)} near-constant columns for support counting: {list(dominant_cols.keys())}")

                    anomalous_for_mining = anomalous_data_in_chunk.drop(columns=dominant_cols.keys(), errors='ignore') if dominant_cols else anomalous_data_in_chunk
                    # Guard: if masking removed everything (or nearly everything), skip masking
                    if args.mask_dominant_cols and dominant_cols and anomalous_for_mining.drop(columns=['label','cluster'], errors='ignore').shape[1] == 0:
                        # Check if all (non label/cluster) columns are truly constant (support=1). If yes, keep masking.
                        cols_for_check = [c for c in anomalous_data_in_chunk.columns if c not in ['label', 'cluster']]
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
                    
                    calculate_and_log_support_stats(
                        #anomalous_data_in_chunk.drop(columns=['label', 'cluster'], errors='ignore'),
                        anomalous_for_mining.drop(columns=['label', 'cluster'], errors='ignore'),
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

                    standard_rules, _ = association_module(
                        #anomalous_data_in_chunk.drop(columns=['label', 'cluster'], errors='ignore'),
                        anomalous_for_mining.drop(columns=['label', 'cluster'], errors='ignore'),
                        **assoc_args
                    )
                    # Re-attach dominant columns (near-constant) so rules include them
                    if standard_rules and dominant_cols:
                        for rule in standard_rules:
                            for d_col, d_val in dominant_cols.items():
                                rule.setdefault(d_col, d_val)
                    # Fallback: if no rules were generated, create minimal rules from dominant cols
                    if (not standard_rules or len(standard_rules) == 0) and dominant_cols:
                        standard_rules = build_rules_from_dominant(dominant_cols)
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
                    
                    def rule_passes_global(rule):
                        """
                        Evaluate a rule against all normal-data chunks, accumulating global support.
                        Batching is for memory only; filtering is based on global support, not worst-case.
                        """
                        if normal_data_empty:
                            return True
                        
                        total_rows = 0
                        total_matches = 0
                        for chunk in normal_data_chunks:
                            if chunk is None or chunk.empty:
                                continue
                            total_rows += len(chunk)
                            
                            rule_keys_in_chunk = [k for k in rule.keys() if k in chunk.columns]
                            if not rule_keys_in_chunk:
                                # No relevant columns in this chunk -> zero matches in this chunk
                                continue
                            
                            rule_series = pd.Series({k: rule[k] for k in rule_keys_in_chunk})
                            match_mask = chunk[rule_keys_in_chunk].eq(rule_series).all(axis=1)
                            total_matches += match_mask.sum()
                        
                        if total_rows == 0:
                            return True  # No normal data present
                        
                        support = total_matches / total_rows
                        
                        # Negative filtering uses the same global support
                        if negative_filtering_enabled and support > negative_filter_threshold:
                            return False
                        
                        # Standard normal_min_support filtering
                        if support > normal_min_support_effective:
                            return False
                        
                        return True
                    
                    def filter_rule_batch(rule_batch):
                        if not rule_batch:
                            return []
                        if normal_data_empty:
                            return list(rule_batch)
                        
                        filtered = []
                        for r in rule_batch:
                            if rule_passes_global(r):
                                filtered.append(r)
                        return filtered

                    total_filtered = 0
                    for rules_batch in rule_spooler.consume_chunks():
                        filtered_rules = filter_rule_batch(rules_batch)
                        total_filtered += len(filtered_rules)
                        for rule in filtered_rules:
                            rule_id = hash(frozenset(rule.items()))
                            if rule_id not in all_valid_signatures and rule_id not in signatures_to_remove:
                                all_valid_signatures[rule_id] = rule
                                signature_turn_created.setdefault(rule_id, turn_counter)
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

            # --- FN CONTROL HOOK ---
            removed = set()
            if args.FN_control and turn_counter in fn_turns:
                fn_strategy = 'oldest' if args.FN_strategy == 'early' else args.FN_strategy
                removed = remove_fn_signatures(
                    all_valid_signatures,
                    signature_turn_created,
                    strategy=fn_strategy,
                    remove_ratio=args.FN_remove_ratio
                )
                fn_removed_history[turn_counter] = removed
                logger.warning(f"[FN_CONTROL] Force-removed {len(removed)} important signatures at turn {turn_counter}")
            else:
                fn_removed_history[turn_counter] = set()
            
            fn_removed_count = len(removed)

            # --- 3. (MODIFIED) Exit Performance Evaluation ---
            rules_at_turn_end = {sig_id: rule for sig_id, rule in all_valid_signatures.items() if sig_id not in signatures_to_remove}
            exit_sig_count = len(rules_at_turn_end)
            exit_recall, exit_precision, exit_f1, exit_accuracy = 0, 0, 0, 0
            if rules_at_turn_end and turn_eval_data is not None and not turn_eval_data.empty:
                formatted_exit_sigs = [{'id': sid, 'name': f'Sig_{sid}', 'rule_dict': r} for sid, r in rules_at_turn_end.items()]
                
                # --- MODIFIED: Use batched performance calculation ---
                _, _, exit_recall, exit_precision, exit_f1, exit_accuracy = calculate_performance_in_batches(
                    turn_eval_data, 
                    formatted_exit_sigs, 
                    args.evaluation_batch_size
                )

            # FN RECOVERY CHECK
            recovered = check_fn_recovery(fn_removed_history, all_valid_signatures, turn_counter)
            recovered_info_latest = recovered
            recovered_ids_flat = set()
            for info in recovered.values():
                recovered_ids_flat.update(info["recovered_ids"])
            new_recovered_ids = recovered_ids_flat - fn_recovered_cumulative
            fn_recovered_cumulative.update(recovered_ids_flat)
            total_recovered = len(recovered_ids_flat)
            fn_recovered_delta = len(new_recovered_ids)

            # FP removal tracking (when injected FPs disappear)
            current_ids = set(all_valid_signatures.keys())
            newly_removed_fp = fp_alive - current_ids
            if newly_removed_fp:
                fp_removed_history[turn_counter] = newly_removed_fp
                fp_alive -= newly_removed_fp
            else:
                fp_removed_history[turn_counter] = set()

            # FP/FN event logging (per turn)
            fp_injected_ids = list(map(str, sorted(fp_injected_history.get(turn_counter, set()))))
            fn_removed_ids = list(map(str, sorted(fn_removed_history.get(turn_counter, set()))))
            fn_recovered_ids = list(map(str, sorted(new_recovered_ids)))
            fp_removed_ids = list(map(str, sorted(fp_removed_history.get(turn_counter, set()))))
            fpfn_events.append({
                "turn": turn_counter,
                "fp_injected_count": len(fp_injected_ids),
                "fp_injected_ids": "|".join(fp_injected_ids),
                "fp_removed_count": len(fp_removed_ids),
                "fp_removed_ids": "|".join(fp_removed_ids),
                "fn_removed_count": len(fn_removed_ids),
                "fn_removed_ids": "|".join(fn_removed_ids),
                "fn_recovered_count": len(fn_recovered_ids),
                "fn_recovered_ids": "|".join(fn_recovered_ids),
            })

            logger.info(f"End of Turn {turn_counter}. Signatures: {entry_sig_count} -> {exit_sig_count}. EXIT Recall: {exit_recall:.4f}. EXIT Precision: {exit_precision:.4f}. EXIT F1: {exit_f1:.4f}. EXIT Accuracy: {exit_accuracy:.4f}")
            history.append({
                'turn': turn_counter,
                'entry_signature_count': entry_sig_count,
                'generated': new_signatures_found,
                'removed': newly_removed_count,
                'exit_signature_count': exit_sig_count,
                'entry_recall': entry_recall,
                'entry_precision': entry_precision,
                'entry_f1': entry_f1,
                'entry_accuracy': entry_accuracy,
                'exit_recall': exit_recall,
                'exit_precision': exit_precision,
                'exit_f1': exit_f1,
                'exit_accuracy': exit_accuracy,
                'fn_forced_removed': fn_removed_count,
                'fn_recovered_total': total_recovered,
                'fn_recovered_delta': fn_recovered_delta,
                'fp_injected': len(injected),
                'fp_removed': len(fp_removed_ids)
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
        final_signatures = {sig_id: rule for sig_id, rule in all_valid_signatures.items() if sig_id not in signatures_to_remove}

        logger.info("--- Process Complete ---")
        logger.info(f"Initial unique signatures generated: {len(all_valid_signatures)}")
        logger.info(f"Signatures removed due to FPs: {len(signatures_to_remove)}")
        logger.info(f"Final count of validated signatures: {len(final_signatures)}")

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
        for sig_id, rule in final_signatures.items():
            created_turn = signature_turn_created.get(sig_id, None)
            attacks = extract_attack_types_for_signature(rule, chunk_cache, attack_cols, args.evaluation_batch_size)
            signature_records.append({
                'signature_rule': str(rule),
                'created_turn': created_turn,
                'attack_types': "|".join(sorted(attacks))
            })

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

            # Attach FP/FN per-turn counts for plotting
            history_df['fp_injected'] = history_df.get('fp_injected', 0)
            history_df['fp_removed'] = history_df.get('fp_removed', 0)
            history_df['fn_removed'] = history_df.get('fn_forced_removed', 0)
            history_df['fn_recovered_delta'] = history_df.get('fn_recovered_delta', 0)

            # Save FP/FN per-turn event log
            if fpfn_events:
                fpfn_events_df = pd.DataFrame(fpfn_events)
                events_filename = f"{args.file_type}_{args.file_number}_{param_str}_fpfn_events.csv"
                events_path = os.path.join(output_dir, events_filename)
                fpfn_events_df.to_csv(events_path, index=False)
                logger.info(f"FP/FN event log saved to: {events_path}")

            performance_filename = f"{args.file_type}_{args.file_number}_{param_str}_performance_history_eex.csv"
            performance_path = os.path.join(output_dir, performance_filename)
            history_df.to_csv(performance_path, index=False)
            logger.info(f"Performance history saved to: {performance_path}")

            if plt:
                logger.info("Generating performance graph...")
                
                # Create a single figure and a primary axis for performance metrics
                fig, ax_perf = plt.subplots(figsize=(18, 8)) # Wider figure for better readability
                fig.suptitle(f'Incremental Signature Performance for {args.file_type}\n(support={args.min_support}, confidence={args.min_confidence})', fontsize=16)

                # Create secondary axes for counts (main) and FP/FN (small counts)
                ax_counts = ax_perf.twinx()  # for Generated/Removed
                ax_counts_fpfn = ax_perf.twinx()  # for FP/FN small counts
                ax_counts_fpfn.spines["right"].set_position(("axes", 1.08))
                ax_counts_fpfn.spines["right"].set_visible(True)

                x_labels = []
                x_ticks = []
                #bar_width = 0.35
                base_bar_width = 0.12

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
                    #ax_counts.bar(bar_center - bar_width/2, row['generated'], bar_width, label='Generated' if i == 0 else "", color='green', alpha=0.6)
                    #ax_counts.bar(bar_center + bar_width/2, row['removed'], bar_width, label='Removed' if i == 0 else "", color='red', alpha=0.6)
                    # Main large counts (generated/removed) on ax_counts
                    ax_counts.bar(bar_center - 2.5*base_bar_width, row['generated'], base_bar_width, label='Generated' if i == 0 else "", color='green', alpha=0.6)
                    ax_counts.bar(bar_center - 1.5*base_bar_width, row['removed'], base_bar_width, label='Removed' if i == 0 else "", color='red', alpha=0.6)

                    # Smaller FP/FN counts on separate axis ax_counts_fpfn for visibility
                    ax_counts_fpfn.bar(bar_center - 0.5*base_bar_width, row['fp_injected'], base_bar_width, label='FP Injected' if i == 0 else "", color='teal', alpha=0.7)
                    ax_counts_fpfn.bar(bar_center + 0.5*base_bar_width, row['fp_removed'], base_bar_width, label='FP Removed' if i == 0 else "", color='cyan', alpha=0.7)
                    ax_counts_fpfn.bar(bar_center + 1.5*base_bar_width, row['fn_removed'], base_bar_width, label='FN Forced Removed' if i == 0 else "", color='orange', alpha=0.7)
                    ax_counts_fpfn.bar(bar_center + 2.5*base_bar_width, row['fn_recovered_delta'], base_bar_width, label='FN Recovered' if i == 0 else "", color='brown', alpha=0.7)

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

                # Scale FP/FN axis separately to make small counts visible
                max_fpfn = max(
                    1,
                    int(history_df[['fp_injected','fp_removed','fn_removed','fn_recovered_delta']].max().max())
                )
                ax_counts_fpfn.set_ylabel('FP/FN Count', color='dimgray')
                ax_counts_fpfn.tick_params(axis='y', labelcolor='dimgray')
                ax_counts_fpfn.set_ylim(0, max_fpfn * 1.2)

                # Combine legends from both axes
                handles_perf, labels_perf = ax_perf.get_legend_handles_labels()
                handles_counts, labels_counts = ax_counts.get_legend_handles_labels()
                handles_counts2, labels_counts2 = ax_counts_fpfn.get_legend_handles_labels()
                #ax_perf.legend(handles_perf + handles_counts, labels_perf + labels_counts, loc='best')
                ax_perf.legend(handles_perf + handles_counts + handles_counts2,
                               labels_perf + labels_counts + labels_counts2,
                               loc='best')

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                
                graph_dir = os.path.join(run_dir, "graphs")
                if not os.path.exists(graph_dir):
                    try:
                        os.makedirs(graph_dir)
                    except OSError as e:
                        logger.error(f"Could not create graph directory {graph_dir}: {e}")
                        graph_dir = run_dir
                
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

        # --- FP/FN final recovery summary ---
        final_turn = turn_counter if 'turn_counter' in locals() else 0
        recovered_final = check_fn_recovery(fn_removed_history, all_valid_signatures, final_turn + 1)
        fn_recovery_summary = summarize_fn_recovery(fn_removed_history, recovered_final)
        total_fp_injected = sum(len(v) for v in fp_injected_history.values())
        total_fp_removed = sum(len(v) for v in fp_removed_history.values())
        total_fn_removed = sum(len(v) for v in fn_removed_history.values())
        total_fn_recovered = fn_recovery_summary.get("fn_total_recovered", 0)

        # Save FP/FN summary CSV
        fpfn_summary_record = {
            "total_fp_injected": total_fp_injected,
            "total_fp_removed": total_fp_removed,
            "fp_removal_rate": (total_fp_removed / total_fp_injected) if total_fp_injected > 0 else 0,
            "total_fn_removed": total_fn_removed,
            "total_fn_recovered": total_fn_recovered,
            "fn_recovery_rate": fn_recovery_summary.get("fn_recovery_rate", 0),
            "fn_avg_latency": fn_recovery_summary.get("fn_avg_latency", 0),
        }
        fpfn_summary_df = pd.DataFrame([fpfn_summary_record])
        fpfn_summary_filename = f"{args.file_type}_{args.file_number}_{param_str}_fpfn_summary.csv"
        fpfn_summary_path = os.path.join(output_dir, fpfn_summary_filename)
        fpfn_summary_df.to_csv(fpfn_summary_path, index=False)
        logger.info(f"FP/FN summary saved to: {fpfn_summary_path}")

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
                "total_recall_final_turn_exit": final_recall,
                "total_fp_injected": total_fp_injected,
                "total_fn_removed": total_fn_removed,
                "total_fn_recovered": total_fn_recovered,
                "fn_recovery_rate": fn_recovery_summary.get("fn_recovery_rate", 0),
                "fn_avg_latency": fn_recovery_summary.get("fn_avg_latency", 0)
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
    parser.add_argument('--FP_control', action='store_true', help="Inject fake FP-causing signatures at specific turns")
    parser.add_argument('--FN_control', action='store_true', help="Force-remove important signatures at specific turns")
    parser.add_argument('--FP_turns', type=str, default="", help="Comma-separated turns to inject FP signatures (e.g., 3,5)")
    parser.add_argument('--FN_turns', type=str, default="", help="Comma-separated turns to remove signatures (e.g., 4)")
    parser.add_argument('--FN_strategy', type=str, default='early', choices=['early', 'oldest', 'newest', 'random'], help='FN removal strategy')
    parser.add_argument('--FN_remove_ratio', type=float, default=0.1, help='Ratio of signatures to forcibly remove')


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