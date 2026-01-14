import pandas as pd
import argparse
import time
import os
import sys
import multiprocessing
import logging
from datetime import datetime
import copy # --- NEW: Import copy module for deepcopy ---
import numpy as np # --- NEW: Import numpy for direct mapping ---
import re # --- NEW: Import re for robust interval parsing ---
# import gc

LEVEL_LIMITS_BY_FILE_TYPE = {
    'MiraiBotnet': 3,
    'NSL-KDD': 3,
    'NSL_KDD': 3,
    'DARPA98': None,
    'DARPA': None,
    'CICIDS2017': 3,
    'CICIDS': 3,
    'CICModbus23': None,
    'CICModbus': None,
    'IoTID20': None,
    'IoTID': None,
    'CICIoT': 3,
    'CICIoT2023': 3,
    'netML': 5,
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
from Heterogeneous_Method.Feature_Encoding import Heterogeneous_Feature_named_featrues # --- NEW: Import for protection list ---

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
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib is not installed. Plotting functionality will be disabled.")
    plt = None


# --- Logger Setup ---
logger = logging.getLogger(__name__)
# MODIFIED: Set level to DEBUG to see detailed logs
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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

def calculate_and_log_support_stats(df, current_min_support, turn_counter):
    """
    Analyzes the support of individual items in the anomalous data chunk and logs guidance.
    """
    if df.empty:
        logger.debug(f"  [Support Analysis] Turn: {turn_counter}, Anomalous data is empty. Skipping analysis.")
        return

    total_rows = len(df)
    
    # Use pandas to efficiently calculate support for all items at once
    # Unpivot the dataframe to get all items and their counts
    unpivoted = df.melt(var_name='feature', value_name='value')
    # FIX: Convert all values to string to prevent TypeError during groupby/sort
    unpivoted['value'] = unpivoted['value'].astype(str)
    # Count occurrences of each feature-value pair
    item_counts = unpivoted.groupby(['feature', 'value']).size()
    
    # Calculate support for each item and sort
    item_supports = (item_counts / total_rows).sort_values(ascending=False)
    
    if item_supports.empty:
        logger.warning(f"  [Support Analysis] Turn: {turn_counter}, No items found in anomalous data to analyze.")
        return

    # --- New Calculations ---
    max_support = item_supports.iloc[0]
    median_support = item_supports.median()
    count_above_threshold = len(item_supports[item_supports >= current_min_support])

    logger.debug(f"--- Support Analysis for Turn {turn_counter} ---")
    
    # --- Top 5 ---
    logger.debug(f"Top 5 most frequent items:")
    for (feature, value), support in item_supports.head(5).items():
        logger.debug(f"    - Item: {{'{feature}': {value}}}, Support: {support:.4f}")

    # --- Bottom 5 ---
    logger.debug(f"Bottom 5 least frequent items:")
    # Use .sort_values() again to display tail correctly if there are ties
    for (feature, value), support in item_supports.tail(5).sort_values(ascending=True).items():
        logger.debug(f"    - Item: {{'{feature}': {value}}}, Support: {support:.4f}")
        
    # --- Summary Stats ---
    logger.debug(f"Support Stats: Max={max_support:.4f}, Median={median_support:.4f}")
    
    # --- Guidance ---
    logger.info(f"  [Support Guidance] To generate rules for this chunk, min_support must be <= {max_support:.4f}.")
    logger.info(f"  [Support Guidance] There are {count_above_threshold} unique items with support >= current min_support ({current_min_support}).")


    if current_min_support > max_support:
        logger.warning(f"  [Support Alert] Current min_support ({current_min_support}) is HIGHER than the max possible support ({max_support:.4f}). "
                       f"NO rules will be generated for this chunk.")
    logger.debug(f"--- End Support Analysis ---")


# NEW HELPER FUNCTION (integrated directly into this file for independence)
def calculate_support_for_itemset(itemset, df, min_support):
    """
    Calculates the support for a single itemset in a given dataframe.

    Args:
        itemset (dict): A dictionary representing the rule/itemset.
        df (pd.DataFrame): The dataframe to check against.
        min_support (float): The minimum support threshold.

    Returns:
        bool: True if the itemset's support is >= min_support, False otherwise.
    """
    if not itemset or df.empty:
        return False
    mask = pd.Series([True] * len(df), index=df.index)
    for key, value in itemset.items():
        if key in df.columns:
            mask &= (df[key] == value)
        else:
            return False
    support = mask.sum() / len(df)
    return support >= min_support


# --- START: New Helper function for parallel filtering ---
# This global tuple will hold the data needed by the worker processes.
# It's a workaround to avoid passing large dataframes to each process,
# which can be slow due to serialization (pickling).
_worker_data_filter = ()

def _init_worker_filter(normal_data, min_support):
    """
    Initializer for each worker process in the pool.
    It sets up the global dataframe for the process.
    """
    global _worker_data_filter
    _worker_data_filter = (normal_data, min_support)

def _is_rule_valid_for_filtering(rule):
    """
    The actual task for each worker process.
    It checks a single rule against the global normal data.
    """
    global _worker_data_filter
    normal_data, min_support = _worker_data_filter

    is_frequent_in_normal = calculate_support_for_itemset(rule, normal_data, min_support)

    # We want to keep rules that are NOT frequent in normal data.
    if not is_frequent_in_normal:
        return rule  # Return the rule if it's a valid signature
    return None  # Return None if it should be filtered out
# --- END: New Helper function for parallel filtering ---


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
    start_time = time.time()
    logger.info("--- Initial Setup: Generating Mapping On-the-fly ---")
    
    # MODIFIED: Use get_clustered_data_path to load data with 'cluster' column
    file_path, total_rows = get_clustered_data_path(args.file_type, args.file_number)
    
    try:
        chunk_iterator = pd.read_csv(file_path, chunksize=args.chunk_size, low_memory=False)
        first_chunk = next(chunk_iterator)
    except (StopIteration, FileNotFoundError) as e:
        logger.error(f"Could not read the first chunk from {file_path}: {e}")
        return

    # FIX: Time transfer is now the VERY FIRST step, as it should be.
    processed_first_chunk = time_scalar_transfer(first_chunk, args.file_type)
    
    # MODIFIED: The extensive labeling logic is removed. 
    # The 'label' for evaluation and 'cluster' for association are already present in the loaded data.
    logger.info("Labeling is skipped as 'label' and 'cluster' columns are pre-loaded.")

    # Generate mapping rules from the fully preprocessed first chunk
    # MODIFIED: Drop both 'label' and 'cluster' before creating mapping rules
    mapping_features = processed_first_chunk.drop(columns=['label', 'cluster'], errors='ignore')
    mapped_first_chunk_features, _, category_mapping, _ = choose_heterogeneous_method(
        mapping_features,
        args.file_type, 'Interval_inverse', 'N', n_splits_override=args.n_splits
    )
    
    if not category_mapping or 'interval' not in category_mapping or category_mapping['interval'].empty:
        logger.error("FATAL: Failed to generate a valid category mapping.")
        return

    logger.info(f"Successfully generated new mapping with {len(category_mapping['interval'].columns)} features.")
    
    # Create the fully mapped first chunk
    mapped_first_chunk = mapped_first_chunk_features
    # Re-attach both label and cluster columns
    mapped_first_chunk['label'] = processed_first_chunk['label'].values
    mapped_first_chunk['cluster'] = processed_first_chunk['cluster'].values
    
    # --- DEBUG: Log data state AFTER mapping ---
    log_dataframe_debug_info(mapped_first_chunk, "First Chunk (After Mapping)")

    # Clean data_list for subsequent turns
    data_list = [pd.DataFrame(), pd.DataFrame()] # --- THIS REMAINS IMPORTANT ---
    
    # ... (The rest of the main function, including the loop, should be reviewed to use this new flow) ...
    # Initialize state variables
    all_valid_signatures = {}
    signatures_to_remove = set()
    history = []
    processed_data_so_far = pd.DataFrame()

    # --- Main Loop ---
    # Manually create a generator to process the first chunk, then the rest
    def chunk_generator():
        yield mapped_first_chunk
        for chunk in chunk_iterator:
            # This is where the main loop's preprocessing will happen
            try:
                # 1. Time Transfer
                processed_chunk = time_scalar_transfer(chunk, args.file_type)
                
                # 2. Drop Columns
                if columns_to_drop:
                    processed_chunk.drop(columns=columns_to_drop, inplace=True, errors='ignore')

                # 3. Label and Map
                mapped_chunk = preprocess_and_map_chunk(processed_chunk, args.file_type, category_mapping, data_list)
                yield mapped_chunk
            except Exception as e:
                 logger.error(f"Failed to preprocess a subsequent chunk: {e}", exc_info=True)
                 # Decide if you want to skip this chunk or stop
                 continue 
    
    turn_counter = 0
    for mapped_chunk in chunk_generator():
        turn_counter += 1
        
        if turn_counter == 1:
            logger.info(f"--- Processing Turn 1 (Initial Chunk) ---")
        else:
            logger.info(f"--- Processing Turn {turn_counter} (Rows {(turn_counter-1)*args.chunk_size + 1} - {turn_counter*args.chunk_size}) ---")
        
        # The mapped_chunk is now ready for the main logic
        processed_data_so_far = pd.concat([processed_data_so_far, mapped_chunk], ignore_index=True)
        
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
        entry_recall, entry_precision, entry_f1 = 0, 0, 0
        rules_at_turn_start = {sig_id: rule for sig_id, rule in all_valid_signatures.items() if sig_id not in signatures_to_remove}
        entry_sig_count = len(rules_at_turn_start)
        if rules_at_turn_start and not processed_data_so_far.empty:
            formatted_entry_sigs = [{'id': sid, 'name': f'Sig_{sid}', 'rule_dict': r} for sid, r in rules_at_turn_start.items()]
            entry_alerts = apply_signatures_to_dataset(processed_data_so_far, formatted_entry_sigs)

            entry_tp, entry_fp = 0, 0
            if not entry_alerts.empty:
                alerted_indices = entry_alerts['alert_index'].unique()
                actual_positives_indices = processed_data_so_far[processed_data_so_far['label'] == 1].index
                actual_negatives_indices = processed_data_so_far[processed_data_so_far['label'] == 0].index
                entry_tp = len(set(alerted_indices).intersection(set(actual_positives_indices)))
                entry_fp = len(set(alerted_indices).intersection(set(actual_negatives_indices)))

            total_anomalies_so_far = (processed_data_so_far['label'] == 1).sum()
            total_entry_alerts = len(entry_alerts['alert_index'].unique()) if not entry_alerts.empty else 0

            entry_recall = entry_tp / total_anomalies_so_far if total_anomalies_so_far > 0 else 0
            entry_precision = entry_tp / total_entry_alerts if total_entry_alerts > 0 else 0
            if (entry_precision + entry_recall) > 0:
                entry_f1 = 2 * (entry_precision * entry_recall) / (entry_precision + entry_recall)
            logger.info(f"Turn {turn_counter} ENTRY Performance - Recall: {entry_recall:.4f}, Precision: {entry_precision:.4f}, F1: {entry_f1:.4f}")

        # MODIFIED: Split chunk data based on the 'cluster' column
        normal_data_in_chunk = mapped_chunk[mapped_chunk['cluster'] == 0].copy().drop(columns=['label', 'cluster'], errors='ignore')
        # Keep label for other potential uses, but it will be dropped before association
        anomalous_data_in_chunk = mapped_chunk[mapped_chunk['cluster'] == 1].copy()

        # --- 1. Validation Step ---
        newly_removed_count = 0
        newly_flagged_for_removal = set() # Store IDs of signatures removed THIS turn
        if all_valid_signatures and not normal_data_in_chunk.empty:
            logger.info(f"Validating {len(all_valid_signatures) - len(signatures_to_remove)} existing signatures against {len(normal_data_in_chunk)} normal data rows...")
            signatures_to_test = [ {'id': sig_id, 'name': f'Sig_{sig_id}', 'rule_dict': rule} for sig_id, rule in all_valid_signatures.items() if sig_id not in signatures_to_remove ]
            if signatures_to_test:
                fp_alerts = apply_signatures_to_dataset(normal_data_in_chunk, signatures_to_test)
                if not fp_alerts.empty:
                    flagged_for_removal = set(fp_alerts['signature_id'].unique())
                    newly_flagged_for_removal = flagged_for_removal - signatures_to_remove
                    newly_removed_count = len(newly_flagged_for_removal)
                    if newly_removed_count > 0:
                        logger.warning(f"Found {newly_removed_count} new signatures causing FPs. Flagging for removal.")
                        signatures_to_remove.update(newly_flagged_for_removal)

        # --- 2. Generation Step ---
        new_signatures_found = 0
        anomalous_rules = [] # A list to hold all generated rules for this turn

        # --- Standard Generation from the current chunk ---
        if not anomalous_data_in_chunk.empty:
            logger.info(f"Generating new candidate rules from {len(anomalous_data_in_chunk)} anomalous data rows...")
            
            # --- NEW: Analyze and log support stats for parameter guidance ---
            # MODIFIED: Drop 'label' and 'cluster' before analysis
            calculate_and_log_support_stats(anomalous_data_in_chunk.drop(columns=['label', 'cluster'], errors='ignore'), args.min_support, turn_counter)

            # --- DEBUG LOGGING: Show parameters before rule generation ---
            logger.debug(f"  [Association Params] Turn: {turn_counter}, "
                         f"Anomalous Rows: {len(anomalous_data_in_chunk)}, "
                         f"min_support: {args.min_support}, "
                         f"min_confidence: {args.min_confidence}")

            max_level = LEVEL_LIMITS_BY_FILE_TYPE.get(args.file_type, LEVEL_LIMITS_BY_FILE_TYPE['default'])
            standard_rules, _ = association_module(
                # MODIFIED: Drop 'label' and 'cluster' before passing to association module
                anomalous_data_in_chunk.drop(columns=['label', 'cluster'], errors='ignore'),
                association_rule_choose=args.association_method,
                min_support=args.min_support,
                min_confidence=args.min_confidence,
                association_metric='confidence',
                num_processes=args.num_processes,
                file_type_for_limit=args.file_type,
                max_level_limit=max_level,
                itemset_limit=args.itemset_limit
            )
            if standard_rules:
                anomalous_rules.extend(standard_rules) # Add to this turn's rule list

        else:
            logger.info(f"No anomalous data found in the current chunk. Skipping new rule generation for Turn {turn_counter}.")

        # --- Filtering and Adding all new rules for this turn ---
        if anomalous_rules:
            logger.info(f"Filtering {len(anomalous_rules)} new rules...")
            filtered_new_rules = []
            if not normal_data_in_chunk.empty:
                with multiprocessing.Pool(processes=args.num_processes, initializer=_init_worker_filter, initargs=(normal_data_in_chunk, args.min_support)) as pool:
                    results_iterator = pool.imap_unordered(_is_rule_valid_for_filtering, anomalous_rules, chunksize=1000)
                    for result_rule in results_iterator:
                        if result_rule is not None:
                            filtered_new_rules.append(result_rule)
            else:
                filtered_new_rules = anomalous_rules
            
            logger.info(f"{len(filtered_new_rules)} rules passed filter.")
            for rule in filtered_new_rules:
                rule_id = hash(frozenset(rule.items()))
                if rule_id not in all_valid_signatures and rule_id not in signatures_to_remove:
                    all_valid_signatures[rule_id] = rule
                    new_signatures_found += 1
            logger.info(f"Added {new_signatures_found} new unique signatures.")

        # --- 3. (MODIFIED) Exit Performance Evaluation ---
        rules_at_turn_end = {sig_id: rule for sig_id, rule in all_valid_signatures.items() if sig_id not in signatures_to_remove}
        exit_sig_count = len(rules_at_turn_end)
        exit_recall, exit_precision, exit_f1 = 0, 0, 0
        if rules_at_turn_end and not processed_data_so_far.empty:
            formatted_exit_sigs = [{'id': sid, 'name': f'Sig_{sid}', 'rule_dict': r} for sid, r in rules_at_turn_end.items()]
            exit_alerts = apply_signatures_to_dataset(processed_data_so_far, formatted_exit_sigs)
            exit_tp, exit_fp = 0, 0
            if not exit_alerts.empty:
                alerted_indices = exit_alerts['alert_index'].unique()
                actual_positives_indices = processed_data_so_far[processed_data_so_far['label'] == 1].index
                exit_tp = len(set(alerted_indices).intersection(set(actual_positives_indices)))
            total_anomalies_so_far = (processed_data_so_far['label'] == 1).sum()
            total_exit_alerts = len(exit_alerts['alert_index'].unique()) if not exit_alerts.empty else 0
            exit_recall = exit_tp / total_anomalies_so_far if total_anomalies_so_far > 0 else 0
            exit_precision = exit_tp / total_exit_alerts if total_exit_alerts > 0 else 0
            if (exit_precision + exit_recall) > 0:
                exit_f1 = 2 * (exit_precision * exit_recall) / (exit_precision + exit_recall)
        
        logger.info(f"End of Turn {turn_counter}. Signatures: {entry_sig_count} -> {exit_sig_count}. EXIT Recall: {exit_recall:.4f}. EXIT Precision: {exit_precision:.4f}. EXIT F1: {exit_f1:.4f}")
        history.append({'turn': turn_counter, 'entry_signature_count': entry_sig_count, 'generated': new_signatures_found, 'removed': newly_removed_count, 'exit_signature_count': exit_sig_count, 'entry_recall': entry_recall, 'entry_precision': entry_precision, 'entry_f1': entry_f1, 'exit_recall': exit_recall, 'exit_precision': exit_precision, 'exit_f1': exit_f1})

    # --- Finalization ---
    final_signatures = {sig_id: rule for sig_id, rule in all_valid_signatures.items() if sig_id not in signatures_to_remove}

    logger.info("--- Process Complete ---")
    logger.info(f"Initial unique signatures generated: {len(all_valid_signatures)}")
    logger.info(f"Signatures removed due to FPs: {len(signatures_to_remove)}")
    logger.info(f"Final count of validated signatures: {len(final_signatures)}")

    final_signatures_df = pd.DataFrame([{'signature_rule': str(rule)} for rule in final_signatures.values()])

    output_dir = f"../Dataset_Paral/validation/{args.file_type}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # UPDATED: Add parameters to filenames for clarity
    param_str = f"{args.association_method}_s{args.min_support}_c{args.min_confidence}_cs{args.chunk_size}"

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
                
                # Plot Adaptation phase (dotted line)
                if i < len(history_df) - 1:
                    next_row = history_df.iloc[i+1]
                    ax_perf.plot([x_exit, x_exit + 1], [row['exit_recall'], next_row['entry_recall']], 'o--', color='blue', alpha=0.5, label='Recall (Adaptation)' if i == 0 else "")
                    ax_perf.plot([x_exit, x_exit + 1], [row['exit_precision'], next_row['entry_precision']], 'x--', color='purple', alpha=0.5, label='Precision (Adaptation)' if i == 0 else "")
                    ax_perf.plot([x_exit, x_exit + 1], [row['exit_f1'], next_row['entry_f1']], 's--', color='orange', alpha=0.5, label='F1-Score (Adaptation)' if i == 0 else "")

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

    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Incrementally generate and validate signatures from a dataset.")
    parser.add_argument('--file_type', type=str, default="MiraiBotnet", help="Type of the dataset file.")
    parser.add_argument('--file_number', type=int, default=1, help="Number of the dataset file.")
    parser.add_argument('--association_method', type=str, default='rarm', help="Association rule algorithm to use.")
    parser.add_argument('--min_support', type=float, default=0.3, help="Minimum support for association rule mining.")
    parser.add_argument('--min_confidence', type=float, default=0.8, help="Minimum confidence for association rule mining.")
    # MODIFIED: Default to None to detect if the user has provided a value.
    parser.add_argument('--num_processes', type=int, default=None, help="Number of processes to use for parallel tasks. Defaults to all available cores.")
    parser.add_argument('--chunk_size', type=int, default=500, help="Number of rows to process in each incremental turn.")
    parser.add_argument('--itemset_limit', type=int, default=10000000, help="Safety limit for frequent itemsets to prevent memory overflow before rule generation.")
    parser.add_argument('--n_splits', type=int, default=40, help="Number of splits to use for dynamic interval mapping. Default is 40.")

    cli_args = parser.parse_args()

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