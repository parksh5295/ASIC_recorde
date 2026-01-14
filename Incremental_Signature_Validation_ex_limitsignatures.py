import pandas as pd
import argparse
import time
import os
import sys
import multiprocessing
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
# import gc

# --- NEW: Maximum number of signatures to maintain in the pool ---
MAX_SIGNATURES = 5000

LEVEL_LIMITS_BY_FILE_TYPE = {
    'MiraiBotnet': 5,
    'NSL-KDD': 5,
    'NSL_KDD': 5,
    'DARPA98': None,
    'DARPA': None,
    'CICIDS2017': 4,
    'CICIDS': 4,
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

try:
    from Dataset_Choose_Rule.association_data_choose import file_path_line_association
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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


#def preprocess_and_map_chunk(chunk_df, file_type, category_mapping, data_list):
def preprocess_and_map_chunk(chunk_df, file_type, category_mapping, args):
    """
    Applies the necessary preprocessing and mapping to a data chunk.
    This function now mirrors the core preprocessing logic of Main_Association_Rule.py
    to ensure consistent and correct data transformation for all datasets.
    """
    # 1. Labeling
    if file_type in ['MiraiBotnet', 'NSL-KDD', 'NSL_KDD']:
        chunk_df['label'], _ = anomal_judgment_nonlabel(file_type, chunk_df)
    elif file_type == 'netML':
        chunk_df['label'] = chunk_df['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
    elif file_type == 'DARPA98':
        chunk_df['label'] = chunk_df['Class'].apply(lambda x: 0 if str(x).strip() == '-' else 1)
    elif file_type in ['CICIDS2017', 'CICIDS']:
        if 'Label' in chunk_df.columns:
            chunk_df['label'] = chunk_df['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
        else:
            logger.warning(f"Chunk for {file_type} is missing 'Label' column. Defaulting label to 0.")
            chunk_df['label'] = 0 # Default
    elif file_type in ['CICModbus23', 'CICModbus']:
        chunk_df['label'] = chunk_df['Attack'].apply(lambda x: 0 if str(x).strip() == 'Baseline Replay: In position' else 1)
    elif file_type in ['IoTID20', 'IoTID']:
        chunk_df['label'] = chunk_df['Label'].apply(lambda x: 0 if str(x).strip() == 'Normal' else 1)
    elif file_type in ['CICIoT', 'CICIoT2023']:
        chunk_df['label'] = chunk_df['attack_flag']
    elif file_type == 'Kitsune':
        chunk_df['label'] = chunk_df['Label']
    elif file_type in ['CICIoT', 'CICIoT2023']:
        chunk_df['label'] = chunk_df['attack_flag']
    else:
        logger.warning(f"Using generic anomal_judgment_label for {file_type}.")
        chunk_df['label'] = anomal_judgment_label(chunk_df)

    # --- START: Replicating the full preprocessing pipeline from Main_Association_Rule.py ---
    
    # Step 1: Time scalar transfer
    processed_chunk = time_scalar_transfer(chunk_df, file_type)
    
    # Step 2: Set regulation parameter (as done in main script)
    regul = 'N'
    
    # --- FIX: Remove choose_heterogeneous_method call ---
    # choose_heterogeneous_method returns Interval objects, which are already mapped data
    # We don't need this step - we'll map directly using mapped_info.csv
    embedded_chunk = processed_chunk.copy()  # Use the time-scalar-transferred data directly
    
    # Create a minimal data_list structure that map_intervals_to_groups expects
    data_list_for_chunk = [
        pd.DataFrame({'placeholder_cat': [0] * len(embedded_chunk)}, index=embedded_chunk.index),  # categorical
        pd.DataFrame(),  # time features (not used for interval mapping)
        pd.DataFrame(),  # packet_length features (not used for interval mapping)  
        pd.DataFrame(),  # count features (not used for interval mapping)
        pd.DataFrame({'placeholder_bin': [0] * len(embedded_chunk)}, index=embedded_chunk.index)   # binary
    ]
    
    logger.info("DEBUG: Using direct mapping approach - bypassing choose_heterogeneous_method")
    # --- END FIX ---

    # --- DEBUG: Check data before mapping ---
    # CRITICAL: Check if columns match between embedded_chunk and interval mapping
    if 'interval' in category_mapping and isinstance(category_mapping['interval'], pd.DataFrame):
        common_cols = [col for col in embedded_chunk.columns if col in category_mapping['interval'].columns]
        if common_cols:
            logger.info(f"DEBUG: Found {len(common_cols)} matching columns for interval mapping")
        else:
            logger.error("DEBUG: CRITICAL ERROR - NO MATCHING COLUMNS FOUND!")
            logger.error(f"DEBUG: embedded_chunk columns: {list(embedded_chunk.columns[:5])}...")
            logger.error(f"DEBUG: interval mapping columns: {list(category_mapping['interval'].columns[:5])}...")
    
    # --- ADDITIONAL DEBUG: Check data types and values ---
    logger.info(f"DEBUG: embedded_chunk data types: {embedded_chunk.dtypes.value_counts().to_dict()}")
    logger.info(f"DEBUG: embedded_chunk sample values (first 3 rows, first 5 cols):")
    for i in range(min(3, len(embedded_chunk))):
        sample_values = embedded_chunk.iloc[i, :5].to_dict()
        logger.info(f"DEBUG: Row {i}: {sample_values}")
    
    # Check if embedded_chunk has any NaN values before mapping
    nan_before = embedded_chunk.isna().sum().sum()
    if nan_before > 0:
        logger.warning(f"DEBUG: embedded_chunk has {nan_before} NaN values BEFORE mapping!")
    else:
        logger.info("DEBUG: embedded_chunk has NO NaN values before mapping")
    # --- END ADDITIONAL DEBUG ---

    # --- FIX: Direct mapping implementation ---
    # Instead of using the problematic map_intervals_to_groups, implement direct mapping
    logger.info("DEBUG: Implementing direct mapping using mapped_info.csv rules")
    
    # Create a copy for mapping
    mapped_chunk = embedded_chunk.copy()
    
    # Apply interval mapping directly for each column
    mapping_applied = 0
    for col in mapped_chunk.columns:
        if col in category_mapping['interval'].columns:
            # Get the mapping rules for this column
            col_rules = category_mapping['interval'][col].dropna()
            if not col_rules.empty:
                # Apply the first rule (assuming it's the main mapping rule)
                rule = col_rules.iloc[0]
                if isinstance(rule, str) and ']=' in rule:
                    # Parse rule like "(0.999, 3981.0]=0" or "(3981.0, 17758.3]=1"
                    try:
                        # Extract the interval part and group part
                        interval_part, group_str = rule.split(']=')
                        group = int(group_str)
                        
                        # Parse the interval: "(0.999, 3981.0" -> lower=0.999, upper=3981.0
                        interval_part = interval_part.strip('()')  # Remove outer parentheses
                        lower_str, upper_str = interval_part.split(',')
                        lower = float(lower_str.strip())
                        upper = float(upper_str.strip())

                        # Note: (lower, upper] means lower < x <= upper
                        '''
                        mapped_chunk[col] = mapped_chunk[col].apply(
                            lambda x: group if (lower < x <= upper) else (group + 1)
                        )
                        '''
                        
                        # --- START: Robust, Vectorized Mapping ---
                        target_col = mapped_chunk[col]
                        
                        # Convert column to a numeric type for comparison
                        if pd.api.types.is_datetime64_any_dtype(target_col.dtype):
                            # If it's a datetime column, convert to Unix timestamp (seconds)
                            # .view('int64') gets nanoseconds, so divide by 10**9 for seconds
                            numeric_col = pd.to_numeric(target_col, errors='coerce').view('int64') // 10**9
                        else:
                            # For other types, just ensure they are numeric, coercing errors to NaN
                            numeric_col = pd.to_numeric(target_col, errors='coerce')
                            
                        # Use vectorized np.select for efficient mapping
                        conditions = [
                            (numeric_col > lower) & (numeric_col <= upper) & numeric_col.notna()
                        ]
                        choices = [group]
                        default_group = group + 1 # Default group for values outside the interval
                        
                        mapped_chunk[col] = np.select(conditions, choices, default=default_group)
                        # --- END: Robust, Vectorized Mapping ---
                        
                        mapping_applied += 1
                        logger.debug(f"DEBUG: Applied rule '{rule}' to column '{col}'")
                        
                        #logger.debug(f"DEBUG: Applied rule '{rule}' to column '{col}': ({lower}, {upper}] -> {group}")
                    except (ValueError, IndexError) as e:
                        logger.warning(f"DEBUG: Failed to parse or apply rule '{rule}' for column '{col}': {e}")
                        continue
    
    logger.info(f"DEBUG: Applied mapping to {mapping_applied} columns")
    
    # Use the directly mapped chunk instead of map_intervals_to_groups
    group_mapped_chunk = mapped_chunk
    # --- END FIX ---
    
    # --- DEBUG: Check data after mapping ---
    logger.info(f"DEBUG: After mapping - group_mapped_chunk shape: {group_mapped_chunk.shape}")
    if not group_mapped_chunk.empty:
        logger.info(f"DEBUG: After mapping - group_mapped_chunk columns: {list(group_mapped_chunk.columns)}")
        # Check for NaN values
        nan_counts = group_mapped_chunk.isna().sum()
        if nan_counts.sum() > 0:
            logger.warning(f"DEBUG: NaN values found in mapped chunk! Total NaN count: {nan_counts.sum()}")
            # Show columns with most NaN values
            nan_counts_sorted = nan_counts[nan_counts > 0].sort_values(ascending=False)
            logger.warning(f"DEBUG: Top 5 columns with NaN values: {nan_counts_sorted.head().to_dict()}")
        else:
            logger.info("DEBUG: No NaN values found in mapped chunk")
    else:
        logger.error("DEBUG: group_mapped_chunk is empty after mapping!")
    # --- END DEBUG ---

    # Re-apply the label from the original chunk
    if 'label' in chunk_df.columns:
        group_mapped_chunk['label'] = chunk_df['label'].values
        
    return group_mapped_chunk


def main(args):
    """
    Main function to run the incremental signature generation and validation process.
    """
    start_time = time.time()
    
    # --- REFACTORED: On-the-fly Mapping Generation ---
    logger.info("--- Initial Setup: Generating Mapping On-the-fly with n_splits=60 ---")
    
    file_path, _ = file_path_line_association(args.file_type, args.file_number)
    
    # 1. Load the FIRST chunk to create the mapping
    try:
        first_chunk = next(pd.read_csv(file_path, chunksize=args.chunk_size, low_memory=False))
    except StopIteration:
        logger.error("Data file is empty. Cannot proceed.")
        return
    except FileNotFoundError:
        logger.error(f"Data file not found at: {file_path}")
        return

    # 2. Preprocess the first chunk to generate the mapping dictionary
    logger.info("Preprocessing first chunk to create a new mapping with 60 splits...")
    
    # Comprehensive labeling for the first chunk, copied from preprocess_and_map_chunk
    if args.file_type in ['MiraiBotnet', 'NSL-KDD', 'NSL_KDD']:
        first_chunk['label'], _ = anomal_judgment_nonlabel(args.file_type, first_chunk)
    elif args.file_type == 'netML':
        first_chunk['label'] = first_chunk['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
    elif args.file_type == 'DARPA98':
        first_chunk['label'] = first_chunk['Class'].apply(lambda x: 0 if str(x).strip() == '-' else 1)
    elif args.file_type in ['CICIDS2017', 'CICIDS']:
        if 'Label' in first_chunk.columns:
            first_chunk['label'] = first_chunk['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
        else:
            first_chunk['label'] = 0
    elif args.file_type in ['CICModbus23', 'CICModbus']:
        first_chunk['label'] = first_chunk['Attack'].apply(lambda x: 0 if str(x).strip() == 'Baseline Replay: In position' else 1)
    elif args.file_type in ['IoTID20', 'IoTID']:
        first_chunk['label'] = first_chunk['Label'].apply(lambda x: 0 if str(x).strip() == 'Normal' else 1)
    elif args.file_type in ['CICIoT', 'CICIoT2023']:
        first_chunk['label'] = first_chunk['attack_flag']
    elif args.file_type == 'Kitsune':
        first_chunk['label'] = first_chunk['Label']
    else:
        first_chunk['label'] = anomal_judgment_label(first_chunk)

    # Time transfer for the first chunk
    processed_first_chunk = time_scalar_transfer(first_chunk, args.file_type)

    # Generate mapping with 60 splits from the first chunk
    _, _, category_mapping, data_list = choose_heterogeneous_method(
        processed_first_chunk.drop(columns=['label'], errors='ignore'), 
        args.file_type, 
        'Interval_inverse', 
        'N', 
        n_splits_override=60  # --- HARDCODED SPLIT COUNT ---
    )
    
    if not category_mapping or 'interval' not in category_mapping or category_mapping['interval'].empty:
        logger.error("FATAL: Failed to generate a valid category mapping from the first chunk.")
        return

    logger.info(f"Successfully generated new mapping with {len(category_mapping['interval'].columns)} features.")

    # The pre-computation logic for rare columns is removed as it's complex and depends on the old mapping logic.
    # This script will now run without rare column removal.
    columns_to_drop = []

    chunk_size = args.chunk_size
    
    try:
        data_iterator = pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)
    except FileNotFoundError:
        logger.error(f"Data file not found at: {file_path}")
        return

    if data_iterator is None:
        logger.error("Failed to create data iterator.")
        return

    all_valid_signatures = {}
    signatures_to_remove = set()
    turn_counter = 0
    
    history = []
    processed_data_so_far = pd.DataFrame()


    for chunk in data_iterator:
        turn_counter += 1
        logger.info(f"--- Processing Turn {turn_counter} (Rows {turn_counter*chunk_size - (chunk_size-1)} - {turn_counter*chunk_size}) ---")
        
        # --- Drop the pre-identified rare columns from the current chunk ---
        if columns_to_drop:
            chunk.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        # --- END NEW ---

        # --- NEW: Reset the blacklist every 2 turns ---
        # We check for turn_counter > 1 to avoid resetting at the very beginning.
        # The reset happens at the START of Turn 3, 5, 7, etc., effectively resetting after every 2 turns complete.
        if turn_counter > 1 and (turn_counter - 1) % 2 == 0:
            if signatures_to_remove:
                logger.warning(f"*** Resetting blacklist at the start of Turn {turn_counter}. "
                               f"Removing {len(signatures_to_remove)} signatures from the blacklist. ***")
                signatures_to_remove.clear()
            else:
                logger.info(f"*** Blacklist reset point at Turn {turn_counter}, but it was already empty. ***")
        # --- END NEW ---
        
        try:
            #mapped_chunk = preprocess_and_map_chunk(chunk, args.file_type, category_mapping, data_list)
            mapped_chunk = preprocess_and_map_chunk(chunk, args.file_type, category_mapping, args)
            processed_data_so_far = pd.concat([processed_data_so_far, mapped_chunk], ignore_index=True)
        except Exception as e:
            logger.error(f"Failed to preprocess chunk {turn_counter}: {e}")
            continue

        # --- 0. (NEW) Entry Performance Evaluation ---
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

        normal_data_in_chunk = mapped_chunk[mapped_chunk['label'] == 0].copy().drop(columns=['label'], errors='ignore')
        anomalous_data_in_chunk = mapped_chunk[mapped_chunk['label'] == 1].copy() # Keep label for association module drop

        # --- 1. Validation Step ---
        newly_removed_count = 0
        if all_valid_signatures and not normal_data_in_chunk.empty:
            logger.info(f"Validating {len(all_valid_signatures) - len(signatures_to_remove)} existing signatures against {len(normal_data_in_chunk)} normal data rows...")
            
            signatures_to_test = [
                {'id': sig_id, 'name': f'Sig_{sig_id}', 'rule_dict': rule}
                for sig_id, rule in all_valid_signatures.items() if sig_id not in signatures_to_remove
            ]

            if signatures_to_test:
                fp_alerts = apply_signatures_to_dataset(normal_data_in_chunk, signatures_to_test)
                
                if not fp_alerts.empty:
                    flagged_for_removal = set(fp_alerts['signature_id'].unique())
                    newly_flagged = flagged_for_removal - signatures_to_remove
                    newly_removed_count = len(newly_flagged)
                    if newly_removed_count > 0:
                        logger.warning(f"Found {newly_removed_count} new signatures causing False Positives in this turn. Flagging for removal.")
                        signatures_to_remove.update(newly_flagged)

        # --- 2. Generation Step ---
        new_signatures_found = 0
        if not anomalous_data_in_chunk.empty:
            #'''
            
            anomalous_features = anomalous_data_in_chunk.drop(columns=['label'], errors='ignore')

            # --- START: DIAGNOSTIC CODE ---
            logger.info("--- Data Sanity Check before Association ---")
            logger.info(f"Shape of data entering association module: {anomalous_features.shape}")
            for col in anomalous_features.columns:
                # Get the value counts for the current column
                counts = anomalous_features[col].value_counts()
                
                # Check if the most frequent item meets the support count threshold
                support_count_threshold = len(anomalous_features) * args.min_support
                most_frequent_count = counts.iloc[0] if not counts.empty else 0
                
                if most_frequent_count < support_count_threshold:
                    logger.warning(f"Column '{col}': Most frequent value appears {most_frequent_count} times, "
                                   f"which is LESS than the required support count of {support_count_threshold:.2f}. "
                                   f"This column will likely not produce any frequent 1-items.")
                else:
                    logger.info(f"Column '{col}': Most frequent value appears {most_frequent_count} times, "
                                f"MEETING the required support count of {support_count_threshold:.2f}.")
            logger.info("--- End of Sanity Check ---")
            # --- END: DIAGNOSTIC CODE ---

            # --- START: ENHANCED DIAGNOSTIC CODE ---
            logger.info("--- Data Sanity Check before Association ---")
            logger.info(f"Shape of data entering association module: {anomalous_features.shape}")
            support_count_threshold = len(anomalous_features) * args.min_support
            
            for col in anomalous_features.columns:
                nan_count = anomalous_features[col].isna().sum()
                
                if nan_count == len(anomalous_features):
                    logger.error(f"Column '{col}': ALL ({nan_count}) values are NaN. No frequent items possible.")
                    continue

                # dropna=False to see all values, including potential NaN if they are mixed
                counts = anomalous_features[col].value_counts(dropna=True)
                
                most_frequent_count = counts.iloc[0] if not counts.empty else 0
                
                logger.info(f"Column '{col}': NaN Count={nan_count}, Most frequent value appears {most_frequent_count} times.")

                if most_frequent_count < support_count_threshold:
                    logger.warning(f"  └─ This is LESS than the required support count of {support_count_threshold:.2f}.")
                else:
                    logger.info(f"  └─ This MEETS the required support count of {support_count_threshold:.2f}.")

            logger.info("--- End of Sanity Check ---")
            # --- END: ENHANCED DIAGNOSTIC CODE ---

            #'''

            logger.info(f"Generating new candidate rules from {len(anomalous_data_in_chunk)} anomalous data rows...")
            
            max_level = LEVEL_LIMITS_BY_FILE_TYPE.get(args.file_type, LEVEL_LIMITS_BY_FILE_TYPE['default'])


            anomalous_rules, _ = association_module(
                anomalous_data_in_chunk.drop(columns=['label'], errors='ignore'),
                association_rule_choose=args.association_method,
                min_support=args.min_support,
                min_confidence=args.min_confidence,
                association_metric='confidence',
                num_processes=args.num_processes,
                file_type_for_limit=args.file_type,
                max_level_limit=max_level,
                itemset_limit=args.itemset_limit # Pass the limit
            )
            
            # --- START of NEW PARALLEL FILTERING LOGIC ---
            # --- NEW: Limit processes for filtering on memory-intensive datasets ---
            if args.file_type in ['CICIDS2017', 'CICIDS', 'Kitsune', 'CICIoT', 'CICIoT2023']:
                # For large datasets, using all cores for filtering can cause memory swapping.
                # Limit to a safer number of processes for this specific step.
                num_filter_processes = 24
            else:
                # For other datasets, use the same number of cores as mining.
                num_filter_processes = args.num_processes

            logger.info(f"Filtering {len(anomalous_rules)} new rules against {len(normal_data_in_chunk)} normal rows in this chunk using {num_filter_processes} processes...")
            
            filtered_new_rules = []
            if not normal_data_in_chunk.empty and anomalous_rules:
                # Use a multiprocessing Pool to parallelize the filtering
                # The initializer passes large data once, improving efficiency.
                with multiprocessing.Pool(
                    #processes=args.num_processes,
                    processes=num_filter_processes,  # MODIFIED: Use the new limited process count
                    initializer=_init_worker_filter,
                    initargs=(normal_data_in_chunk, args.min_support)
                ) as pool:
                    # imap_unordered is memory-efficient for large result sets
                    results_iterator = pool.imap_unordered(_is_rule_valid_for_filtering, anomalous_rules, chunksize=1000)
                    
                    # --- NEW: Add progress bar for filtering step ---
                    pbar = tqdm(results_iterator, total=len(anomalous_rules), desc="[ISV] Filtering Rules")
                    # Process results as they complete
                    #for result_rule in results_iterator:
                    for result_rule in pbar:
                        if result_rule is not None:
                            filtered_new_rules.append(result_rule)
                    # --- END NEW ---
            else:
                # If there's no normal data to check against, all rules are considered valid for now.
                filtered_new_rules = anomalous_rules

            logger.info(f"{len(filtered_new_rules)} rules passed the intra-turn filter.")
            # --- END of NEW PARALLEL FILTERING LOGIC ---

            # --- MODIFIED: Add new rules with default precision ---
            for rule in filtered_new_rules:
                rule_id = hash(frozenset(rule.items()))
                # Add if not already present in the main pool or the removal list
                if rule_id not in all_valid_signatures and rule_id not in signatures_to_remove:
                    # Add with a default precision of 1.0 to ensure it gets evaluated fairly
                    all_valid_signatures[rule_id] = {'rule': rule, 'precision': 1.0}
                    new_signatures_found += 1
            
            logger.info(f"Added {new_signatures_found} new unique signatures to the main pool.")
        
        # --- 3. (MODIFIED) Exit Performance Evaluation and Culling ---
        rules_at_turn_end_with_meta = {
            sig_id: data 
            for sig_id, data in all_valid_signatures.items() 
            if sig_id not in signatures_to_remove
        }
        
        exit_recall, exit_precision, exit_f1 = 0, 0, 0
        
        if rules_at_turn_end_with_meta and not processed_data_so_far.empty:
            formatted_exit_sigs = [
                {'id': sid, 'name': f'Sig_{sid}', 'rule_dict': data['rule']} 
                for sid, data in rules_at_turn_end_with_meta.items()
            ]
            
            exit_alerts = apply_signatures_to_dataset(processed_data_so_far, formatted_exit_sigs)
            
            # --- Global performance calculation (for logging) ---
            exit_tp, exit_fp = 0, 0
            actual_positives_indices = set(processed_data_so_far[processed_data_so_far['label'] == 1].index)
            
            if not exit_alerts.empty:
                alerted_indices = exit_alerts['alert_index'].unique()
                actual_negatives_indices = set(processed_data_so_far[processed_data_so_far['label'] == 0].index)
                exit_tp = len(set(alerted_indices).intersection(actual_positives_indices))
                exit_fp = len(set(alerted_indices).intersection(actual_negatives_indices))

            total_anomalies_so_far = len(actual_positives_indices)
            total_exit_alerts = len(exit_alerts['alert_index'].unique()) if not exit_alerts.empty else 0
            
            exit_recall = exit_tp / total_anomalies_so_far if total_anomalies_so_far > 0 else 0
            exit_precision = exit_tp / total_exit_alerts if total_exit_alerts > 0 else 0
            if (exit_precision + exit_recall) > 0:
                exit_f1 = 2 * (exit_precision * exit_recall) / (exit_precision + exit_recall)

            # --- NEW: Per-signature precision calculation and update ---
            logger.info("Calculating precision for each signature...")
            active_sig_ids_in_alerts = set(exit_alerts['signature_id'].unique()) if not exit_alerts.empty else set()

            # Reset precision for all signatures that will be evaluated
            for sig_id in rules_at_turn_end_with_meta:
                if sig_id not in active_sig_ids_in_alerts:
                    # If a signature did not fire any alerts on the data so far, its precision is 0.
                    all_valid_signatures[sig_id]['precision'] = 0.0

            # Calculate and update precision for signatures that did fire alerts
            if not exit_alerts.empty:
                alerts_by_sig = exit_alerts.groupby('signature_id')
                for sig_id, sig_alerts in alerts_by_sig:
                    if sig_id in all_valid_signatures:
                        alerted_indices_sig = set(sig_alerts['alert_index'].unique())
                        tp_sig = len(alerted_indices_sig.intersection(actual_positives_indices))
                        total_alerts_sig = len(alerted_indices_sig)
                        precision_sig = tp_sig / total_alerts_sig if total_alerts_sig > 0 else 0.0
                        all_valid_signatures[sig_id]['precision'] = precision_sig
            
            # --- NEW: Culling logic to enforce MAX_SIGNATURES limit ---
            if len(all_valid_signatures) > MAX_SIGNATURES:
                logger.warning(f"Signature pool size ({len(all_valid_signatures)}) exceeds limit ({MAX_SIGNATURES}). Culling...")
                
                sorted_signatures = sorted(
                    all_valid_signatures.items(), 
                    key=lambda item: item[1]['precision'], 
                    reverse=True
                )
                
                top_signatures = sorted_signatures[:MAX_SIGNATURES]
                num_culled = len(all_valid_signatures) - len(top_signatures)
                logger.info(f"Culled {num_culled} signatures with the lowest precision.")
                
                all_valid_signatures = dict(top_signatures)

        # Recalculate the final count after potential culling
        rules_at_turn_end = {sig_id: data['rule'] for sig_id, data in all_valid_signatures.items() if sig_id not in signatures_to_remove}
        exit_sig_count = len(rules_at_turn_end)

        logger.info(f"End of Turn {turn_counter}. Signatures: {entry_sig_count} -> {exit_sig_count}. EXIT Recall: {exit_recall:.4f}. EXIT Precision: {exit_precision:.4f}. EXIT F1: {exit_f1:.4f}")

        # MODIFIED: Append all metrics
        history.append({
            'turn': turn_counter,
            'entry_signature_count': entry_sig_count,
            'generated': new_signatures_found,
            'removed': newly_removed_count,
            'exit_signature_count': exit_sig_count,
            'entry_recall': entry_recall,
            'entry_precision': entry_precision,
            'entry_f1': entry_f1,
            'exit_recall': exit_recall,
            'exit_precision': exit_precision,
            'exit_f1': exit_f1,
        })
        
        # --- START: END OF TURN MEMORY CLEANUP ---
        # Explicitly delete large, temporary objects created within this turn
        # to help the garbage collector free up memory sooner.
        try:
            del mapped_chunk
            del normal_data_in_chunk
            del anomalous_data_in_chunk
            if 'anomalous_rules' in locals():
                del anomalous_rules
            if 'filtered_new_rules' in locals():
                del filtered_new_rules
            if 'alerts' in locals():
                del alerts
            
            # Forcing garbage collection can be helpful in tight memory situations,
            # especially in long-running loops.
            # import gc
            # collected_count = gc.collect()
            # logger.info(f"End of Turn {turn_counter}. Garbage collection freed {collected_count} objects.")
        except NameError:
            # This might happen if an object wasn't created due to empty data, which is fine.
            pass
        # --- END: END OF TURN MEMORY CLEANUP ---


    # --- Finalization ---
    final_signatures = {sig_id: data['rule'] for sig_id, data in all_valid_signatures.items() if sig_id not in signatures_to_remove}
    
    logger.info("--- Process Complete ---")
    logger.info(f"Initial unique signatures generated over all turns: {len(all_valid_signatures)}")
    logger.info(f"Signatures in final blacklist: {len(signatures_to_remove)}")
    logger.info(f"Final count of validated signatures (after culling): {len(final_signatures)}")

    final_signatures_df = pd.DataFrame([{'signature_rule': str(rule)} for rule in final_signatures.values()])
    
    output_dir = f"../Dataset_Paral/validation/{args.file_type}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # UPDATED: Add parameters to filenames for clarity
    # Example: s0.3_c0.8_cs500_rarm
    param_str = f"{args.association_method}_s{args.min_support}_c{args.min_confidence}_cs{args.chunk_size}"
    
    output_filename = f"{args.file_type}_{args.file_number}_{param_str}_incremental_signatures_limitsignatures.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    final_signatures_df.to_csv(output_path, index=False)
    logger.info(f"Final signatures saved to: {output_path}")

    # --- PLOTTING and HISTORY CSV---
    if history:
        history_df = pd.DataFrame(history)
        
        # UPDATED: Use the same param_str for all filenames
        performance_filename = f"{args.file_type}_{args.file_number}_{param_str}_performance_history_limitsignatures.csv"
        performance_path = os.path.join(output_dir, performance_filename)
        history_df.to_csv(performance_path, index=False)
        logger.info(f"Performance history saved to: {performance_path}")

        if plt:
            logger.info("Generating performance graph...")
            # history_df is already created above
            
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
            
            # UPDATED: Use the same param_str for all filenames
            graph_filename = f"{args.file_type}_{args.file_number}_{param_str}_metrics_limitsignatures.jpg"
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
    
    # --- NEW: Argument for sampling ---
    parser.add_argument('--sample_10pct', action='store_true',
                        help='(Optional) If set and file_type is CICIoT2023, samples 10%% of the dataset for faster evaluation.')
                        # for reproducibility

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
