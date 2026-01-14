# A program that automatically generates signatures using association rules.
# EXPERIMENTAL VERSION: Implements a memory-efficient signature filtering logic.

import argparse
import numpy as np
import time
import multiprocessing # Added for parallel processing
import functools # Added for functools.partial if needed, though starmap is used here
import os
import pandas as pd  # ADDED for the new helper function
from tqdm import tqdm
from Dataset_Choose_Rule.association_data_choose import get_clustered_data_path # MODIFIED: Import the new function
from Dataset_Choose_Rule.choose_amount_dataset import file_cut
from definition.Anomal_Judgment import anomal_judgment_label, anomal_judgment_nonlabel
from utils.time_transfer import time_scalar_transfer
from utils.class_row import anomal_class_data, without_labelmaking_out, nomal_class_data, without_label
from Modules.Heterogeneous_module import choose_heterogeneous_method
from Heterogeneous_Method.separate_group_mapping import map_intervals_to_groups
from utils.remove_rare_columns import remove_rare_columns
from Modules.Association_module import association_module
from Modules.Signature_evaluation_module import signature_evaluate
from Modules.Signature_underlimit import under_limit
# from Modules.Whitelist_module import create_whitelist
from Evaluation.calculate_signature import calculate_signatures
from Modules.Difference_sets import dict_list_difference
from Dataset_Choose_Rule.save_csv import csv_association
from Dataset_Choose_Rule.time_save import time_save_csv_CS_ex # Import the new function
import logging
from datetime import datetime

# Define the logger object at the module level so it can be used throughout the script.
# The actual configuration of the logger (level, format, handlers) will be done
# in the if __name__ == '__main__': block or in the main() function.
logger = logging.getLogger(__name__)

# --- NEW: Efficient Parallel Filtering Helpers (ported from ISV_ex.py) ---
# This global tuple will hold the data needed by the worker processes.
# It's a workaround to avoid passing large dataframes to each process repeatedly.
_worker_data_filter = ()

def _init_worker_filter(normal_data, min_support):
    """
    Initializer for each worker process in the pool. It sets up the 
    global dataframe for the process, passing the large data only once.
    """
    global _worker_data_filter
    _worker_data_filter = (normal_data, min_support)

def _is_rule_valid_for_filtering(rule):
    """
    The actual task for each worker process. It checks a single rule 
    against the globally available normal data.
    """
    global _worker_data_filter
    normal_data, min_support = _worker_data_filter
    
    is_frequent_in_normal = calculate_support_for_itemset(rule, normal_data, min_support)
    
    # We want to keep rules that are NOT frequent in normal data.
    if not is_frequent_in_normal:
        return rule  # Return the rule if it's a valid signature
    return None  # Return None if it should be filtered out
# --- END NEW HELPERS ---


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
    'netML': 5,
    'Kitsune': 5,
    'CICIoT': 4,
    'CICIoT2023': 4,
    'default': None
}

# NEW HELPER FUNCTION (integrated directly into this file for independence)
def calculate_support_for_itemset(itemset, df, min_support):
    """
    Calculates the support for a single itemset in a given dataframe.

    Args:
        itemset (dict): A dictionary representing the rule/itemset (e.g., {'feature1': 'value1', 'feature2': 'value2'}).
        df (pd.DataFrame): The dataframe to check against (e.g., nomal_grouped_data).
        min_support (float): The minimum support threshold.

    Returns:
        bool: True if the itemset's support is >= min_support, False otherwise.
    """
    if not itemset or df.empty:
        return False

    # Build a boolean mask by checking each condition in the itemset
    mask = pd.Series([True] * len(df), index=df.index)
    for key, value in itemset.items():
        if key in df.columns:
            mask &= (df[key] == value)
        else:
            # If a feature from the rule doesn't even exist in the dataframe,
            # it can't be a match.
            return False

    # Calculate support
    support = mask.sum() / len(df)

    return support >= min_support


# Helper function for parallel processing
# MODIFIED: Added cores_per_algo_task as the last argument
def process_confidence_iteration(min_confidence, anomal_grouped_data, nomal_grouped_data, Association_mathod, min_support, association_metric, group_mapped_df, signature_ea, precision_underlimit, cores_per_algo_task, current_file_type, itemset_limit):
    """Processes a single iteration of the confidence loop."""
    iteration_start_time = time.time()
    print(f"  [PCI Start] conf={min_confidence}, algo={Association_mathod}, algo_procs={cores_per_algo_task}") # New, more concise log
    
    max_level = LEVEL_LIMITS_BY_FILE_TYPE.get(current_file_type, LEVEL_LIMITS_BY_FILE_TYPE['default'])
    
    # --- Start of MAJOR LOGIC CHANGE based on user's idea ---

    # 1. Generate association rules from ANOMALY data ONLY.
    # This step can use all available cores as it's CPU-bound.
    print(f"  [PCI Mining] Generating association rules from anomaly data using {cores_per_algo_task} processes...")
    association_list_anomal, max_level_reached = association_module(
        anomal_grouped_data, Association_mathod, min_support, 
        min_confidence, association_metric, 
        num_processes=cores_per_algo_task, 
        file_type_for_limit=current_file_type, 
        max_level_limit=max_level,
        itemset_limit=itemset_limit # Pass the limit
    )
    
    # Memory management: Clear intermediate variables if they're large
    if len(association_list_anomal) > 1000000:  # If more than 1M rules
        print(f"  [MEMORY] Large number of rules detected ({len(association_list_anomal):,}). Applying memory optimization...")
        import gc
        gc.collect()  # Force garbage collection

    # 2. Instead of generating all normal rules, filter the anomaly rules by checking their support in NORMAL data.
    # --- NEW: Limit processes for filtering on memory-intensive datasets ---
    if current_file_type in ['CICIDS2017', 'CICIDS', 'Kitsune', 'CICIoT', 'CICIoT2023']:
        # For large datasets, using all cores for filtering can cause memory swapping.
        # Limit to a safer number of processes for this specific step.
        num_filter_processes = 12
    else:
        # For other datasets, use the same number of cores as mining.
        num_filter_processes = cores_per_algo_task
    
    print(f"  [PCI Filter] Filtering {len(association_list_anomal):,} anomaly rules against normal data using {num_filter_processes} processes...")
    signatures = []

    '''
    # Process in batches to manage memory
    batch_size = 100000  # Process 100K rules at a time
    for i in range(0, len(association_list_anomal), batch_size):
        batch_end = min(i + batch_size, len(association_list_anomal))
        batch_rules = association_list_anomal[i:batch_end]

        for anomal_rule in batch_rules:
            # Check if the rule is ALSO frequent in the normal data.
            is_frequent_in_normal = calculate_support_for_itemset(anomal_rule, nomal_grouped_data, min_support)
            
            # If the rule is NOT frequent in normal data, it's a potential signature.
            if not is_frequent_in_normal:
                signatures.append(anomal_rule)
        
        # Memory management: Clear batch variables
        del batch_rules
        
        if i % (batch_size * 5) == 0:  # Every 5 batches
            print(f"    [MEMORY] Processed {batch_end:,}/{len(association_list_anomal):,} rules. Current signatures: {len(signatures):,}")
            import gc
            gc.collect()  # Periodic garbage collection


    # --- START: Parallel Filtering Logic with Progress Bar ---
    check_func = functools.partial(calculate_support_for_itemset, df=nomal_grouped_data, min_support=min_support)

    # Use a list to store the results, as we need to preserve the order to match them back to the rules.
    is_frequent_results = [False] * len(association_list_anomal)
    signatures_to_add = []

    with multiprocessing.Pool(processes=num_filter_processes) as pool: # Use the adjusted number of processes
        # Use imap_unordered for efficiency, as the order of completion doesn't matter before the final assembly.
        # We wrap the pool iterator with tqdm to create the progress bar.
        chunk_size = max(1, len(association_list_anomal) // (num_filter_processes * 4))
        
        # We will process rules and their original indices to reconstruct the final list correctly.
        tasks = enumerate(association_list_anomal) # Provides (index, rule)
        
        # Wrap the imap_unordered iterator with tqdm for a live progress bar
        # The total is the number of rules to process.
        pbar = tqdm(pool.imap_unordered(check_func, association_list_anomal, chunksize=chunk_size), 
                    total=len(association_list_anomal),
                    desc="[PCI Filter] Progress")

    '''

    # --- EMERGENCY RESCUE BLOCK ---
    # This block is designed to save intermediate results if the user interrupts
    # the process, preventing the loss of long computation work.
    try:
        # --- REFACTORED: Use the efficient worker initializer pattern ---
        if not nomal_grouped_data.empty and association_list_anomal:
            with multiprocessing.Pool(
                processes=num_filter_processes,
                initializer=_init_worker_filter,
                initargs=(nomal_grouped_data, min_support)
            ) as pool:
                chunk_size = max(1, len(association_list_anomal) // (num_filter_processes * 10))
                
                results_iterator = pool.imap_unordered(_is_rule_valid_for_filtering, association_list_anomal, chunksize=chunk_size)

                pbar = tqdm(results_iterator, total=len(association_list_anomal), desc="[PCI Filter] Progress")
                
                for result_rule in pbar:
                    if result_rule is not None:
                        signatures.append(result_rule)
        else:
            signatures = association_list_anomal
        # --- END REFACTORED LOGIC ---
    
    except KeyboardInterrupt:
        print("\n[RESCUE] KeyboardInterrupt detected! Attempting to save intermediate results.", flush=True)
        try:
            import pickle
            # Save the crucial data
            with open('rescue_anomal_rules.pkl', 'wb') as f:
                pickle.dump(association_list_anomal, f)
            print("[RESCUE] Successfully saved anomaly rules to: rescue_anomal_rules.pkl", flush=True)

            with open('rescue_normal_data.pkl', 'wb') as f:
                pickle.dump(nomal_grouped_data, f)
            print("[RESCUE] Successfully saved normal data to: rescue_normal_data.pkl", flush=True)
            
            with open('rescue_minsupport.pkl', 'wb') as f:
                pickle.dump(min_support, f)
            print("[RESCUE] Successfully saved min_support value to: rescue_minsupport.pkl", flush=True)

            print("\n[RESCUE] Data saved successfully. Please manually stop the script now if it doesn't exit.", flush=True)
        except Exception as e:
            print(f"[RESCUE] FAILED to save data: {e}", flush=True)
        
        # Exit after rescue attempt
        import sys
        sys.exit(1)
    # --- END EMERGENCY RESCUE BLOCK ---

    print(f"  [PCI Filter] Found {len(signatures):,} rules that are unique to anomaly data.")

    # --- End of MAJOR LOGIC CHANGE ---

    signature_result = signature_evaluate(group_mapped_df, signatures)
    signature_sets = under_limit(signature_result, signature_ea, precision_underlimit)

    if not signature_sets: # signature_sets is empty or None
        current_recall = 0 # Or an appropriate default value for recall when no sets
    else:
        current_recall = calculate_signatures(group_mapped_df, signature_sets)

    # Debug prints for this iteration (optional, can be removed for cleaner output)
    print(f"  [PCI Finish] conf: {min_confidence}, "
          f"Anomal Rules Found: {len(association_list_anomal)}, "
          f"Final Signatures (after filtering): {len(signatures)}, "
          f"Signature Sets (after evaluation): {len(signature_sets) if signature_sets else 0}, "
          f"Recall: {current_recall:.4f}")
    
    # Final memory cleanup
    del association_list_anomal, signatures
    import gc
    gc.collect()
    
    total_time_per_iteration = time.time() - iteration_start_time
    return min_confidence, current_recall, signature_sets, total_time_per_iteration

def main():
    """
    Main function to execute the association rule mining process.
    """
    # CORRECTED: Full timing logic implementation
    total_start_time = time.time()
    timing_info = {}

    # argparser
    # Create an instance that can receive argument values
    parser = argparse.ArgumentParser(description='Argparser')

    # Set the argument values to be input (default value can be set)
    parser.add_argument('--file_type', type=str, default="MiraiBotnet")   # data file type
    parser.add_argument('--file_number', type=int, default=1)   # Detach files
    parser.add_argument('--train_test', type=int, default=0)    # train = 0, test = 1
    parser.add_argument('--heterogeneous', type=str, default="Normalized")   # Heterogeneous(Embedding) Methods
    parser.add_argument('--clustering', type=str, default="kmeans")   # Clustering Methods
    parser.add_argument('--eval_clustering_silhouette', type=str, default="n")
    parser.add_argument('--association', type=str, default="apriori")   # Association Rule
    parser.add_argument('--precision_underlimit', type=float, default=0.6)
    parser.add_argument('--signature_ea', type=int, default=15)
    parser.add_argument('--association_metric', type=str, default='confidence')
    parser.add_argument('--num_processes', type=int, default=None, help="Number of processes to use for the main pool (outer loop) and internal algorithms.")
    parser.add_argument('--itemset_limit', type=int, default=10000000, help="Safety limit for frequent itemsets to prevent memory overflow before rule generation.")

    # Save the above in args
    args = parser.parse_args()

    # Output the value of the input arguments
    file_type = args.file_type
    file_number = args.file_number
    train_tset = args.train_test
    heterogeneous_method = args.heterogeneous
    clustering_algorithm = args.clustering
    eval_clustering_silhouette = args.eval_clustering_silhouette
    Association_mathod = args.association
    precision_underlimit = args.precision_underlimit
    signature_ea = args.signature_ea
    association_metric = args.association_metric
    num_processes = args.num_processes

    logger.info(f"Global start time: {datetime.fromtimestamp(total_start_time).strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Data loading
    start = time.time()

    # MODIFIED: Use the new function to get the path to the clustered data
    file_path, file_number = get_clustered_data_path(file_type, file_number)
    # cut_type = str(input("Enter the data cut type: "))
    cut_type = 'all'
    data = file_cut(file_type, file_path, cut_type)
    '''
    # ADDED FOR DEBUGGING: Print column names after loading
    if file_type == 'NSL-KDD':
        print(f"DEBUG_NSL-KDD_COLUMNS: Columns in 'data' DataFrame after file_cut for NSL-KDD: {data.columns.tolist()}")
    '''

    timing_info['1_load_data'] = time.time() - start
    logger.info(f"Loading data from file: {file_path}")


    # 2. Handling judgments of Anomal or Nomal
    start = time.time()

    if file_type in ['MiraiBotnet', 'NSL-KDD', 'NSL_KDD']:
        data['label'], _ = anomal_judgment_nonlabel(file_type, data)
    elif file_type == 'netML':
        # print(f"[DEBUG netML MAR] Columns in 'data' DataFrame for netML before processing: {data.columns.tolist()}")
        data['label'] = data['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
    elif file_type == 'DARPA98':
        data['label'] = data['Class'].apply(lambda x: 0 if str(x).strip() == '-' else 1)
    elif file_type in ['CICIDS2017', 'CICIDS']:
        print(f"INFO: Processing labels for {file_type}. Mapping BENIGN to 0, others to 1.")
        # Ensure 'Label' column exists
        if 'Label' in data.columns:
            data['label'] = data['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
            logger.info(f"Applied BENIGN/Attack mapping for {file_type}.")
        else:
            logger.error(f"ERROR: 'Label' column not found in data for {file_type}. Cannot apply labeling.")
            # Potentially raise an error or exit if label column is critical and missing
            # For now, it will proceed and might fail later if 'label' is expected
            data['label'] = 0 # Default to 0 or some other placeholder if Label is missing
    elif file_type in ['CICModbus23', 'CICModbus']:
        data['label'] = data['Attack'].apply(lambda x: 0 if str(x).strip() == 'Baseline Replay: In position' else 1)
    elif file_type in ['IoTID20', 'IoTID']:
        data['label'] = data['Label'].apply(lambda x: 0 if str(x).strip() == 'Normal' else 1)
    elif file_type in ['CICIoT', 'CICIoT2023']:
        data['label'] = data['attack_flag']
    elif file_type == 'Kitsune':
        data['label'] = data['Label']
    else:
        # This is a fallback, ensure your file_type is covered above for specific handling
        logger.warning(f"WARNING: Using generic anomal_judgment_label for {file_type}.")
        data['label'] = anomal_judgment_label(data)

    timing_info['2_anomal_judgment'] = time.time() - start

    # RESTORED: Add the logger.info line here, after 'label' column is created, with a check.
    if 'label' in data.columns:
        try:
            nomal_count = len(data[data['label'] == 0])
            anomal_count = len(data) - nomal_count # Or len(data[data['label'] != 0]) if 1 is not the only anomal label
            logger.info(f"Label generation complete. Total: {len(data)}, Nomal transactions: {nomal_count}, Anomal transactions: {anomal_count}")
            logger.info(f"Label distribution in 'data' DataFrame after judgment:\n{data['label'].value_counts(dropna=False)}")
        except Exception as e:
            logger.error(f"Error logging label counts: {e}")
            logger.info(f"Data columns: {data.columns}") # Log columns if error occurs
    else:
        logger.warning("Could not find 'label' column in 'data' DataFrame after judgment block to log transaction counts.")


    # 3. Feature-specific embedding and preprocessing
    start = time.time()

    data = time_scalar_transfer(data, file_type)

    # regul = str(input("\nDo you want to Regulation? (Y/n): ")) # Whether to normalize or not
    regul = 'N'

    print(f"[DEBUG MAR Hetero] Columns in 'data' df before choose_heterogeneous_method for {file_type}: {{data.columns.tolist()}}") # Debug print added
    embedded_dataframe, feature_list, category_mapping, data_list = choose_heterogeneous_method(data, file_type, heterogeneous_method, regul)
    print("embedded_dataframe: ", embedded_dataframe)

    group_mapped_df, mapped_info_df = map_intervals_to_groups(embedded_dataframe, category_mapping, data_list, regul)
    print("mapped group: ", group_mapped_df)
    print("mapped_info: ", mapped_info_df)

    # Save mapped_info_df for Validate_Signature.py
    # Ensure the directory exists
    mapped_info_save_path_dir = f"../Dataset_Paral/signature/{file_type}/"
    if not os.path.exists(mapped_info_save_path_dir):
        os.makedirs(mapped_info_save_path_dir)
    mapped_info_save_path = f"{mapped_info_save_path_dir}{file_type}_{file_number}_mapped_info.csv"
    mapped_info_df.to_csv(mapped_info_save_path, index=False)
    print(f"Saved mapped_info_df to: {mapped_info_save_path}")

    # group_mapped_df['label'] = data['label']
    # DIAGNOSTIC PRINT: Check lengths before label assignment
    print(f"DIAGNOSTIC: Length of data: {len(data)}, Length of group_mapped_df: {len(group_mapped_df)}")

    # Original label assignment logic with added diagnostics for NSL-KDD
    if len(group_mapped_df) == len(data) and file_type == 'netML':
        # logger.info(f"[{file_type}] Assigning 'label' to group_mapped_df using .values for robust index handling.")
        group_mapped_df['label'] = data['label'].values
    elif file_type != 'netML':
        group_mapped_df['label'] = data['label']
    '''
    else:
        logger.critical(f"[{file_type}] CRITICAL: Length mismatch between group_mapped_df ({len(group_mapped_df)}) and data ({len(data)}). Cannot assign 'label'.")

    if file_type in ['NSL-KDD', 'NSL_KDD']:
        print(f"DIAGNOSTIC NSL-KDD: After group_mapped_df['label'] = data['label']")
        if 'label' in group_mapped_df.columns:
            print(f"  NaN count in group_mapped_df['label']: {group_mapped_df['label'].isna().sum()}")
            print(f"  Value counts (raw strings):\n{group_mapped_df['label'].value_counts(dropna=False).to_string()}")
        else:
            print("  DIAGNOSTIC NSL-KDD ERROR: 'label' column not found after assignment!")
    else: # netML and lengths differ
        logger.critical(f"[{file_type}] CRITICAL: Length mismatch for netML between group_mapped_df ({len(group_mapped_df)}) and data ({len(data)}). Cannot assign 'label'.")
        # To prevent KeyError, ensure 'label' column exists, even if problematic
        if 'label' not in group_mapped_df.columns:
            group_mapped_df['label'] = np.nan
    '''
    

    # ===== Check group_mapped_df before splitting =====
    # RESTORED: Final check of the dataframe shape and label distribution before splitting.
    print(f"Shape of group_mapped_df: {group_mapped_df.shape}")
    if 'label' in group_mapped_df.columns:
        print("Label distribution in group_mapped_df:")
        print(group_mapped_df['label'].value_counts())
    else:
        print("Warning: 'label' column not found in group_mapped_df before splitting.")
    # ====================================================


    # Information about how to set up association rule groups
    # MODIFIED: Use the 'cluster' column for association rule mining instead of 'label'
    print("INFO: Splitting data for association rules based on the 'cluster' column.")
    anomal_grouped_data = data[data['cluster'] == 1]
    anomal_grouped_data = without_labelmaking_out(anomal_grouped_data, ['label', 'cluster']) # Also remove original label

    print("anomal_grouped_data: ", anomal_grouped_data)
    # anomal_grouped_data is DataFrame
    # fl: feature list; Same contents but not used because it's not inside a DF.

    # Make nomal row
    nomal_grouped_data = data[data['cluster'] == 0]
    nomal_grouped_data = without_labelmaking_out(nomal_grouped_data, ['label', 'cluster']) # Also remove original label

    print("nomal_grouped_data: ", nomal_grouped_data)
    # nomal_grouped_data is DataFrame
    # flo: feature list; Same contents but not used because it's not inside a DF.

    timing_info['3_embedding'] = time.time() - start


    # 4. Set association statements (confidence ratios, etc.)
    start = time.time()

    if file_type in ['CICModbus23', 'CICModbus']:
        min_support = 0.1
    elif file_type in ['NSL-KDD', 'NSL_KDD', 'netML', 'MiraiBotnet']:
        min_support = 0.01
    elif file_type in ['DARPA98', 'DARPA']:
        min_support = 0.01
    elif file_type in ['Kitsune']:
        min_support = 0.05
    elif file_type in ['CICIDS2017', 'CICIDS']:
        min_support = 0.05
    elif file_type in ['CICIoT', 'CICIoT2023']:
        min_support = 0.05
    else:
        min_support = 0.2

    # Use a lower min_support value for NSL-KDD
    if file_type in ['NSL-KDD', 'NSL_KDD', 'netML', 'MiraiBotnet']:
        # Restore to previously successful settings
        min_support_ratio_for_rare = 0.1
        min_distinct = 2
        print(f"NSL-KDD settings for remove_rare_columns: min_support_ratio={min_support_ratio_for_rare}, min_distinct={min_distinct}")
    elif file_type in ['CICIDS2017', 'CICIDS']:
        min_support_ratio_for_rare = 0.1
        min_distinct = 2
    else:
        min_support_ratio_for_rare = 0.1   # Other dataset defaults, Kitsune should not go through this logic.
        min_distinct = 2 # Other dataset defaults

    best_confidence = 0.8    # Initialize the variables to change
    # Considering anomalies and nomals simultaneously
    
    if file_type in ['CICIDS2017', 'CICIDS']:
        confidence_values = [0.3]
    elif file_type in ['Kitsune']:
        confidence_values = [0.1]
    elif file_type in ['CICModbus23', 'CICModbus']:
        confidence_values = np.arange(0.1, 0.96, 0.05)
    # elif file_type in ['NSL-KDD', 'NSL_KDD']:
    #     confidence_values = np.arange(0.5, 0.96, 0.05)
    elif file_type in ['MiraiBotnet', 'NSL-KDD', 'NSL_KDD']:
        confidence_values = np.arange(0.1, 0.14, 0.05)
    elif file_type in ['CICIoT', 'CICIoT2023']:
        confidence_values = [0.1]
    else:
        confidence_values = np.arange(0.1, 0.96, 0.05)
    best_recall = 0

    print("min_support: ", min_support)
    
    # CORRECTED LOGIC: The complex if/elif/else was redundant.
    # Set the parameters directly with clear variable names.
    #min_support_ratio_for_rare = 0.1
    #min_distinct = 2
    
    print("Applying remove_rare_columns...")

    # --- MODIFIED: Ensure consistent column removal across normal and anomal data ---
    # 1. Prune normal data first to establish the definitive set of columns to keep.
    print("[Filter] Analyzing rare columns in NORMAL data...")
    normal_data_pruned = remove_rare_columns(
        nomal_grouped_data, 
        min_support_ratio=min_support_ratio_for_rare, 
        file_type=file_type, 
        min_distinct_frequent_values=min_distinct
    )
    
    # 2. Get the list of columns that were KEPT in the normal data.
    final_columns_to_keep = normal_data_pruned.columns
    print(f"[Info] Normal data is pruned. Keeping {len(final_columns_to_keep)} columns.")

    # 3. Prune the anomal data to keep ONLY the same columns as the pruned normal data.
    # This ensures both dataframes have the exact same feature set for valid comparison.
    original_anomal_cols = anomal_grouped_data.columns
    anomal_grouped_data_pruned = anomal_grouped_data.drop(
        columns=[col for col in original_anomal_cols if col not in final_columns_to_keep],
        errors='ignore'
    )

    last_signature_sets = None

    print("Starting parallel processing for confidence values...")

    # === NEW LOGIC for Hierarchical Parallelism ===
    # This logic replaces the previous hardcoded or simple min() logic.
    
    # 1. Determine the number of parallel processes for the main (outer) pool.
    # This is the number of confidence values we want to test in parallel.
    # It should not exceed the number of available cores.
    num_confidence_tasks = len(confidence_values)
    
    # MODIFICATION START: Prioritize the user-provided num_processes argument
    # If the user provides --num_processes, use that value. Otherwise, use all cores.
    if num_processes is not None:
        available_cores = num_processes
        print(f"INFO: Using user-specified number of processes: {num_processes}")
    else:
        available_cores = multiprocessing.cpu_count()
        print(f"INFO: --num_processes not set. Using all available CPU cores: {available_cores}")
    
    # 2. Calculate the number of cores to be allocated to each internal algorithm task.
    # The total available cores are distributed among the main pool's parallel tasks.
    # At least 1 core is guaranteed for each task.
    #main_pool_procs = min(num_confidence_tasks, available_cores) if num_confidence_tasks > 0 else 1
    
    # 3. Calculate the number of cores to be allocated to each internal algorithm task.
    # The total available cores are distributed among the main pool's parallel tasks.
    # At least 1 core is guaranteed for each task.
    #cores_per_algo_task = max(1, available_cores // main_pool_procs if main_pool_procs > 0 else available_cores)
    
    # NEW LOGIC: Force sequential processing for the outer confidence loop
    # to dedicate all parallel resources to the inner association rule algorithm.
    # This simplifies the parallel structure and avoids errors like from NumExpr.
    main_pool_procs = 1
    
    # --- REMOVED: Hardcoded 100-core limit ---
    # The following block that limited cores to 100 for certain datasets has been removed.
    # The script will now use all available cores for the rule mining phase.
    # The filtering phase will still be limited for memory-intensive datasets inside 'process_confidence_iteration'.
    #
    # if file_type in ['CICIDS2017', 'CICIDS']:
    #     cores_per_algo_task = min(100, available_cores)
    #     print(f"[MEMORY OPTIMIZATION] CICIDS2017 detected. Limiting processes to {cores_per_algo_task} to prevent memory overflow.")
    # else:
    #     cores_per_algo_task = available_cores
    #     print(f"[MEMORY OPTIMIZATION] Using all available cores: {cores_per_algo_task}")
    '''
    # MODIFIED: Limit the number of processes to prevent memory overflow
    # For large datasets like CICIDS2017, using all cores can cause OOM
    if file_type in ['CICIDS2017', 'CICIDS']:
        # Limit to 100 processes for memory-intensive datasets (user's preference)
        cores_per_algo_task = min(100, available_cores)
        print(f"[MEMORY OPTIMIZATION] CICIDS2017 detected. Limiting processes to {cores_per_algo_task} to prevent memory overflow.")
    #elif file_type in ['Kitsune']:
        # Limit to 40 processes for Kitsune
        #cores_per_algo_task = min(40, available_cores)
        #print(f"[MEMORY OPTIMIZATION] Kitsune detected. Limiting processes to {cores_per_algo_task} to prevent memory overflow.")
    else:
        # For other datasets, use all available cores
        cores_per_algo_task = available_cores
        print(f"[MEMORY OPTIMIZATION] Using all available cores: {cores_per_algo_task}")
    '''
    
    cores_per_algo_task = available_cores
    # --- END REMOVAL ---

    print("\n" + "="*50)
    print("Parallelism Configuration (Optimized for Algorithm Performance):")
    print(f"  - User-specified/Available CPU cores: {available_cores}")
    print(f"  - Number of confidence values to test: {num_confidence_tasks}")
    print(f"  - Main pool size (confidence tasks): {main_pool_procs} (Sequential Execution)")
    print(f"  - Cores allocated per internal algorithm: {cores_per_algo_task} (Memory-Optimized)")
    print("="*50 + "\n")

    # === OLD LOGIC - Kept as comments for reference ===
    # main_pool_procs = min(num_confidence_tasks, available_cores) if num_confidence_tasks > 0 else 1 # Original logic commented out
    # cores_per_algo_task = max(1, available_cores // main_pool_procs if main_pool_procs > 0 else available_cores) # Original logic
    
    # === OLD LOGIC - Kept as comments for reference ===
    # main_pool_procs = min(num_confidence_tasks, available_cores) if num_confidence_tasks > 0 else 1 # Original logic commented out
    
    # Force sequential processing for confidence values to prioritize internal algorithm parallelism
    # main_pool_procs = 1
    # print(f"Set main_pool_procs to 1 to prioritize internal algorithm parallelism.") # Redundant print, new prints below cover this
    
    # Internal algorithms will use all available cores
    # algo_internal_procs = available_cores 
    
    # print(f"Number of confidence values (tasks): {num_confidence_tasks}")
    # print(f"Available CPU cores: {available_cores}")
    # print(f"Main pool will run sequentially (main_pool_procs = {main_pool_procs}).")
    # print(f"Internal algorithms will use {algo_internal_procs} processes (algo_internal_procs).")

    # Determine num_processes for internal algorithm parallelization (algo_internal_procs)
    # The following block is COMMENTED OUT as it's replaced by the logic above to prioritize internal parallelism.
    # if main_pool_procs > 1:
    #     # If the main confidence loop is parallel, run internal algorithms sequentially
    #     algo_internal_procs = 1
    #     print(f"Main pool is parallel ({main_pool_procs} processes), so internal algorithms will run sequentially (algo_internal_procs = 1).")
    # else:
    #     # If the main confidence loop is sequential, internal algorithms can use all available cores
    #     algo_internal_procs = available_cores
    #     print(f"Main pool is sequential, so internal algorithms can use all available cores (algo_internal_procs = {available_cores}).")
    # 
    # print(f"Calculated algo_internal_procs (for each association algorithm): {algo_internal_procs}") # This print is now covered by the one above


    # Prepare arguments for the worker function (process_confidence_iteration)
    static_args = (
        anomal_grouped_data,
        nomal_grouped_data,
        Association_mathod,
        min_support,
        association_metric,
        group_mapped_df,
        signature_ea,
        precision_underlimit,
        cores_per_algo_task, # MODIFIED: Pass the dynamically calculated number of cores for the algorithm
        file_type
    )

    # Create a list of arguments for starmap: (min_confidence_value, *static_args)
    tasks = [(conf_val,) + static_args + (args.itemset_limit,) for conf_val in confidence_values]

    results = []
    try:
        # Use the determined main_pool_procs for the outer pool
        # Only use Pool if there's actual parallelism to be gained (more than 1 task and more than 1 process for pool)
        if main_pool_procs > 1 and len(tasks) > 1:
            print(f"Executing {len(tasks)} confidence tasks in parallel using a pool of {main_pool_procs} processes.")
            with multiprocessing.Pool(processes=main_pool_procs) as pool:
                results = pool.starmap(process_confidence_iteration, tasks)
        elif tasks: # If there are tasks but no parallelism needed for the pool (or only 1 task)
            print(f"Executing {len(tasks)} confidence tasks sequentially.")
            results = [process_confidence_iteration(*task_args) for task_args in tasks]
        else:
            print("No confidence tasks to execute.")

    except Exception as e:
        print(f"An error occurred during processing confidence iterations: {e}")

    print("Parallel processing finished. Aggregating results...")
    
    # Process results to find the best one
    if results: # Check if results were successfully populated
        for res_min_confidence, res_current_recall, res_signature_sets, res_total_time in results:
            if res_current_recall > best_recall:
                best_recall = res_current_recall
                best_confidence = res_min_confidence
                last_signature_sets = res_signature_sets
                total_time_per_iteration = res_total_time
    else:
        print("No results from parallel processing. Check for errors.")


    association_result = {
        'Verified_Signatures': last_signature_sets,
        'Recall': best_recall,
        'Best_confidence': best_confidence
    }
    print(association_result)

    # save = csv_association(file_type, file_number, Association_mathod, association_result, association_metric, signature_ea)
    # --- NEW: Prepare summary data for saving ---
    final_signature_count = len(last_signature_sets) if last_signature_sets else 0
    # Get the loop limit for the current file type from the dictionary
    loop_limit_for_file = LEVEL_LIMITS_BY_FILE_TYPE.get(file_type, LEVEL_LIMITS_BY_FILE_TYPE['default'])

    save = csv_association(
        file_type, 
        file_number, 
        Association_mathod, 
        association_result, 
        association_metric, 
        signature_ea,
        loop_limit=loop_limit_for_file,      # Pass loop limit
        signature_count=final_signature_count # Pass signature count
    )

    timing_info['5_excute_association_and_filter'] = time.time() - start

    # Full time history
    total_end_time = time.time()
    timing_info['0_total_time'] = total_end_time - total_start_time

    # Save time information as a CSV using the new function
    time_save_csv_CS_ex(
        file_type, 
        file_number, 
        Association_mathod, 
        timing_info, 
        association_result.get('Best_confidence'), 
        min_support
    )

    logger.info(f"Total execution time: {timing_info['0_total_time']:.2f} seconds")
    logger.info(f"Global end time: {datetime.fromtimestamp(total_end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n--- Final Results ---")
    print(association_result)

    return association_result


if __name__ == '__main__':
    # Setup basic logging configuration.
    # This ensures that if the script is run directly, logging is configured.
    # If this script is imported as a module, this block won't run, and the
    # importing module would be responsible for configuring logging if desired.
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    main() 