import pandas as pd
import pickle
import multiprocessing
from tqdm import tqdm
import time
import os
from datetime import datetime

# This script is a standalone utility to filter association rules 
# using the rescued intermediate data from a long-running Main_Association_Rule_ex.py process.

# --- Helper Functions (copied from the main script for compatibility) ---

_worker_data_filter = ()

def calculate_support_for_itemset(itemset, df):
    """
    Calculates the support for a single itemset in a given dataframe.
    Note: min_support is handled outside this function in this script.
    """
    if not itemset or df.empty:
        return 0
    mask = pd.Series([True] * len(df), index=df.index)
    for key, value in itemset.items():
        if key in df.columns:
            mask &= (df[key] == value)
        else:
            return 0
    support = mask.sum() / len(df)
    return support

def _init_worker_filter(normal_data):
    """
    Initializer for each worker process in the pool.
    """
    global _worker_data_filter
    _worker_data_filter = normal_data

def _is_rule_valid_for_filtering(rule_with_support):
    """
    The actual task for each worker process. Checks a rule against global normal data.
    """
    rule, min_support = rule_with_support
    global _worker_data_filter
    normal_data = _worker_data_filter
    
    is_frequent_in_normal = calculate_support_for_itemset(rule, normal_data) >= min_support
    
    if not is_frequent_in_normal:
        return rule
    return None

def main():
    """
    Main function to load rescued data and perform fast, parallel filtering.
    """
    start_time = time.time()
    print("--- Starting Standalone Rule Filtering Script ---")

    # --- 1. Load the rescued data ---
    try:
        print("Loading rescued anomaly rules from 'rescue_anomal_rules.pkl'...")
        with open('rescue_anomal_rules.pkl', 'rb') as f:
            association_list_anomal = pickle.load(f)
        print(f"  > Loaded {len(association_list_anomal):,} rules.")

        print("Loading rescued normal data from 'rescue_normal_data.pkl'...")
        with open('rescue_normal_data.pkl', 'rb') as f:
            nomal_grouped_data = pickle.load(f)
        print(f"  > Loaded normal data with shape: {nomal_grouped_data.shape}")
        
        print("Loading rescued min_support from 'rescue_minsupport.pkl'...")
        with open('rescue_minsupport.pkl', 'rb') as f:
            min_support = pickle.load(f)
        print(f"  > Loaded min_support value: {min_support}")

    except FileNotFoundError as e:
        print(f"\n[ERROR] Rescue file not found: {e}")
        print("Please ensure you have run the main script and interrupted it to generate the .pkl files.")
        return
    except Exception as e:
        print(f"\n[ERROR] Failed to load rescue files: {e}")
        return

    # --- 2. Perform efficient parallel filtering ---
    num_filter_processes = 12  # As specified in the main script for this dataset
    print(f"\nFiltering {len(association_list_anomal):,} rules using {num_filter_processes} processes...")
    
    signatures = []
    if not nomal_grouped_data.empty and association_list_anomal:
        tasks = [(rule, min_support) for rule in association_list_anomal]
        
        with multiprocessing.Pool(
            processes=num_filter_processes,
            initializer=_init_worker_filter,
            initargs=(nomal_grouped_data,) # Pass tuple with one element
        ) as pool:
            chunk_size = max(1, len(tasks) // (num_filter_processes * 10))
            results_iterator = pool.imap_unordered(_is_rule_valid_for_filtering, tasks, chunksize=chunk_size)

            pbar = tqdm(results_iterator, total=len(tasks), desc="[Standalone Filter] Progress")
            
            for result_rule in pbar:
                if result_rule is not None:
                    signatures.append(result_rule)
    else:
        signatures = association_list_anomal

    print(f"\n[SUCCESS] Filtering complete. Found {len(signatures):,} valid signatures.")

    # --- 3. Save the final signatures to a CSV file ---
    output_dir = "rescued_results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"filtered_signatures_{timestamp}.csv")
    
    try:
        # Convert list of dicts to DataFrame for easy saving
        signatures_df = pd.DataFrame(signatures)
        signatures_df.to_csv(output_path, index=False)
        print(f"Final signatures saved to: {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save final signatures to CSV: {e}")

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    # Add this to ensure multiprocessing works correctly on all platforms (especially Windows)
    multiprocessing.freeze_support()
    main()
