import argparse
import numpy as np
import pandas as pd
import time
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, classification_report
import joblib
import multiprocessing

# === Project Root Path Correction ===
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    if '..' not in sys.path:
        sys.path.insert(0, '..')

# Now, import project modules
from Modules.Clustering_Algorithm_Autotune import choose_clustering_algorithm_for_cache
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify
# MODIFIED: Import the new, specific function
from Dataset_Choose_Rule.time_save import time_save_csv_VL_eval

# --- Configuration ---
CACHE_DIR = "../Dataset/cache/Data_Labeling/"

def get_cache_path(file_type, file_number, clustering_algorithm):
    """Generates a consistent path for cache files."""
    filename = f"{file_type}_{file_number}_{clustering_algorithm}_clustering_cache.pkl"
    return os.path.join(CACHE_DIR, filename)

def generate_cache(args, timing_info):
    """
    Runs the full data labeling process once and saves the expensive-to-compute results.
    """
    print("--- Mode: Generate Cache ---")
    print("Running the full data processing and clustering pipeline to generate cache...")

    try:
        from Dataset_Choose_Rule.choose_amount_dataset import file_path_line_nonnumber, file_cut
        from definition.Anomal_Judgment import anomal_judgment_nonlabel, anomal_judgment_label
        from utils.time_transfer import time_scalar_transfer
        from Modules.Heterogeneous_module import choose_heterogeneous_method
        from Heterogeneous_Method.separate_group_mapping import map_intervals_to_groups
        from Modules.PCA import pca_func
        from utils.minmaxscaler import apply_minmax_scaling_and_save_scalers

        start_time = time.time()

        # Step 1: Data Loading & Preprocessing
        start_step_time = time.time()
        print("Executing steps 1-4: Data Loading, Preprocessing, and PCA...")
        file_path, _ = file_path_line_nonnumber(args.file_type, args.file_number)

        # --- MODIFIED: Load full data first, create label, then stratify sample ---
        # This approach avoids modifying the shared file_cut function.
        print("Loading full dataset to perform stratified sampling...")
        # Use a temporary full dataframe to create labels before sampling
        try:
            full_df = pd.read_csv(file_path) # Consider adding dtype optimization if memory is an issue
        except Exception as e:
            print(f"[ERROR] Failed to load the full dataset from {file_path}. Error: {e}")
            return
            
        # 1. Create the label column on the full dataframe
        full_df.columns = full_df.columns.str.strip()
        if args.file_type in ['MiraiBotnet', 'NSL-KDD', 'NSL_KDD']:
            full_df['label'], _ = anomal_judgment_nonlabel(args.file_type, full_df)
        elif args.file_type == 'netML':
            full_df['label'] = full_df['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
        elif args.file_type == 'DARPA98':
            full_df['label'] = full_df['Class'].apply(lambda x: 0 if str(x).strip() == '-' else 1)
        elif args.file_type in ['CICIDS2017', 'CICIDS']:
            if 'Label' in full_df.columns:
                full_df['label'] = full_df['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
            else:
                full_df['label'] = 0 # Default
        elif args.file_type in ['CICModbus23', 'CICModbus']:
            normal_labels = ['Normal', 'Baseline Replay: In position']
            full_df['label'] = full_df['Attack'].apply(lambda x: 0 if str(x).strip() in normal_labels else 1)
        elif args.file_type in ['IoTID20', 'IoTID']:
            full_df['label'] = full_df['Label'].apply(lambda x: 0 if str(x).strip() == 'Normal' else 1)
        elif args.file_type == 'Kitsune':
            full_df['label'] = full_df['Label']
        elif args.file_type in ['CICIoT', 'CICIoT2023']:
            full_df['label'] = full_df['attack_flag']
        else:
            full_df['label'] = anomal_judgment_label(full_df)

        # 2. Perform stratified sampling on the labeled dataframe
        amount_to_sample = 10000
        print(f"Performing stratified sampling to get {amount_to_sample} rows...")
        
        if full_df['label'].nunique() < 2:
            print("WARN: Label column has less than 2 unique values. Falling back to random sampling.")
            data = full_df.sample(n=amount_to_sample, random_state=42)
        else:
            from sklearn.model_selection import train_test_split
            # We use train_test_split to get a representative sample
            _, data = train_test_split(
                full_df,
                test_size=amount_to_sample,
                stratify=full_df['label'],
                random_state=42
            )
        
        del full_df # Free up memory
        print("Sampling complete.")

        data = time_scalar_transfer(data, args.file_type)
        embedded_df, _, category_mapping, data_list = choose_heterogeneous_method(data, args.file_type, "Interval_inverse", 'N')
        group_mapped_df, _ = map_intervals_to_groups(embedded_df, category_mapping, data_list, 'N')
        
        #'''
        # --- FIX: Handle NaN values after mapping ---
        # The mapping process can create NaNs if a value doesn't fit any interval.
        # These must be handled before scaling and PCA.
        if group_mapped_df.isnull().values.any():
            print(f"[Warning] NaN values found after group mapping (shape: {group_mapped_df.shape}). Filling with 0.")
            total_nans = group_mapped_df.isnull().sum().sum()
            print(f"Total NaN values: {total_nans}")
            group_mapped_df.fillna(0, inplace=True)
        # --- END FIX ---
        #'''

        X_scaled, _ = apply_minmax_scaling_and_save_scalers(group_mapped_df, args.file_type, args.file_number, "Interval_inverse")
        
        pca_want = 'N' if args.file_type in ['DARPA98', 'DARPA', 'CICModbus23', 'CICModbus'] else 'Y'
        X_reduced = pca_func(X_scaled) if pca_want == 'Y' else X_scaled.to_numpy()
        original_labels = data['label'].to_numpy()
        timing_info['1_load_and_preprocess'] = time.time() - start_step_time
        print(f"Preprocessing finished. Shape of data for clustering: {X_reduced.shape}")

        # Step 2: Run clustering
        start_step_time = time.time()
        print(f"Executing step 5: Clustering with {args.clustering} to get raw labels...")
        
        # This variable is needed by the function signature, but for cache generation
        # where we don't identify normal clusters yet, it can be None.
        known_normal_samples_pca = None

        # Determine the number of processes to use for the algorithm itself
        algo_processes_to_use = args.num_processes_for_clustering

        # Prepare kwargs for the clustering algorithm to conditionally add parameters.
        clustering_kwargs = {
            'data': data,
            'X_reduced_features': X_reduced,  # Corrected keyword argument
            'original_labels_aligned': original_labels,
            'clustering_algorithm_choice': args.clustering,
            'max_clusters': args.max_clusters,
            'global_known_normal_samples_pca': known_normal_samples_pca,
            'threshold_value': 0.3, # This is a placeholder, not used for raw label generation
            'num_processes_for_algo': algo_processes_to_use # CORRECTED: Use all available cores
        }

        # Hardcode eps for DARPA98 with DBSCAN
        if args.file_type in ['DARPA98', 'DARPA', 'CICIDS2017', 'CICIDS'] and args.clustering == 'DBSCAN':
            # MODIFIED: Changed from 0.5 to a smaller value to encourage more clusters.
            hardcoded_eps = 0.1 
            print(f"[INFO] Applying hardcoded eps={hardcoded_eps} for DARPA98 with DBSCAN.")
            clustering_kwargs['dbscan_eps'] = hardcoded_eps

        # Hardcode k for specific combinations where Elbow method is unreliable
        hardcoded_k_combinations = {
            ('NSL-KDD', 'Kmeans'): 150,
            ('NSL-KDD', 'GMM'): 150,
            ('Kitsune', 'Kmeans'): 150,
            ('Kitsune', 'GMM'): 150,
            ('Kitsune', 'SGMM'): 150,
            ('CICIDS2017', 'Kmeans'): 150,
            ('CICIDS2017', 'GMM'): 150,
            ('CICIDS2017', 'SGMM'): 150,
            ('CICIoT2023', 'Kmeans'): 150,
            ('CICIoT2023', 'GMM'): 150,
            ('CICIoT2023', 'SGMM'): 150,
            ('CICModbus23', 'Kmeans'): 150,
            ('CICModbus23', 'GMM'): 150,
            ('CICModbus23', 'SGMM'): 150,
        }
        current_combination = (args.file_type, args.clustering)
        if current_combination in hardcoded_k_combinations:
            hardcoded_k = hardcoded_k_combinations[current_combination]
            print(f"[INFO] Applying hardcoded k={hardcoded_k} for {args.file_type} with {args.clustering}.")
            clustering_kwargs['k'] = hardcoded_k

        # Generate and cache the clustering results
        try:
            # Pass the arguments via dictionary unpacking
            # CORRECTED: Unpack the tuple (result_dict, gmm_type) returned by the function.
            result_dict, _ = choose_clustering_algorithm_for_cache(**clustering_kwargs)
            
            raw_labels = result_dict.get('raw_cluster_labels')
            num_clusters = result_dict.get('num_clusters')

            if raw_labels is None or num_clusters is None:
                print("[ERROR] The selected clustering algorithm did not return 'raw_cluster_labels' or 'num_clusters'.")
                #print("[ERROR] Please refactor the corresponding clustering script (e.g., clustering_GMM.py) like we did for K-Means.")
                return

            # What to cache:
            cache_data = {
                'X_for_identification': X_reduced,
                'raw_cluster_labels': raw_labels,
                'original_labels': original_labels,
                'num_clusters': num_clusters,
                'description': f'Cache generated for {args.file_type} with {args.clustering} on {time.ctime()}'
            }
            
            os.makedirs(CACHE_DIR, exist_ok=True)
            cache_path = get_cache_path(args.file_type, args.file_number, args.clustering)
            
            print(f"\nSaving cache data to: {cache_path}...")
            joblib.dump(cache_data, cache_path)
            
            total_time = time.time() - start_time
            print(f"Cache generation complete. Total time: {total_time:.2f} seconds.")
            timing_info['2_clustering_execution'] = time.time() - start_step_time

        except Exception as e:
            print(f"[ERROR] An unexpected error occurred during cache generation: {e}")
            timing_info['2_clustering_execution'] = time.time() - start_step_time # Ensure timing is recorded even on error

    except ImportError as e:
        print(f"[ERROR] Failed to import a required module: {e}")
        print("Please ensure all project dependencies are in the Python path.")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during cache generation: {e}")
        timing_info['2_clustering_execution'] = time.time() - start_step_time # Ensure timing is recorded even on error

def _evaluate_single_threshold(args_tuple):
    """Helper function to evaluate a single threshold in a separate process."""
    threshold, X, original_labels, raw_labels, num_clusters = args_tuple
    
    try:
        # --- FIX 1: Disable nested parallelism ---
        # Pass num_processes_for_algo=1 to prevent the child process from creating its own pool.
        final_labels = clustering_nomal_identify(
            data_features_for_clustering=X,
            original_labels_aligned=original_labels,
            clusters_assigned=raw_labels,
            num_total_clusters=num_clusters,
            threshold_value=threshold,
            num_processes_for_algo=1  # Force single-threaded execution inside the worker
        )

        # --- FIX 2: Filter out noise and invalid labels before metric calculation ---
        # DBSCAN can produce -1 for noise, and original_labels can have NaNs.
        # Create a mask for valid data points to be included in metrics.
        valid_predicted_indices = (final_labels != -1)
        valid_original_indices = np.isfinite(original_labels)
        
        # Combine masks: a point is valid only if both its predicted and original labels are valid.
        valid_indices = valid_predicted_indices & valid_original_indices

        # Apply the mask to get clean arrays for calculation
        original_labels_filtered = original_labels[valid_indices]
        final_labels_filtered = final_labels[valid_indices]

        if len(original_labels_filtered) == 0:
            # If no valid labels are left after filtering, return zero metrics
            return threshold, 0, 0, 0, 0, 0

        # Calculate metrics using the filtered (clean) data
        accuracy = accuracy_score(original_labels_filtered, final_labels_filtered)
        jaccard = jaccard_score(original_labels_filtered, final_labels_filtered, average='binary', zero_division=0)
        
        # Use a dictionary for detailed classification_report
        report_dict = classification_report(original_labels_filtered, final_labels_filtered, output_dict=True, zero_division=0)
        precision = report_dict.get('1', {}).get('precision', 0)
        recall = report_dict.get('1', {}).get('recall', 0)
        f1 = report_dict.get('1', {}).get('f1-score', 0)

        return threshold, accuracy, precision, recall, f1, jaccard

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during threshold evaluation: {e}")
        return threshold, 0, 0, 0, 0, 0

def evaluate_thresholds(args, timing_info):
    """
    Loads pre-computed clustering results from cache and evaluates multiple thresholds.
    """
    print("--- Mode: Evaluate Thresholds (in Parallel) ---")
    
    # Step 1: Load Cache
    start_step_time = time.time()
    cache_path = get_cache_path(args.file_type, args.file_number, args.clustering)

    if not os.path.exists(cache_path):
        print(f"Error: Cache file not found at '{cache_path}'")
        print(f"Please run with the --generate_cache flag first for this configuration.")
        return

    print(f"Loading cached data from: {cache_path}")
    cached_data = joblib.load(cache_path)
    
    # RESTORED: Unpack the essential data from the cache. This is a critical step.
    X_for_identification = cached_data['X_for_identification']
    raw_cluster_labels = cached_data['raw_cluster_labels']
    original_labels = cached_data['original_labels']
    num_clusters = cached_data['num_clusters']

    timing_info['1_load_cache'] = time.time() - start_step_time
    
    # Step 2: Parallel Threshold Evaluation
    start_step_time = time.time()
    results = []
    print(f"\nEvaluating {len(args.thresholds)} thresholds using up to {args.num_processes_for_clustering} processes...")
    
    tasks = [(thresh, X_for_identification, original_labels, raw_cluster_labels, num_clusters) for thresh in args.thresholds]

    with multiprocessing.Pool(processes=args.num_processes_for_clustering) as pool:
        results = pool.map(_evaluate_single_threshold, tasks)
    timing_info['2_parallel_evaluation'] = time.time() - start_step_time

    # Step 3: Save Results
    start_step_time = time.time()
    results_df = pd.DataFrame(results, columns=['threshold', 'accuracy', 'precision', 'recall', 'f1_score', 'jaccard'])
    
    # --- NEW: Add the number of clusters to the results DataFrame ---
    # This value is constant for the entire evaluation run for a given cache.
    results_df['num_clusters'] = num_clusters
    
    # Reorder columns to have num_clusters appear after threshold
    cols = ['threshold', 'num_clusters', 'accuracy', 'precision', 'recall', 'f1_score', 'jaccard']
    results_df = results_df[cols]

    print("\n\n--- Threshold Evaluation Results ---")
    print(results_df.to_string(index=False))
    
    # --- New Save Logic ---
    output_dir = f"../Dataset/threshold_evaluation/{args.file_type}/"
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{args.clustering}_thresholds.csv"
    output_path = os.path.join(output_dir, filename)
    
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    timing_info['3_save_results'] = time.time() - start_step_time


def main():
    parser = argparse.ArgumentParser(
        description="Generate cached clustering data or evaluate multiple thresholds on cached data.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # --- Arguments for both modes ---
    parser.add_argument('--file_type', type=str, default="MiraiBotnet", help="Type of the dataset file.")
    parser.add_argument('--file_number', type=int, default=1, help="Number of the dataset file.")
    parser.add_argument('--clustering', type=str, default="Kmeans", help="Clustering algorithm used.")
    parser.add_argument('--max_algo_processes', type=int, default=0,
                        help="Max processes for parallel tasks. 0 for all available cores.")
    parser.add_argument('--max_clusters', type=int, default=1000, 
                        help="Maximum number of clusters for algorithms that require it (e.g., K-Means).")

    # --- Mode Selection ---
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--generate_cache', 
        action='store_true', 
        help="Run the full pipeline once and save results to a cache file."
    )
    group.add_argument(
        '--evaluate_thresholds', 
        action='store_true', 
        help="Load a cache file and evaluate multiple thresholds in parallel."
    )

    # --- Arguments for evaluation mode ---
    parser.add_argument(
        '--thresholds', 
        type=float, 
        nargs='*', # Use '*' to allow zero or more arguments
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="A list of threshold values to evaluate. If not provided, a default range (0.1-0.9) will be used."
    )

    args = parser.parse_args()

    # --- Start of Timing Logic ---
    total_start_time = time.time()
    timing_info = {}

    # Determine the number of processes to use, following the logic from Data_Labeling.py
    available_cores = multiprocessing.cpu_count()
    if args.max_algo_processes > 0 and args.max_algo_processes <= available_cores:
        num_processes_for_clustering = args.max_algo_processes
    else:
        num_processes_for_clustering = available_cores
    
    print(f"[INFO] Using up to {num_processes_for_clustering} processes for parallel tasks.")
    
    # Add the determined number of processes to the args object for easy access
    args.num_processes_for_clustering = num_processes_for_clustering

    if args.generate_cache:
        generate_cache(args, timing_info)
    elif args.evaluate_thresholds:
        evaluate_thresholds(args, timing_info)

    # --- End of Timing Logic ---
    total_end_time = time.time()
    timing_info['0_total_time'] = total_end_time - total_start_time
    
    print(f"\n--- Total Execution Time: {timing_info['0_total_time']:.2f} seconds ---")
    
    # MODIFIED: Call the new, specific function to save the timing log
    time_save_csv_VL_eval(
        file_type=args.file_type,
        file_number=args.file_number,
        clustering_algorithm=args.clustering,
        timing_info=timing_info
    )


if __name__ == '__main__':
    main() 