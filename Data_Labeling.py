# A machine to test clustering algorithm for labeling data and determine the performance of each clustering algorithm.

import argparse
import numpy as np
import pandas as pd # Ensure pandas is imported as pd for X.to_numpy()
import time
import math # Added for math.ceil
import os # Added import for os.cpu_count()
import gc
import glob
from sklearn.preprocessing import MinMaxScaler
from Dataset_Choose_Rule.choose_amount_dataset import file_path_line_nonnumber, file_cut
from definition.Anomal_Judgment import anomal_judgment_nonlabel, anomal_judgment_label
from utils.time_transfer import time_scalar_transfer
from Modules.Heterogeneous_module import choose_heterogeneous_method
from Heterogeneous_Method.separate_group_mapping import map_intervals_to_groups
from Modules.PCA import pca_func
from Modules.Clustering_Algorithm_Autotune import choose_clustering_algorithm
from Modules.Clustering_Algorithm_Nonautotune import choose_clustering_algorithm_Non_optimization
from utils.cluster_adjust_mapping import cluster_mapping
from Clustering_Method.clustering_score import evaluate_clustering, evaluate_clustering_wos
from Dataset_Choose_Rule.save_csv import csv_compare_clustering, csv_compare_matrix_clustering
from Dataset_Choose_Rule.time_save import time_save_csv_VL
from utils.minmaxscaler import apply_minmax_scaling_and_save_scalers


def main():
    # argparser
    # Create an instance that can receive argument values
    parser = argparse.ArgumentParser(description='Argparser')

    # Set the argument values to be input (default value can be set)
    parser.add_argument('--file_type', type=str, default="MiraiBotnet")   # data file type
    parser.add_argument('--file_number', type=int, default=1)   # Detach files
    parser.add_argument('--train_test', type=int, default=0)    # train = 0, test = 1
    parser.add_argument('--heterogeneous', type=str, default="Interval_inverse")   # Heterogeneous(Embedding) Methods
    parser.add_argument('--clustering', type=str, default="kmeans")   # Clustering Methods
    parser.add_argument('--eval_clustering_silhouette', type=str, default="n", help="Evaluate with silhouette score (y/n)")
    parser.add_argument('--autotune', type=str, default="y", help="Enable autotuning for clustering parameters (y/n)")
    parser.add_argument('--association', type=str, default="apriori")   # Association Rule
    parser.add_argument('--max_algo_processes', type=int, default=0, help="Max processes for internal clustering algorithms. 0 for all available cores.") # New argument

    # Save the above in args
    args = parser.parse_args()

    # Output the value of the input arguments
    file_type = args.file_type
    file_number = args.file_number
    train_tset = args.train_test
    heterogeneous_method = args.heterogeneous
    clustering_algorithm = args.clustering
    eval_clustering_silhouette_flag = args.eval_clustering_silhouette.lower() == 'y'
    autotune_enabled = args.autotune.lower() == 'y'
    Association_mathod = args.association
    max_algo_processes_arg = args.max_algo_processes # New argument

    total_start_time = time.time()  # Start All Time
    timing_info = {}  # For step-by-step time recording

    # Determine the number of processes for internal algorithms
    available_cores = os.cpu_count()
    if max_algo_processes_arg > 0 and max_algo_processes_arg <= available_cores:
        num_processes_for_clustering_algo = max_algo_processes_arg
    else:
        num_processes_for_clustering_algo = available_cores
    
    print(f"[INFO] Using up to {num_processes_for_clustering_algo} processes for internal clustering algorithms.")

    # Define candidate threshold values for CNI
    threshold_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # Restored to include 0.1 and 0.2
    # optimal_cni_threshold will be determined after processing all chunks

    # 0. Create Global Reference Normal Samples PCA (for CNI function)
    # This is done once based on the full dataset if possible.
    start_global_ref = time.time()
    print("\nStep 0: Creating Global Reference Normal Samples PCA...")
    global_known_normal_samples_pca_for_cni = None
    try:
        # --- MOVED file_path definition here for Step 0 ---
        file_path_for_global_ref, _ = file_path_line_nonnumber(file_type, file_number) # Use a distinct variable if file_number is also used/modified later
        # If file_number from args is only for main processing, using it here might be okay.
        # For clarity, used file_path_for_global_ref.

        # Load FULL data for reference normal selection, regardless of main cut_type
        # This assumes file_path_line_nonnumber gives a path that can be read fully
        # For CICIDS2017, select_csv_file() might need adjustment or a specific full file path
        # For simplicity, we assume file_path is suitable for a full read here.
        # If file_type specific full paths are needed, this logic needs more conditions.
        
        # Temporarily load full data to get global normal distribution
        print("[DEBUG GlobalRef] Loading full data for reference normal selection...")
        # NOTE: file_cut will use its own dtypes. This might be different from the main data load if cut_type varies.
        #       It might be better to have a dedicated full data loader if dtypes or post-processing needs to be identical.
        full_data_for_ref = file_cut(file_type, file_path_for_global_ref, 'all') # Force 'all' to get all data
        full_data_for_ref.columns = full_data_for_ref.columns.str.strip()
        print(f"[DEBUG GlobalRef] Full data for ref loaded. Shape: {full_data_for_ref.shape}")

        # Apply same basic labeling as in Step 2 to this full data
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
        elif file_type == 'Kitsune':
            data['label'] = data['Label']
        elif file_type in ['CICIoT', 'CICIoT2023']:
            data['label'] = data['attack_flag']
        else:
            # This is a fallback, ensure your file_type is covered above for specific handling
            logger.warning(f"WARNING: Using generic anomal_judgment_label for {file_type}.")
            data['label'] = anomal_judgment_label(data)
        print(f"[DEBUG GlobalRef] Full data labeled. Label counts: {full_data_for_ref['label'].value_counts(dropna=False)}")

        # Apply same embedding and group mapping (Steps 3, simplified for ref normal generation)
        full_data_for_ref_processed = time_scalar_transfer(full_data_for_ref.copy(), file_type) # Use copy
        # Assuming 'N' for regul for consistency in generating reference normals
        ref_embedded_df, _, ref_cat_map, ref_data_list = choose_heterogeneous_method(full_data_for_ref_processed, file_type, heterogeneous_method, 'N')
        ref_group_mapped_df, _ = map_intervals_to_groups(ref_embedded_df, ref_cat_map, ref_data_list, 'N')
        print(f"[DEBUG GlobalRef] Full data group mapped. Shape: {ref_group_mapped_df.shape}")
        
        # Apply MinMax scaling (like Step 3.5)
        # Note: Scalers from this step are not saved globally, only used for this ref normal generation
        ref_scalers_temp = {}
        ref_scaled_features_list = []
        if not ref_group_mapped_df.empty:
            for col_ref in ref_group_mapped_df.columns:
                scaler_ref = MinMaxScaler()
                ref_feature_vals = ref_group_mapped_df[col_ref].values.reshape(-1,1)
                ref_scaled_vals = scaler_ref.fit_transform(ref_feature_vals)
                ref_scaled_features_list.append(pd.Series(ref_scaled_vals.flatten(), name=col_ref, index=ref_group_mapped_df.index))
            ref_X_scaled = pd.concat(ref_scaled_features_list, axis=1)
        else:
            ref_X_scaled = pd.DataFrame(index=ref_group_mapped_df.index)
        print(f"[DEBUG GlobalRef] Full data scaled. Shape: {ref_X_scaled.shape}")

        # Apply PCA (like Step 4)
        # Assuming pca_want is 'Y' for generating global normals for consistency, unless specific file types always avoid PCA
        ref_pca_want = 'N' if file_type in ['DARPA98', 'DARPA', 'CICModbus23', 'CICModbus'] else 'Y'
        if ref_pca_want == 'Y':
            # Ensure pca_func can handle DataFrame input and returns DataFrame/NumPy array
            # If X_reduced is dataframe, it will have columns. If numpy, it won't.
            print("[DEBUG GlobalRef] Applying PCA to scaled full data for reference...")
            ref_X_pca = pca_func(ref_X_scaled) 
        else:
            print("[DEBUG GlobalRef] Skipping PCA for reference normal generation based on file_type.")
            ref_X_pca = ref_X_scaled.to_numpy() if hasattr(ref_X_scaled, 'to_numpy') else ref_X_scaled # Ensure NumPy array
        print(f"[DEBUG GlobalRef] Full data PCA applied. Shape: {ref_X_pca.shape}")

        # Now, from ref_X_pca and full_data_for_ref['label'], create the global reference
        all_normal_samples_pca_ref = ref_X_pca[full_data_for_ref['label'] == 0]
        num_all_normal_ref = all_normal_samples_pca_ref.shape[0]
        print(f"[DEBUG GlobalRef] Total normal samples in full data (PCA space): {num_all_normal_ref}")

        if num_all_normal_ref > 1:
            # sample_size_ref = int(num_all_normal_ref * 0.95)
            sample_size_ref = int(num_all_normal_ref * 0.90)
            if sample_size_ref == 0 and num_all_normal_ref > 0: sample_size_ref = 1 # Ensure at least 1 if possible
            # Handle cases where sample_size_ref might be larger than population if num_all_normal_ref is small
            if sample_size_ref > num_all_normal_ref : sample_size_ref = num_all_normal_ref 
            
            if sample_size_ref > 0 : # Proceed only if sample_size_ref is valid
                random_indices_ref = np.random.choice(num_all_normal_ref, size=sample_size_ref, replace=False)
                global_known_normal_samples_pca_for_cni = all_normal_samples_pca_ref[random_indices_ref]
                # print(f"[DEBUG GlobalRef] Global reference normal samples (95% of all normals in full data, PCA space) created. Shape: {global_known_normal_samples_pca_for_cni.shape}")
                print(f"[DEBUG GlobalRef] Global reference normal samples (90% of all normals in full data, PCA space) created. Shape: {global_known_normal_samples_pca_for_cni.shape}")
            else:
                print("[WARN GlobalRef] Sample size for global reference normals is 0. No global reference created.")
                global_known_normal_samples_pca_for_cni = np.array([]) # Empty array

        elif num_all_normal_ref == 1:
            global_known_normal_samples_pca_for_cni = all_normal_samples_pca_ref
            print("[DEBUG GlobalRef] Global reference normal samples (1 sample from full data, PCA space) created.")
        else:
            print("[WARN GlobalRef] No normal samples found in the full dataset to create global reference.")
            global_known_normal_samples_pca_for_cni = np.array([]) # Empty array with potentially 0 columns if ref_X_pca was empty
            if ref_X_pca.ndim == 2 and ref_X_pca.shape[1] > 0: # Try to give it correct num_cols if possible
                 global_known_normal_samples_pca_for_cni = np.empty((0, ref_X_pca.shape[1]))

        del full_data_for_ref, full_data_for_ref_processed, ref_embedded_df, ref_group_mapped_df, ref_X_scaled, ref_X_pca, all_normal_samples_pca_ref # Free memory
        print("[DEBUG GlobalRef] Freed memory from temporary full data load.")

    except FileNotFoundError:
        print(f"[WARN GlobalRef] Full data file not found for {file_type} at expected path {file_path_for_global_ref}. Cannot create global normal reference.")
    except KeyError as ke:
        print(f"[WARN GlobalRef] KeyError during global normal reference creation (e.g., 'label' or other column missing): {ke}. Cannot create global normal reference.")
    except ValueError as ve:
        print(f"[WARN GlobalRef] ValueError during global normal reference creation: {ve}. Cannot create global normal reference.")
    except Exception as e:
        print(f"[ERROR GlobalRef] Failed to create global reference normal samples: {e}. Proceeding without it.")
        # Ensure it's None or an empty array so later logic doesn't break
        if global_known_normal_samples_pca_for_cni is not None and not isinstance(global_known_normal_samples_pca_for_cni, np.ndarray):
             global_known_normal_samples_pca_for_cni = None # Fallback
        elif isinstance(global_known_normal_samples_pca_for_cni, np.ndarray) and global_known_normal_samples_pca_for_cni.size == 0 and global_known_normal_samples_pca_for_cni.ndim == 1: # e.g. np.array([])
            # Try to give it 2 dims if it's an empty 1D array from np.array([]) init
            # This depends on whether ref_X_pca was successfully created to get num_cols
            # This part is tricky; ideally, it should be initialized with correct num_cols if possible
            pass # Keep as is, CNI function has robust empty checks

    timing_info['0_global_ref_creation'] = time.time() - start_global_ref
    print(f"Step 0 finished. Time: {timing_info['0_global_ref_creation']:.2f}s. Global ref shape: {global_known_normal_samples_pca_for_cni.shape if global_known_normal_samples_pca_for_cni is not None else 'None'}")


    # 1. Load data from csv
    start = time.time()
    # Define file_path for main data loading using original file_number from args
    file_path, file_number = file_path_line_nonnumber(file_type, args.file_number) 
    # cut_type = str(input("Enter the data cut type: "))
    if file_type in ['DARPA98', 'DARPA', 'NSL-KDD', 'NSL_KDD', 'CICModbus23', 'CICModbus', 'MitM', 'Kitsune', 'ARP']:
        cut_type = 'random'
    else:
        cut_type = 'all'
    data = file_cut(file_type, file_path, cut_type)

    # Clean column names by stripping leading/trailing whitespace
    data.columns = data.columns.str.strip()

    timing_info['1_load_data'] = time.time() - start


    # 2. Check data 'label'
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
    elif file_type == 'Kitsune':
        data['label'] = data['Label']
    else:
        # This is a fallback, ensure your file_type is covered above for specific handling
        logger.warning(f"WARNING: Using generic anomal_judgment_label for {file_type}.")
        data['label'] = anomal_judgment_label(data)

    timing_info['2_label_check'] = time.time() - start


    # 3. Feature-specific embedding and preprocessing
    start = time.time()

    data = time_scalar_transfer(data, file_type)

    # regul = str(input("\nDo you want to Regulation? (Y/n): ")) # Whether to normalize or not
    regul = 'N'

    embedded_dataframe, feature_list, category_mapping, data_list = choose_heterogeneous_method(data, file_type, heterogeneous_method, regul)
    print("embedded_dataframe: ", embedded_dataframe)

    group_mapped_df, mapped_info_df = map_intervals_to_groups(embedded_dataframe, category_mapping, data_list, regul)
    print("mapped group: ", group_mapped_df)
    print("mapped_info: ", mapped_info_df)
    
    # --- FIX: Handle NaN values after mapping ---
    # The mapping process can create NaNs if a value doesn't fit any interval.
    # These must be handled before scaling and PCA.
    if group_mapped_df.isnull().values.any():
        print(f"[Warning] NaN values found after group mapping (shape: {group_mapped_df.shape}). Filling with 0.")
        total_nans = group_mapped_df.isnull().sum().sum()
        print(f"Total NaN values: {total_nans}")
        group_mapped_df.fillna(0, inplace=True)
    # --- END FIX ---

    timing_info['3_embedding'] = time.time() - start


    # 3.5 Apply MinMaxScaler using the utility function
    X_scaled_for_pca, saved_scaler_path = apply_minmax_scaling_and_save_scalers(
        group_mapped_df,
        file_type,
        file_number,
        heterogeneous_method
        # base_output_dir can be specified if different from "results"
    )
    # Optionally, you can store saved_scaler_path if needed later, though it's printed by the function.


    # 4. Numpy(hstack) processing and PCA
    print("\nStep 4: PCA for main data processing...")
    start_pca_main = time.time() # PCA Timer Start
    X = X_scaled_for_pca # Use scaled data for PCA
    columns_data = list(data.columns)
    columns_X = list(X.columns)
    diff_columns = list(set(columns_data) - set(columns_X))
    print("data-X col: ", diff_columns)


    if file_type in ['DARPA98', 'DARPA', 'CICModbus23', 'CICModbus']:
        pca_want = 'N'
    else:
        pca_want = 'Y'

    # pca_want = str(input("\nDo you want to do PCA? (Y/n): "))
    if pca_want in ['Y', 'y']:
        if clustering_algorithm in ['CANNwKNN', 'CANN']:
            print("CANN is a classification, which means you need to use the full data.")
            X_reduced = X
        else:
            print("[PCA Main] Applying PCA...")
            X_reduced = pca_func(X)
            print("[PCA Main] PCA finished.")
    else:
        print("[PCA Main] Skipping PCA for main data based on user/file_type setting.")
        X_reduced = X

    print(f"\n[DEBUG Data_Labeling.py] X_reduced (data used for clustering) shape: {X_reduced.shape}")
    if hasattr(X_reduced, 'columns'): # X_reduced is a DataFrame
        print(f"[DEBUG Data_Labeling.py] X_reduced columns: {list(X_reduced.columns)}")
    else: # X_reduced is a NumPy array
        print(f"[DEBUG Data_Labeling.py] X_reduced is a NumPy array (no direct column names). First 5 cols of first row: {X_reduced[0, :5] if X_reduced.shape[0] > 0 else 'empty'}")
    
    # Note: Information about X (group_mapped_df, before PCA) is also good to output
    if hasattr(X, 'columns'):
        print(f"[DEBUG Data_Labeling.py] X (pre PCA, group_mapped_df) shape: {X.shape}")
        print(f"[DEBUG Data_Labeling.py] X (pre PCA, group_mapped_df) columns: {list(X.columns)}")

    end_pca_main = time.time() # PCA Timer End
    pca_duration_main = end_pca_main - start_pca_main
    print(f"[PCA Main] PCA processing (or skipping) for main data took: {pca_duration_main:.2f}s")

    # Create original labels aligned with X_reduced
    # Assumption: Rows in 'data' DataFrame correspond to rows in 'X' (group_mapped_df),
    # and subsequently to rows in 'X_reduced' if X is a DataFrame OR if pca_func preserves row order from X (if X is NumPy).
    
    if 'label' not in data.columns:
        raise ValueError("'label' column is missing from 'data' DataFrame. Ensure labeling step (Step 2) is correct.")
    
    # Check if X (group_mapped_df) has an index that can be used to align with 'data'
    # If X was derived from 'data' and row order is preserved, direct use of data['label'] is fine.
    # If X involved row reordering or filtering inconsistent with data, a more robust alignment (e.g., using original indices) would be needed.
    # For now, we assume direct correspondence in row order and length.
    if len(data) != X_reduced.shape[0]:
        # This case should ideally not happen if data processing keeps row counts consistent
        raise ValueError(f"Row count mismatch: 'data' ({len(data)}) vs 'X_reduced' ({X_reduced.shape[0]}). Cannot reliably align labels.")
    
    original_labels_for_X_reduced = data['label'].to_numpy()
    print(f"[DEBUG Data_Labeling.py] 'original_labels_for_X_reduced' created - Shape: {original_labels_for_X_reduced.shape}, Unique values: {np.unique(original_labels_for_X_reduced, return_counts=True)}")

    timing_info['4_pca_time'] = time.time() - start_pca_main # PCA Timer End
    print(f"Step 4 finished. PCA Time: {timing_info['4_pca_time']:.2f}s. X_pca shape: {X_reduced.shape if pca_want == 'Y' else 'PCA_SKIPPED'}")

    # Prepare data for clustering (either PCA output or scaled data if PCA was skipped)
    if pca_want == 'Y':
        data_for_clustering = X_reduced
    else:
        # Ensure X (scaled data) is numpy before chunking, if it's a DataFrame
        data_for_clustering = X.to_numpy() if hasattr(X, 'to_numpy') else X 
    
    original_labels_for_chunking = data['label'].to_numpy() # Ensure original labels are numpy array

    # 5. Clustering Algorithm Application (with Chunking)
    print("\nStep 5: Clustering with {clustering_algorithm}...")
    start = time.time()

    chunk_size = 5000
    num_samples = data_for_clustering.shape[0]
    num_chunks = math.ceil(num_samples / chunk_size)
    print(f"Total samples: {num_samples}, Chunk size: {chunk_size}, Number of chunks: {num_chunks}")

    all_chunk_cluster_labels = []
    all_chunk_best_params = []
    all_chunk_gmm_types = []

    # Store Jaccard scores for each threshold from each chunk
    # Structure: {threshold_value: [jaccard_score_chunk1, jaccard_score_chunk2, ...]}
    threshold_jaccard_scores_across_chunks = {thresh: [] for thresh in threshold_candidates}
    # Store actual cluster labels for each chunk and each threshold, to be used after optimal threshold is found
    # Structure: {(chunk_index, threshold_value): [cluster_labels_for_this_chunk_and_threshold]}
    chunk_threshold_labels_temp_storage = {}

    # --- Phase 1: Iterate through chunks and thresholds to collect Jaccard scores and temp labels ---
    print("\nPhase 1: Collecting Jaccard scores and temporary labels for each chunk and threshold...")
    for i in range(num_chunks):
        start_chunk_time = time.time()
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_samples)
        
        current_chunk_data_np = data_for_clustering[start_idx:end_idx] # This is a numpy slice from data_for_clustering (which is already np array)
        current_chunk_original_labels_np = original_labels_for_chunking[start_idx:end_idx] # This is also a numpy slice

        print(f"  Processing Chunk {i+1}/{num_chunks} (Samples {start_idx}-{end_idx-1}), Shape: {current_chunk_data_np.shape}")
        
        # --- FIX: Run clustering only once per chunk, then apply different thresholds ---
        print(f"    Chunk {i+1}, Running clustering once...")
        gmm_type_for_this_run = None 
        
        if autotune_enabled: 
            temp_chunk_clustering_result, gmm_type_for_this_run = choose_clustering_algorithm(
                data, 
                current_chunk_data_np, 
                current_chunk_original_labels_np, 
                clustering_algorithm, 
                global_known_normal_samples_pca=global_known_normal_samples_pca_for_cni,
                threshold_value=0.3,  # Use default threshold for clustering
                num_processes_for_algo=num_processes_for_clustering_algo
            )
        else: 
            temp_chunk_clustering_result = choose_clustering_algorithm_Non_optimization(
                data, 
                current_chunk_data_np, 
                current_chunk_original_labels_np, 
                clustering_algorithm, 
                global_known_normal_samples_pca=global_known_normal_samples_pca_for_cni,
                threshold_value=0.3,  # Use default threshold for clustering
                num_processes_for_algo=num_processes_for_clustering_algo
            )
        
        # Extract raw cluster labels from the clustering result
        if isinstance(temp_chunk_clustering_result, dict) and 'raw_cluster_labels' in temp_chunk_clustering_result:
            raw_cluster_labels = temp_chunk_clustering_result['raw_cluster_labels']
            print(f"    Chunk {i+1}, Clustering completed. Raw clusters: {len(np.unique(raw_cluster_labels))}")
        else:
            print(f"    Chunk {i+1}, ERROR: No raw_cluster_labels found in clustering result")
            continue
        
        # Now test different thresholds using the same raw cluster labels
        for current_threshold_in_chunk_loop in threshold_candidates:
            print(f"    Chunk {i+1}, Testing CNI threshold: {current_threshold_in_chunk_loop}")
            
            # Apply CNI with the current threshold using the same raw cluster labels
            try:
                from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify
                
                y_pred_chunk_current_thresh, jaccard_score, _ = clustering_nomal_identify(
                    data_features_for_clustering=current_chunk_data_np,
                    clusters_assigned=raw_cluster_labels,
                    original_labels_aligned=current_chunk_original_labels_np,
                    global_known_normal_samples_pca=global_known_normal_samples_pca_for_cni,
                    threshold_value=current_threshold_in_chunk_loop,
                    num_processes_for_algo=num_processes_for_clustering_algo, # Added parameter  # Force single-threaded to prevent re-clustering
                    data_for_clustering=current_chunk_data_np,
                    known_normal_idx=None  # Use global reference
                )
                
                print(f"      DEBUG: Thresh {current_threshold_in_chunk_loop} - CNI applied successfully. Shape: {y_pred_chunk_current_thresh.shape}")
                
            except Exception as e:
                print(f"      ERROR: Thresh {current_threshold_in_chunk_loop} - CNI failed: {e}")
                continue
            
            # Process the CNI results
            if y_pred_chunk_current_thresh is not None and y_pred_chunk_current_thresh.size > 0:
                # --- Start Debug Prints ---
                print(f"      DEBUG: Thresh {current_threshold_in_chunk_loop} - y_pred_chunk_current_thresh shape: {y_pred_chunk_current_thresh.shape}, dtype: {y_pred_chunk_current_thresh.dtype}")
                print(f"      DEBUG: Thresh {current_threshold_in_chunk_loop} - current_chunk_original_labels_np shape: {current_chunk_original_labels_np.shape}, dtype: {current_chunk_original_labels_np.dtype}")
                if y_pred_chunk_current_thresh.size > 0 : print(f"      DEBUG: Thresh {current_threshold_in_chunk_loop} - y_pred unique values: {np.unique(y_pred_chunk_current_thresh, return_counts=True)}")
                if current_chunk_original_labels_np.size > 0 : print(f"      DEBUG: Thresh {current_threshold_in_chunk_loop} - original_labels unique values: {np.unique(current_chunk_original_labels_np, return_counts=True)}")
                # --- End Debug Prints ---

                if y_pred_chunk_current_thresh.size == current_chunk_original_labels_np.size and y_pred_chunk_current_thresh.size > 0:
                    chunk_threshold_labels_temp_storage[(i, current_threshold_in_chunk_loop)] = y_pred_chunk_current_thresh
                    chunk_metrics = evaluate_clustering_wos(current_chunk_original_labels_np, y_pred_chunk_current_thresh)
                    
                    # --- Start Debug Print for chunk_metrics ---
                    print(f"      DEBUG: Thresh {current_threshold_in_chunk_loop} - Full chunk_metrics: {chunk_metrics}")
                    # --- End Debug Print for chunk_metrics ---

                    # Correct way to get micro jaccard score
                    micro_metrics_dict = chunk_metrics.get('average=micro', {})
                    current_jaccard_micro_chunk = micro_metrics_dict.get('jaccard', -1.0)

                    print(f"      INFO: Thresh {current_threshold_in_chunk_loop} - Calculated Jaccard (micro): {current_jaccard_micro_chunk}")
                    
                    # --- Start Debug Print if jaccard is -1.0 AND no error message was seen from jaccard_basic ---
                    if current_jaccard_micro_chunk == -1.0:
                        # This print helps confirm if jaccard_basic itself returned -1.0 vs. key missing
                        print(f"        DEBUG: Thresh {current_threshold_in_chunk_loop} - Jaccard is -1.0. Micro metrics dict was: {micro_metrics_dict}") 
                    # --- End Debug Print ---

                    if current_jaccard_micro_chunk != -1.0: # Only store valid Jaccard scores
                         threshold_jaccard_scores_across_chunks[current_threshold_in_chunk_loop].append(current_jaccard_micro_chunk)
                         print(f"        DEBUG: Thresh {current_threshold_in_chunk_loop} - Stored Jaccard. Current list for this thresh: {threshold_jaccard_scores_across_chunks[current_threshold_in_chunk_loop]}")
                    else:
                        print(f"        DEBUG: Thresh {current_threshold_in_chunk_loop} - Jaccard score is -1.0, not storing.")
                else: 
                    print(f"      WARN: Thresh {current_threshold_in_chunk_loop} - Label size mismatch or empty labels. y_pred size: {y_pred_chunk_current_thresh.size}, original_labels size: {current_chunk_original_labels_np.size}. No Jaccard calculated or stored for this run.")
            else: 
                print(f"      WARN: Thresh {current_threshold_in_chunk_loop} - 'Cluster_labeling' missing, None, or result not a dict. Result was: {temp_chunk_clustering_result}")
        
        end_chunk_time = time.time()
        print(f"  Chunk {i+1} (threshold sweep) processed in {end_chunk_time - start_chunk_time:.2f}s.")
        
        # Clean up memory after each chunk to prevent accumulation
        if i % 2 == 0:  # Every 2 chunks to avoid too frequent cleanup
            import gc
            gc.collect()
            print(f"  [MEMORY] Cleaned up memory after chunk {i+1}")

    # --- Phase 2: Determine Optimal CNI Threshold (IQR Outlier Removal + Mean) ---
    print("\nPhase 2: Determining Optimal CNI Threshold...")
    optimal_cni_threshold = 0.3 # Default if all else fails or no scores
    best_robust_average_jaccard = -1.0

    for thresh_val, jaccard_list in threshold_jaccard_scores_across_chunks.items():
        if not jaccard_list:
            print(f"  Threshold {thresh_val}: No Jaccard scores recorded.")
            continue

        scores_np = np.array(jaccard_list)
        q1 = np.percentile(scores_np, 25)
        q3 = np.percentile(scores_np, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Filter out outliers
        filtered_scores = scores_np[(scores_np >= lower_bound) & (scores_np <= upper_bound)]
        
        if filtered_scores.size > 0:
            robust_average_jaccard = np.mean(filtered_scores)
            print(f"  Threshold {thresh_val}: Robust Avg Jaccard (micro) = {robust_average_jaccard:.4f} (from {filtered_scores.size}/{scores_np.size} scores after outlier removal)")
            if robust_average_jaccard > best_robust_average_jaccard:
                best_robust_average_jaccard = robust_average_jaccard
                optimal_cni_threshold = thresh_val
    else:
            # If all scores were outliers, fall back to mean of original scores for this threshold
            # Or, could skip this threshold or assign a penalty. For now, use original mean.
            original_mean = np.mean(scores_np)
            print(f"  Threshold {thresh_val}: All scores considered outliers or no scores left. Original mean Jaccard = {original_mean:.4f} (from {scores_np.size} scores)")
            # Consider if this threshold should still be a candidate if all were outliers
            if original_mean > best_robust_average_jaccard: # Check against current best
                 best_robust_average_jaccard = original_mean # Use original mean if it's better than any robust mean found so far
                 optimal_cni_threshold = thresh_val
                 print(f"    (Using original mean as it's currently the best overall: {original_mean:.4f}) ")

    print(f"Optimal CNI Threshold selected: {optimal_cni_threshold} with best robust average Jaccard (micro): {best_robust_average_jaccard:.4f}")
    timing_info['5.1_optimal_threshold_determination_time'] = time.time() - start # Assuming 'start' was from beginning of Step 5 (chunking)
    timing_info['optimal_cni_threshold'] = optimal_cni_threshold # Store the selected threshold

    # --- Memory Cleanup Between Phases ---
    print("\n[MEMORY CLEANUP] Cleaning up memory between Phase 1 and Phase 3...")
    gc.collect()  # Force garbage collection
    
    # Clean up temporary progress files to prevent I/O issues
    try:
        progress_files = glob.glob("../Dataset_ex/progress_tracking/*_progress.csv")
        for file in progress_files:
            try:
                os.remove(file)
                print(f"[MEMORY CLEANUP] Removed progress file: {file}")
            except:
                pass
    except Exception as e:
        print(f"[MEMORY CLEANUP] Could not clean progress files: {e}")
    
    print(f"[MEMORY CLEANUP] Memory cleanup completed. Available memory before Phase 3.")
    
    # --- Phase 3: Re-process chunks with Optimal CNI Threshold to get final labels, params, and GMM types ---
    print(f"\nPhase 3: Assembling final predictions by re-processing chunks with optimal_cni_threshold = {optimal_cni_threshold}...")
    final_predict_results_list = []
    # all_chunk_best_params and all_chunk_gmm_types are cleared and repopulated here.
    all_chunk_best_params.clear()
    all_chunk_gmm_types.clear()
    start_phase3_time = time.time()

    for i in range(num_chunks):
        # print(f"  Re-processing Chunk {i+1}/{num_chunks} with optimal threshold {optimal_cni_threshold}...") # Verbose
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_samples)
        current_chunk_data_np = data_for_clustering[start_idx:end_idx]
        current_chunk_original_labels_np = original_labels_for_chunking[start_idx:end_idx]
        
        # Option 1: Re-run clustering for this chunk with the optimal_cni_threshold
        final_gmm_type_for_chunk = None
        final_params_for_chunk = {}
        labels_for_this_chunk_optimal = None

        if autotune_enabled:
            final_chunk_clustering_result, final_gmm_type_for_chunk = choose_clustering_algorithm(
                data, current_chunk_data_np, current_chunk_original_labels_np, clustering_algorithm,
                global_known_normal_samples_pca=global_known_normal_samples_pca_for_cni, threshold_value=optimal_cni_threshold,
                num_processes_for_algo=num_processes_for_clustering_algo # Added parameter
            )
        else:
            final_chunk_clustering_result = choose_clustering_algorithm_Non_optimization(
                data, current_chunk_data_np, current_chunk_original_labels_np, clustering_algorithm,
                global_known_normal_samples_pca=global_known_normal_samples_pca_for_cni, threshold_value=optimal_cni_threshold,
                num_processes_for_algo=num_processes_for_clustering_algo # Added parameter
            )
        
        if 'Cluster_labeling' in final_chunk_clustering_result and final_chunk_clustering_result['Cluster_labeling'] is not None:
            labels_for_this_chunk_optimal = final_chunk_clustering_result['Cluster_labeling']
            final_params_for_chunk = final_chunk_clustering_result.get('Best_parameter_dict', {})
        else:
            print(f"    [ERROR] Chunk {i+1} re-processing with optimal threshold failed to produce labels. Using empty array for this chunk.")
            labels_for_this_chunk_optimal = np.array([]) 
        
        # Option 2: Retrieve from chunk_threshold_labels_temp_storage (if we decide not to re-run for labels)
        # labels_for_this_chunk_optimal = chunk_threshold_labels_temp_storage.get((i, optimal_cni_threshold))
        # if labels_for_this_chunk_optimal is None:
        # print(f"    [ERROR] Could not retrieve stored labels for chunk {i+1} and optimal threshold {optimal_cni_threshold}. This should not happen if Phase 1 was complete.")
        # labels_for_this_chunk_optimal = np.array([]) # Fallback
        # NOTE: If using Option 2 for labels, we still need a strategy for Best_parameter_dict and GMM type.
        #       Re-running (Option 1) is cleaner for getting all associated info for the optimal run.

        final_predict_results_list.append(labels_for_this_chunk_optimal)
        all_chunk_best_params.append(final_params_for_chunk)
        if clustering_algorithm.upper().startswith("GMM"):
            all_chunk_gmm_types.append(final_gmm_type_for_chunk)
        else:
            all_chunk_gmm_types.append(None) 

    if final_predict_results_list and not all(arr.size == 0 for arr in final_predict_results_list): # Check if not all are empty
        # Filter out empty arrays before concatenation to avoid errors if some chunks failed
        valid_results_to_concat = [arr for arr in final_predict_results_list if arr.size > 0]
        if valid_results_to_concat:
            final_predict_results = np.concatenate(valid_results_to_concat)
            if len(final_predict_results) != num_samples:
                 print(f"[WARNING Phase 3] Length of concatenated final labels ({len(final_predict_results)}) does not match total samples ({num_samples}) after potentially excluding failed chunks.")
        else:
            print("[ERROR Phase 3] All chunks failed to produce labels in Phase 3. final_predict_results will be empty.")
            final_predict_results = np.array([])
    else:
        print("[ERROR Phase 3] No valid prediction results to concatenate after Phase 3.")
        final_predict_results = np.array([])
    
    timing_info['5.2_chunk_reprocessing_time'] = time.time() - start_phase3_time
    print(f"Phase 3 finished. Chunk re-processing time: {timing_info['5.2_chunk_reprocessing_time']:.2f}s")

    # timing_info['5_clustering_time'] should now encompass all phases of step 5
    timing_info['5_clustering_time'] = (timing_info.get('5.1_optimal_threshold_determination_time',0) + 
                                       timing_info.get('5.2_chunk_reprocessing_time',0) + 
                                       (start_phase3_time - start)) # Add time from start of phase 1 to start of phase 3

    # Concatenate results from all chunks (This line seems redundant now as final_predict_results is already populated)
    # if all_chunk_cluster_labels: # all_chunk_cluster_labels is not the primary source of final labels anymore
    #     final_predict_results = np.concatenate(all_chunk_cluster_labels)
    # Assign final_predict_results to data['cluster'] for evaluation and saving
    if len(final_predict_results) == len(data):
        data['cluster'] = final_predict_results
        print(f"[INFO] Successfully assigned final_predict_results to data['cluster']. Unique values: {np.unique(data['cluster'], return_counts=True)}")
    else:
        print(f"[WARN] Length mismatch: final_predict_results ({len(final_predict_results)}) and data ({len(data)}). Not assigning to data['cluster']. This may affect evaluation and saving.")
        # If final_predict_results is empty and data['cluster'] doesn't exist, create a dummy one to avoid KeyError later
        # but this means results are based on potentially meaningless data.
        if 'cluster' not in data.columns and len(data) > 0:
            print("[WARN] data['cluster'] does not exist. Creating a dummy 'cluster' column with all zeros.")
            data['cluster'] = np.zeros(len(data), dtype=int) # Or handle as error

    # REVISED PATCH: Populate 'adjusted_cluster' by inverting data['cluster'] (0->1, 1->0).
    # This assumes data['cluster'] correctly holds the final 0/1 CNI-adjusted labels.
    if 'cluster' in data.columns:
        # 'cluster' column exists. We will define/overwrite 'adjusted_cluster'.
        print(f"[INFO REVISED PATCH DL] 'cluster' column found. Populating 'adjusted_cluster' by inverting 'data['cluster']' (0->1, 1->0).")
        # Check if Series and has non-NaN before showing unique values
        if hasattr(data['cluster'], 'notna') and data['cluster'].notna().any(): # Check if Series and has non-NaN
            print(f"                 Input unique values in data['cluster'] (non-NaN): {pd.unique(data['cluster'][data['cluster'].notna()])}")
        elif not hasattr(data['cluster'], 'notna'): # Handle if data['cluster'] is ndarray from np.zeros
             print(f"                 Input unique values in data['cluster'] (ndarray): {np.unique(data['cluster'])}")
        else:
            print(f"                 Input data['cluster'] is all NaN or empty.")
        
        # Convert 0 to 1, 1 to 0. NaN values remain NaN.
        # Try converting data['cluster'] to Series first if it's a NumPy array
        if not isinstance(data['cluster'], pd.Series):
            current_cluster_series = pd.Series(data['cluster'], index=data.index if hasattr(data, 'index') else None)
        else:
            current_cluster_series = data['cluster']
        
        data['adjusted_cluster'] = 1 - current_cluster_series 
        
        if hasattr(data['adjusted_cluster'], 'notna') and data['adjusted_cluster'].notna().any():
            print(f"                 Output unique values in data['adjusted_cluster'] (non-NaN): {pd.unique(data['adjusted_cluster'][data['adjusted_cluster'].notna()])}")
        elif not hasattr(data['adjusted_cluster'], 'notna'):
             print(f"                 Output unique values in data['adjusted_cluster'] (ndarray): {np.unique(data['adjusted_cluster'])}")
        else:
            print(f"                 Output data['adjusted_cluster'] is all NaN or empty.")

    elif len(data) > 0 : # 'cluster' column does NOT exist, but data exists.
        print("[WARN REVISED PATCH DL] 'cluster' column is missing. Creating dummy 'cluster' (all zeros) and 'adjusted_cluster' (all ones).")
        # Ensure data['cluster'] is a Series with DataFrame index
        data['cluster'] = pd.Series(np.zeros(len(data), dtype=int), index=data.index if hasattr(data, 'index') else None)
        # If data['cluster'] is 0, 'adjusted_cluster' will be 1
        data['adjusted_cluster'] = pd.Series(np.ones(len(data), dtype=int), index=data.index if hasattr(data, 'index') else None)
    # If len(data) == 0, no action needed.


    # 6. Evaluation and Comparison (using concatenated results)
    start = time.time()

    y_true = data['label'].to_numpy() # Ground truth labels from the full dataset
    
    # Prepare X_data for silhouette score calculation - use X_reduced (PCA or scaled data)
    # Ensure it's a numpy array for evaluation functions
    if isinstance(X_reduced, pd.DataFrame):
        X_data_for_eval = X_reduced.to_numpy()
    else:
        X_data_for_eval = X_reduced # Assuming it's already a NumPy array

    metrics_original = {}
    metrics_adjusted = {}

    if 'cluster' in data.columns and len(data['cluster']) == len(y_true):
        y_pred_original = data['cluster'].to_numpy() # This will now use the final_predict_results if assigned
        if y_pred_original.size == y_true.size and y_pred_original.size > 0: # Added safety check
            if eval_clustering_silhouette_flag:
                metrics_original = evaluate_clustering(y_true, y_pred_original, X_data_for_eval)
            else:
                metrics_original = evaluate_clustering_wos(y_true, y_pred_original)
            print("Clustering Scores (Original - using data['cluster'] from final_predict_results): ", metrics_original)
        else:
            print(f"[WARN Eval] Size mismatch or empty arrays for original metrics. y_pred_original size: {y_pred_original.size}, y_true size: {y_true.size}")
    else:
        print("[WARN Eval] 'cluster' column not available or length mismatch. Skipping original metrics calculation.")

    if 'adjusted_cluster' in data.columns and len(data['adjusted_cluster']) == len(y_true):
        y_pred_adjusted = data['adjusted_cluster'].to_numpy()
        if y_pred_adjusted.size == y_true.size and y_pred_adjusted.size > 0: # Added safety check
            if eval_clustering_silhouette_flag:
                metrics_adjusted = evaluate_clustering(y_true, y_pred_adjusted, X_data_for_eval)
            else:
                metrics_adjusted = evaluate_clustering_wos(y_true, y_pred_adjusted)
            print("Clustering Scores (Adjusted - using 'adjusted_cluster' column): ", metrics_adjusted)
        else:
            print(f"[WARN Eval] Size mismatch or empty arrays for adjusted metrics. y_pred_adjusted size: {y_pred_adjusted.size}, y_true size: {y_true.size}")
    else:
        print("[WARN Eval] 'adjusted_cluster' column not available or length mismatch. Skipping adjusted metrics calculation.")

    # Fallback for y_pred if 'cluster' column was not populated, for functions expecting a single y_pred
    # This part was from the chunking logic, ensure it's still relevant or integrated with metrics_original
    y_pred_for_general_use = final_predict_results
    if len(y_pred_for_general_use) != len(y_true) and len(y_true) > 0:
        print(f"[WARN Eval] Length mismatch between y_true ({len(y_true)}) and y_pred_for_general_use ({len(y_pred_for_general_use)}). Evaluation might be incorrect or fail.")
        if len(y_pred_for_general_use) == 0 and len(y_true) > 0:
            print("[WARN Eval] y_pred_for_general_use is empty. Using dummy predictions (all zeros) to avoid crash.")
            y_pred_for_general_use = np.zeros_like(y_true)
    
    # Ensure y_pred_for_general_use is not empty before general evaluation (if any part still uses it directly)
    # The primary evaluation is now through metrics_original and metrics_adjusted.

    # --- This section with diff_columns might be redundant if metrics_original/adjusted cover all cases ---
    # if len(y_pred_for_general_use) > 0:
    #     # if 'label' is an original column, use evaluate_clustering_wos, otherwise use evaluate_clustering.
    #     if 'label' in diff_columns: # diff_columns was defined much earlier, check its scope and relevance
    #         # This logic might be superseded by the specific metrics_original/metrics_adjusted calls above
    #         pass # clustering_scores = evaluate_clustering_wos(y_true, y_pred_for_general_use)
    #     else:
    #         pass # clustering_scores = evaluate_clustering(y_true.flatten(), y_pred_for_general_use.flatten(), X_data_for_eval)
    #     # print("Clustering Scores (based on concatenated results / y_pred_for_general_use): ", clustering_scores)
    # else:
    #     print("[ERROR] No predicted labels (y_pred_for_general_use is empty). Skipping some evaluation parts.")
    # --- End potentially redundant section ---

    timing_info['6_evaluation_time'] = time.time() - start
    print(f"Step 6 finished. Evaluation Time: {timing_info['6_evaluation_time']:.2f}s")

    # 7. Save results
    start = time.time()

    determined_gmm_type = None 
    if clustering_algorithm.upper().startswith("GMM"): # "GMM", "GMM_full", "SGMM" 등
        parts = clustering_algorithm.split('_')
        if len(parts) == 1 and parts[0].upper() == 'GMM': 
            determined_gmm_type = "normal"
        elif len(parts) == 2 and parts[0].upper() == 'GMM' and parts[1].lower() in ['normal', 'full', 'tied', 'diag']:
            determined_gmm_type = parts[1].lower()
        # For cases like SGMM, determined_gmm_type can be kept as None here (SGMM does not directly use GMM_type)

        # Use GMM type from all_chunk_gmm_types if available, as it reflects what was actually run
        if autotune_enabled and all_chunk_gmm_types: # Check autotune_enabled here for clarity
            if all_chunk_gmm_types[0] is not None: 
                 determined_gmm_type = all_chunk_gmm_types[0]
        elif not autotune_enabled and all_chunk_gmm_types: # Non-Autotune path, still check collected type
             if all_chunk_gmm_types[0] is not None:
                  determined_gmm_type = all_chunk_gmm_types[0]
    
    timing_info['7_save_time_start_hook'] = time.time() - start # Placeholder, real save time is at the end

    time_save_csv_VL(file_type, file_number, clustering_algorithm, 
                     timing_info)

    if 'cluster' in data.columns and len(data['cluster']) == len(y_true):
        # For csv_compare_clustering, the arguments were simplified in the save_csv.py
        # Original call might have been:
        # csv_compare_clustering(data, y_true, data['cluster'].to_numpy(), metrics_original, 
        #                        file_type, file_number, heterogeneous_method, clustering_algorithm, 
        #                        Association_mathod, eval_clustering_silhouette, 
        #                        gmm_type=determined_gmm_type)
        # Matching the current definition in save_csv.py:
        # def csv_compare_clustering(file_type, clusterint_method, file_number, data, GMM_type=None):
        csv_compare_clustering(
            file_type, # file_type
            clustering_algorithm, # clusterint_method
            file_number, # file_number
            data, # data (DataFrame containing 'cluster', 'adjusted_cluster', 'label')
            GMM_type=determined_gmm_type, # GMM_type
            optimal_cni_threshold=optimal_cni_threshold # Pass the determined optimal threshold
        )
        
        # For csv_compare_matrix_clustering, arguments were simplified too.
        # Original call might have been:
        # csv_compare_matrix_clustering(y_true, data['cluster'].to_numpy(), data['label'], 
        #                               file_type, file_number, heterogeneous_method, clustering_algorithm, 
        #                               Association_mathod, eval_clustering_silhouette, 
        #                               gmm_type=determined_gmm_type)
        # Matching the current definition in save_csv.py:
        # def csv_compare_matrix_clustering(file_type, file_number, clusterint_method, metrics_original, metrics_adjusted, GMM_type):
        csv_compare_matrix_clustering(
            file_type, # file_type
            file_number, # file_number
            clustering_algorithm, # clusterint_method
            metrics_original, # metrics_original
            metrics_adjusted, # metrics_adjusted
            GMM_type=determined_gmm_type, # GMM_type
            optimal_cni_threshold=optimal_cni_threshold # Pass the determined optimal threshold
        )
    else:
        print("[WARN Save] 'cluster' column not available or length mismatch with y_true. Skipping CSV result comparison saving.")


    timing_info['7_save_time'] = time.time() - start # This is the actual end of step 7 including saves
    print(f"Step 7 finished. Save Time: {timing_info['7_save_time']:.2f}s")


    # Calculate total time
    total_end_time = time.time()
    timing_info['0_total_time'] = total_end_time - total_start_time

    # Save time information as a CSV
    time_save_csv_VL(file_type, file_number, clustering_algorithm, timing_info)


    return


if __name__ == '__main__':
    main()