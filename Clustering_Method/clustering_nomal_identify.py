# Identify nomal clusters and anomalous clusters with nomal data
# Input 'data' is initial data

import numpy as np
import multiprocessing
import os
# from utils.class_row import nomal_class_data
import pandas as pd
import gc
import time
import logging
from sklearn.metrics import jaccard_score

from Dataset_Choose_Rule.choose_amount_dataset import file_path_line_nonnumber, file_cut
from utils.generate_data_hash import generate_stable_data_hash
from utils.apply_labeling import apply_labeling_logic
from utils.time_transfer import time_scalar_transfer
from Modules.Heterogeneous_module import choose_heterogeneous_method
from Heterogeneous_Method.separate_group_mapping import map_intervals_to_groups
from Modules.PCA import pca_func


# Setup logger
logger = logging.getLogger(__name__)


def _create_structured_dtype(num_features, base_dtype):
    """
    Create a structured dtype so entire rows can be compared in vectorized form.
    """
    if num_features <= 0:
        num_features = 1
    return np.dtype({
        'names': [f'f{i}' for i in range(num_features)],
        'formats': [base_dtype] * num_features
    })


def _as_structured_view(array, structured_dtype):
    """
    Return a 1-D structured view of the given 2-D feature array.
    """
    if array is None or array.size == 0:
        return np.empty(0, dtype=structured_dtype)
    contiguous = np.ascontiguousarray(array)
    if contiguous.ndim == 1:
        contiguous = contiguous.reshape(1, -1)
    return contiguous.view(structured_dtype).reshape(-1)


def _build_normal_lookup(normal_samples, structured_dtype, batch_size=None):
    """
    Build a deduplicated lookup table of normal samples (as structured rows).
    When batch_size is provided, the unique rows are accumulated per batch
    to reduce peak memory usage.
    """
    if normal_samples is None or normal_samples.size == 0:
        return np.empty(0, dtype=structured_dtype)

    normal_samples = np.ascontiguousarray(normal_samples)
    total_rows = normal_samples.shape[0]

    if batch_size is None or total_rows <= batch_size:
        structured = _as_structured_view(normal_samples, structured_dtype)
        return np.unique(structured)

    unique_blocks = []
    for start in range(0, total_rows, batch_size):
        chunk = normal_samples[start:start + batch_size]
        if chunk.size == 0:
            continue
        unique_chunk = np.unique(chunk, axis=0)
        if unique_chunk.size > 0:
            unique_blocks.append(unique_chunk)

    if not unique_blocks:
        return np.empty(0, dtype=structured_dtype)

    concatenated = np.vstack(unique_blocks)
    structured_concat = _as_structured_view(concatenated, structured_dtype)
    return np.unique(structured_concat)


# Helper function for parallel processing of a single cluster
#def _process_single_cluster_label(cluster_id, data_features_for_clustering, clusters_assigned, known_normal_samples_features_to_compare, threshold):
def _process_single_cluster_label(cluster_id, data_features_for_clustering, clusters_assigned, normal_lookup_info, threshold):
    cluster_mask = (clusters_assigned == cluster_id)
    
    if not np.any(cluster_mask):
        print(f"[DEBUG CNI Cluster {cluster_id}] Empty cluster, returning -1")
        return cluster_id, None, -1, -1 # cluster_id, mask, label (-1 indicates no label assigned / empty cluster)

    current_cluster_features = data_features_for_clustering[cluster_mask]

    normal_lookup_structured, structured_dtype, target_dtype = normal_lookup_info

    if current_cluster_features.size == 0 or normal_lookup_structured.size == 0:
        num_normal_in_cluster = 0
    else:
        try:
            current_cluster_features = np.asarray(current_cluster_features, dtype=target_dtype)
            if current_cluster_features.ndim == 1:
                current_cluster_features = current_cluster_features.reshape(1, -1)
            
            '''
            # Memory-efficient streaming approach
            try:
                # Calculate memory requirements
                n_current = current_cluster_features.shape[0]
                n_known = known_normal_samples_features_to_compare.shape[0]
                n_features = current_cluster_features.shape[1]
                
                # Estimate memory usage for full comparison
                estimated_memory_gb = (n_current * n_known * n_features * 1) / (1024**3)  # 1 byte per boolean
                
                # Use streaming approach if memory requirement is too high
                if estimated_memory_gb > 1.0:  # If more than 1GB, use streaming
                    # print(f"[DEBUG CNI Cluster {cluster_id}] Using streaming approach (chunk size: 10000)")
                    
                    # More efficient streaming with larger chunks and optimized comparison
                    chunk_size = 10000  # Increased chunk size for better performance
                    num_normal_in_cluster = 0
                    
                    # Pre-compute known normal samples for faster lookup with rounding
                    known_normal_set = set()
                    for i in range(0, n_known, 1000):  # Process known samples in chunks too
                        end_i = min(i + 1000, n_known)
                        known_chunk = known_normal_samples_features_to_compare[i:end_i]
                        for row in known_chunk:
                            # Convert numpy array to tuple to be hashable for set insertion
                            known_normal_set.add(tuple(row))
                    
                    # print(f"[DEBUG CNI Cluster {cluster_id}] Created lookup set with {len(known_normal_set)} known normal samples")
                    
                    # Process current cluster samples
                    for i in range(0, n_current, chunk_size):
                        end_i = min(i + chunk_size, n_current)
                        current_chunk = current_cluster_features[i:end_i]
                        
                        # Use set lookup for exact matches
                        for row in current_chunk:
                            if tuple(row) in known_normal_set:
                                num_normal_in_cluster += 1
                        
                        # Periodic garbage collection
                        if i % (chunk_size * 5) == 0:
                            import gc
                            gc.collect()
                else:
                    # Use direct approach for small datasets
                    # print(f"[DEBUG CNI Cluster {cluster_id}] Using direct approach (small dataset)")
                    comparison_matrix = (current_cluster_features[:, None, :] == known_normal_samples_features_to_compare[None, :, :])
                    all_features_match = np.all(comparison_matrix, axis=2)
                    any_known_normal_matches = np.any(all_features_match, axis=1)
                    num_normal_in_cluster = np.sum(any_known_normal_matches)
                    
                    # Cleanup
                    del comparison_matrix, all_features_match, any_known_normal_matches
                
                # print(f"[DEBUG CNI Cluster {cluster_id}] Memory-efficient processing completed")
                
            except ValueError as e:
                print(f"[DEBUG CNI Cluster {cluster_id}] ValueError: {e}, treating as anomalous")
                num_normal_in_cluster = 0
            except MemoryError as me:
                print(f"[DEBUG CNI Cluster {cluster_id}] Memory error: {me}, treating as anomalous")
                num_normal_in_cluster = 0
            except Exception as e:
                print(f"[DEBUG CNI Cluster {cluster_id}] Unexpected error: {e}, treating as anomalous")
                num_normal_in_cluster = 0
                
                print(f"[DEBUG CNI Cluster {cluster_id}] Number of normal matches: {num_normal_in_cluster}")
            '''

            unique_rows, row_counts = np.unique(current_cluster_features, axis=0, return_counts=True)
            if unique_rows.size == 0:
                num_normal_in_cluster = 0
            else:
                structured_cluster_rows = _as_structured_view(unique_rows, structured_dtype)
                if structured_cluster_rows.size == 0:
                    num_normal_in_cluster = 0
                else:
                    matches = np.isin(structured_cluster_rows, normal_lookup_structured, assume_unique=False)
                    if np.any(matches):
                        num_normal_in_cluster = int(row_counts[matches].sum())
                    else:
                        num_normal_in_cluster = 0
                
        except ValueError as e:
            num_normal_in_cluster = 0 # Treat as anomalous on error
            print(f"[DEBUG CNI Cluster {cluster_id}] ValueError during comparison: {e}, num_normal_in_cluster = 0")
        except MemoryError as me:
            num_normal_in_cluster = 0 # Treat as anomalous on error
            print(f"[DEBUG CNI Cluster {cluster_id}] MemoryError during comparison: {me}, num_normal_in_cluster = 0")
    
    normal_ratio = num_normal_in_cluster / len(current_cluster_features) if len(current_cluster_features) > 0 else 0
    label_for_final_output = 0 if normal_ratio >= threshold else 1
    
    # return cluster_id, cluster_mask, label_for_final_output
    # print(f"[DEBUG CNI Cluster {cluster_id}] Normal ratio: {normal_ratio:.4f}, Final label: {label_for_final_output}")
    
    # MODIFIED: Return normal_ratio as well
    return cluster_id, cluster_mask, label_for_final_output, normal_ratio

# Change function signature:
# data_features_for_clustering: NumPy array of features used for clustering (e.g. X_reduced, shape (N, num_pca_features))
# original_labels_aligned: Original labels for each row in data_features_for_clustering (0 or 1, shape (N,))
# clusters_assigned: Cluster IDs assigned to each row in data_features_for_clustering (shape (N,))
# num_total_clusters: Total number of clusters generated by the clustering algorithm
# global_known_normal_samples_pca: Pre-sampled (e.g., 80% of all known normals) PCA features of known normal samples from the entire dataset.
def clustering_nomal_identify(data_features_for_clustering, clusters_assigned, original_labels_aligned, global_known_normal_samples_pca=None, threshold_value=0.3, num_processes_for_algo=None, data_for_clustering=None, known_normal_idx=None, normal_data_batch_size=None): # Added data_for_clustering to match call signature, but it is not used here
    
    print(f"[DEBUG CNI] clustering_nomal_identify called - data shape: {data_features_for_clustering.shape}, clusters: {len(np.unique(clusters_assigned))}")
    
    if global_known_normal_samples_pca is None:
        print(f"[DEBUG CNI] global_known_normal_samples_pca is None, using fallback logic...")
        if original_labels_aligned is not None and data_features_for_clustering.shape[0] == original_labels_aligned.shape[0]:
            temp_known_normal_samples = data_features_for_clustering[original_labels_aligned == 0]
            num_temp_known_normal = temp_known_normal_samples.shape[0]
            print(f"[DEBUG CNI] Found {num_temp_known_normal} normal samples in current chunk for fallback")
            
            if num_temp_known_normal > 1:
                sample_size = int(num_temp_known_normal * 0.90)
                if sample_size == 0 and num_temp_known_normal > 0: sample_size = 1
                if sample_size > num_temp_known_normal : sample_size = num_temp_known_normal
                if sample_size > 0:
                    # Fix random seed for consistency across processes
                    np.random.seed(42)
                    random_indices = np.random.choice(num_temp_known_normal, size=sample_size, replace=False)
                    global_known_normal_samples_pca = temp_known_normal_samples[random_indices]
                    # Ensure 2D array
                    if global_known_normal_samples_pca.ndim == 1:
                        global_known_normal_samples_pca = global_known_normal_samples_pca.reshape(1, -1)
                    print(f"[DEBUG CNI] Fallback: Created global_known_normal_samples_pca with shape {global_known_normal_samples_pca.shape}")
                else:
                    global_known_normal_samples_pca = np.array([])
                    print(f"[DEBUG CNI] Fallback: Created empty global_known_normal_samples_pca")
            elif num_temp_known_normal == 1:
                global_known_normal_samples_pca = temp_known_normal_samples
                # Ensure 2D array
                if global_known_normal_samples_pca.ndim == 1:
                    global_known_normal_samples_pca = global_known_normal_samples_pca.reshape(1, -1)
                print(f"[DEBUG CNI] Fallback: Using single normal sample as global_known_normal_samples_pca")
            else:
                global_known_normal_samples_pca = np.array([])
                print(f"[DEBUG CNI] Fallback: No normal samples found, created empty global_known_normal_samples_pca")
        else:
            print(f"[DEBUG CNI] Fallback failed: original_labels_aligned is None or shape mismatch")
            raise ValueError("CNI: global_known_normal_samples_pca must be provided, or fallback from original_labels_aligned failed.")
    else:
        print(f"[DEBUG CNI] Using provided global_known_normal_samples_pca with shape {global_known_normal_samples_pca.shape}")

    known_normal_samples_features_to_compare = global_known_normal_samples_pca
    print(f"[DEBUG CNI] After assignment - known_normal_samples_features_to_compare shape: {known_normal_samples_features_to_compare.shape if known_normal_samples_features_to_compare is not None else 'None'}")
    print(f"[DEBUG CNI] After assignment - known_normal_samples_features_to_compare ndim: {known_normal_samples_features_to_compare.ndim if known_normal_samples_features_to_compare is not None else 'N/A'}")

    # Ensure known_normal_samples_features_to_compare is properly formatted
    if known_normal_samples_features_to_compare is not None and known_normal_samples_features_to_compare.size > 0:
        if known_normal_samples_features_to_compare.ndim == 1:
            known_normal_samples_features_to_compare = known_normal_samples_features_to_compare.reshape(1, -1)
            print(f"[DEBUG CNI] Reshaped 1D to 2D: {known_normal_samples_features_to_compare.shape}")
        elif known_normal_samples_features_to_compare.ndim == 0:
            known_normal_samples_features_to_compare = np.array([])
            print(f"[DEBUG CNI] Converted 0D to empty array")
    else:
        if data_features_for_clustering.ndim == 2 and data_features_for_clustering.shape[1] > 0:
            known_normal_samples_features_to_compare = np.empty((0, data_features_for_clustering.shape[1]))
            print(f"[DEBUG CNI] Created empty 2D array: {known_normal_samples_features_to_compare.shape}")
        else:
            known_normal_samples_features_to_compare = np.array([])
            print(f"[DEBUG CNI] Created empty 1D array")
    
    print(f"[DEBUG CNI] Final known_normal_samples_features_to_compare shape: {known_normal_samples_features_to_compare.shape}")
    print(f"[DEBUG CNI] Final known_normal_samples_features_to_compare ndim: {known_normal_samples_features_to_compare.ndim}")

    # === START: New Robustness Checks ===
    unique_clusters = np.unique(clusters_assigned)
    
    # Handle cases with no valid clusters or a single cluster
    # If the only cluster is noise (-1), or there's only one cluster, CNI is not meaningful.
    valid_clusters = [c for c in unique_clusters if c != -1]
    if len(valid_clusters) <= 1:
        # If there's one cluster, label all as normal (0) by default.
        # Jaccard score will be calculated based on this assumption.
        final_labels = np.zeros_like(clusters_assigned, dtype=int)
        
        # If the single valid cluster exists, assign its members to 0, others (noise) remain 0.
        if len(valid_clusters) == 1:
            final_labels[clusters_assigned == valid_clusters[0]] = 0
        
        jaccard = 0.0
        if original_labels_aligned is not None:
            try:
                #jaccard = jaccard_score(original_labels_aligned, final_labels, average='weighted')
                orig_arr = np.asarray(original_labels_aligned, dtype=int).ravel()
                final_arr = np.asarray(final_labels, dtype=int).ravel()
                jaccard = jaccard_score(orig_arr, final_arr, labels=[0,1], average='weighted', zero_division=0)
            except Exception as e:
                print(f"[WARN CNI] Could not calculate Jaccard for single-cluster case: {e}")
        
        print(f"[DEBUG CNI] Single cluster detected (or only noise). Returning Jaccard={jaccard:.4f}.")
        return final_labels, jaccard, pd.DataFrame()
    # === END: New Robustness Checks ===

    final_labels = np.zeros(len(data_features_for_clustering), dtype=int)
    # MODIFIED: Prepare a list to store results for the DataFrame
    results_list = []
    # Prepare lookup info for worker tasks
    target_dtype = data_features_for_clustering.dtype if hasattr(data_features_for_clustering, "dtype") else known_normal_samples_features_to_compare.dtype
    if known_normal_samples_features_to_compare.size > 0 and known_normal_samples_features_to_compare.dtype != target_dtype:
        known_normal_samples_features_to_compare = known_normal_samples_features_to_compare.astype(target_dtype, copy=False)
    num_features = data_features_for_clustering.shape[1] if data_features_for_clustering.ndim > 1 else 1
    structured_dtype = _create_structured_dtype(num_features, target_dtype)
    normal_lookup_structured = _build_normal_lookup(
        known_normal_samples_features_to_compare,
        structured_dtype,
        batch_size=normal_data_batch_size
    )
    normal_lookup_info = (normal_lookup_structured, structured_dtype, target_dtype)

    tasks = [(cid, data_features_for_clustering, clusters_assigned, normal_lookup_info, threshold_value) for cid in valid_clusters]
    
    actual_clusters_to_process = len(tasks)
    print(f"[DEBUG CNI] Processing {actual_clusters_to_process} clusters")
    
    if actual_clusters_to_process == 0:
        return final_labels, 0.0, pd.DataFrame([]) # Return all zeros if no actual clusters

    if num_processes_for_algo is not None and num_processes_for_algo > 0:
        num_processes_to_use = min(num_processes_for_algo, actual_clusters_to_process) # Don't use more processes than tasks
    else:
        available_cpus = os.cpu_count()
        if available_cpus is None:
            num_processes_to_use = 1 # Fallback to 1 if CPU count cannot be determined
        elif num_processes_for_algo == 0:
            num_processes_to_use = available_cpus
        else: # num_processes_for_algo is None or invalid negative
            num_processes_to_use = max(1, int(available_cpus / 2))
        num_processes_to_use = min(num_processes_to_use, actual_clusters_to_process) # Ensure not more processes than tasks
        if num_processes_to_use == 0 and actual_clusters_to_process > 0 : num_processes_to_use = 1 # Ensure at least 1 process if there are tasks
    
    # --- MODIFIED: Skip Pool creation entirely if num_processes_to_use is 1 ---
    try:
        '''
        if actual_clusters_to_process > 0 and num_processes_to_use > 0:
            with multiprocessing.Pool(processes=num_processes_to_use) as pool:
                results = pool.starmap(_process_single_cluster_label, tasks)
            
            for cluster_id, cluster_mask, label_for_final_output in results:
                if cluster_mask is not None and label_for_final_output != -1:
                    final_labels[cluster_mask] = label_for_final_output
        else:
            for task_args in tasks:
                _ , cluster_mask_seq, label_for_final_output_seq = _process_single_cluster_label(*task_args)
                if cluster_mask_seq is not None and label_for_final_output_seq != -1:
                    final_labels[cluster_mask_seq] = label_for_final_output_seq
        '''
        
        if actual_clusters_to_process > 0:
            # Only create a multiprocessing Pool if more than 1 process is requested.
            if num_processes_to_use > 1:
                with multiprocessing.Pool(processes=num_processes_to_use) as pool:
                    results = pool.starmap(_process_single_cluster_label, tasks)
                
                # MODIFIED: Unpack new return value
                for cluster_id, cluster_mask, label_for_final_output, normal_ratio in results:
                    if cluster_mask is not None and label_for_final_output != -1:
                        final_labels[cluster_mask] = label_for_final_output
                        results_list.append({'cluster_id': cluster_id, 'final_label': label_for_final_output, 'normal_ratio': normal_ratio})
                        # print(f"[DEBUG CNI] Applied label {label_for_final_output} to cluster {cluster_id}")
            else: # If num_processes_to_use is 1, run sequentially.
                # print("[DEBUG CNI] Running sequentially as num_processes_to_use is 1.")
                for task_args in tasks:
                    # MODIFIED: Unpack new return value
                    cluster_id_seq, cluster_mask_seq, label_for_final_output_seq, normal_ratio_seq = _process_single_cluster_label(*task_args)
                    if cluster_mask_seq is not None and label_for_final_output_seq != -1:
                        final_labels[cluster_mask_seq] = label_for_final_output_seq
                        results_list.append({'cluster_id': cluster_id_seq, 'final_label': label_for_final_output_seq, 'normal_ratio': normal_ratio_seq})
                        # print(f"[DEBUG CNI] Applied label {label_for_final_output_seq} to cluster {task_args[0]}")
        # No 'else' needed here as final_labels is already initialized with zeros.

    except Exception as e:
        print(f"[ERROR CNI] Processing failed: {e}. Falling back to sequential processing for remaining tasks or all tasks.")
        results_list = [] # Clear partial results
        for task_args in tasks:
            # MODIFIED: Unpack new return value
            cluster_id_seq, cluster_mask_seq, label_for_final_output_seq, normal_ratio_seq = _process_single_cluster_label(*task_args)
            if cluster_mask_seq is not None and label_for_final_output_seq != -1:
                final_labels[cluster_mask_seq] = label_for_final_output_seq
                results_list.append({'cluster_id': cluster_id_seq, 'final_label': label_for_final_output_seq, 'normal_ratio': normal_ratio_seq})

    # Final summary
    normal_count = np.sum(final_labels == 0)
    anomaly_count = np.sum(final_labels == 1)
    total_count = len(final_labels)
    # print(f"[DEBUG CNI] Final results: {normal_count} normal, {anomaly_count} anomaly (ratio: {normal_count/total_count:.4f})")
    
    # MODIFIED: Create DataFrame and calculate Jaccard score
    final_results_df = pd.DataFrame(results_list)
    
    jaccard = 0.0
    if original_labels_aligned is not None:
        try:
            #jaccard = jaccard_score(original_labels_aligned, final_labels, average='weighted')
            orig_arr = np.asarray(original_labels_aligned, dtype=int).ravel()
            final_arr = np.asarray(final_labels, dtype=int).ravel()
            jaccard = jaccard_score(orig_arr, final_arr, labels=[0,1], average='weighted', zero_division=0)
        except Exception as e:
            print(f"[WARN CNI] Could not calculate Jaccard score: {e}")
            jaccard = 0.0

    # Summary log instead of per-cluster logs
    normal_clusters = sum(1 for row in final_results_df.itertuples() if row.normal_ratio >= threshold_value)
    total_clusters = len(final_results_df)
    print(f"[CNI SUMMARY] Processed {total_clusters} clusters: {normal_clusters} normal, {total_clusters - normal_clusters} anomalous. Jaccard: {jaccard:.4f}")

    return final_labels, jaccard, final_results_df


# --- For best_clustering_selector_parallel.py ---
def create_global_reference_normal_samples(file_type, file_number, heterogeneous_method):
    """
    Loads the full dataset, processes it using the new robust pipeline 
    (feature selection, cleaning, scaling, PCA), and extracts a global reference 
    normal samples set. This function's pipeline MUST mirror the main() function's pipeline.
    
    MODIFIED: Now also returns the indices of the known normal samples.
    """
    logger.info("Step 0: Creating Global Reference Normal Samples PCA and Indices...")
    start_global_ref = time.time()
    global_known_normal_samples_pca_for_cni = None
    known_normal_indices_for_surrogate = None # New variable to store indices
    
    try:
        file_path_for_global_ref, _ = file_path_line_nonnumber(file_type, file_number)
        
        # Load FULL data for reference normal selection
        logger.info("[GlobalRef] Loading full data for reference normal selection...")
        full_data_for_ref = file_cut(file_type, file_path_for_global_ref, 'all')
        full_data_for_ref.columns = full_data_for_ref.columns.str.strip()

        # Apply labeling logic using reusable function
        full_data_for_ref = apply_labeling_logic(full_data_for_ref, file_type)
        ref_labels = full_data_for_ref['label'].to_numpy() # Store labels early

        ref_X_pca = None

        # --- SELECTIVE PREPROCESSING PIPELINE (Mirrors main()) ---
        if heterogeneous_method == 'scaling_label_encoding':
            logger.info("[GlobalRef] Using 'scaling_label_encoding' pipeline.")
            
            # Step 3 (GlobalRef): Feature Selection
            from Heterogeneous_Method.Feature_Encoding import Heterogeneous_Feature_named_featrues
            feature_dict = Heterogeneous_Feature_named_featrues(file_type)
            all_features = feature_dict.get('categorical_features', []) + \
                           feature_dict.get('time_features', []) + \
                           feature_dict.get('packet_length_features', []) + \
                           feature_dict.get('count_features', []) + \
                           feature_dict.get('binary_features', [])
            existing_features = [f for f in all_features if f in full_data_for_ref.columns]
            
            # Select only the necessary columns + label for processing
            ref_data_for_processing = full_data_for_ref[existing_features + ['label']].copy()

            # Step 4 (GlobalRef): Time Scalar Transfer
            ref_data_for_processing = time_scalar_transfer(ref_data_for_processing, file_type)

            # Step 5 (GlobalRef): Data Cleaning
            numerical_features = feature_dict.get('time_features', []) + \
                                 feature_dict.get('packet_length_features', []) + \
                                 feature_dict.get('count_features', [])
            columns_to_clean = [col for col in numerical_features if col in ref_data_for_processing.columns]
            
            if columns_to_clean:
                # Coerce to numeric, handle inf, drop NaN (same as main)
                for col in columns_to_clean:
                    if not pd.api.types.is_numeric_dtype(ref_data_for_processing[col]):
                        ref_data_for_processing[col] = pd.to_numeric(ref_data_for_processing[col], errors='coerce')
                for col in columns_to_clean:
                    if not np.isfinite(ref_data_for_processing[col].values).all():
                        if np.isinf(ref_data_for_processing[col].values).any():
                            finite_max = ref_data_for_processing.loc[np.isfinite(ref_data_for_processing[col]), col].max()
                            replacement_val = 0 if pd.isna(finite_max) else finite_max
                            ref_data_for_processing[col].replace([np.inf, -np.inf], replacement_val, inplace=True)
                
                # Update labels after potential row drops
                initial_rows = len(ref_data_for_processing)
                ref_data_for_processing.dropna(subset=columns_to_clean, inplace=True)
                if len(ref_data_for_processing) < initial_rows:
                    ref_labels = ref_data_for_processing['label'].to_numpy()

            # Step 6 (GlobalRef): Final Scaling/Encoding
            ref_X = ref_data_for_processing.drop(columns=['label'])
            ref_X_processed, _, _, _ = choose_heterogeneous_method(ref_X, file_type, 'scaling_label_encoding', regul='N')

            # Step 7 (GlobalRef): PCA
            ref_pca_want = 'N' if file_type in ['DARPA98', 'DARPA', 'CICModbus23', 'CICModbus'] else 'Y'
            if ref_pca_want == 'Y':
                ref_X_pca = pca_func(ref_X_processed)
            else:
                ref_X_pca = ref_X_processed.to_numpy() if hasattr(ref_X_processed, 'to_numpy') else ref_X_processed

        else: # Pipeline B for Interval_inverse and others
            logger.info(f"[GlobalRef] Using default embedding pipeline for '{heterogeneous_method}'.")

            # Step 3 (GlobalRef): Time Scalar Transfer
            ref_data_processed = time_scalar_transfer(full_data_for_ref, file_type)
            
            # Step 4 (GlobalRef): Embedding and Group Mapping
            regul = 'N'
            ref_embedded_df, _, ref_cat_map, ref_data_list = choose_heterogeneous_method(ref_data_processed, file_type, heterogeneous_method, regul)
            ref_group_mapped_df, _ = map_intervals_to_groups(ref_embedded_df, ref_cat_map, ref_data_list, regul)

            # Step 5 (GlobalRef): MinMax Scaling
            ref_X_scaled = pd.DataFrame(MinMaxScaler().fit_transform(ref_group_mapped_df), columns=ref_group_mapped_df.columns, index=ref_group_mapped_df.index)

            # Step 6 (GlobalRef): PCA
            ref_pca_want = 'N' if file_type in ['DARPA98', 'DARPA', 'CICModbus23', 'CICModbus'] else 'Y'
            if ref_pca_want == 'Y':
                ref_X_pca = pca_func(ref_X_scaled) 
            else:
                ref_X_pca = ref_X_scaled.to_numpy() if hasattr(ref_X_scaled, 'to_numpy') else ref_X_scaled

        # --- END OF PIPELINES ---

        # Create global reference from the final processed normal samples
        if ref_X_pca is not None:
            # Get the indices of all normal samples from the original, pre-processed data
            all_normal_indices_ref = np.where(ref_labels == 0)[0]
            all_normal_samples_pca_ref = ref_X_pca[all_normal_indices_ref]

            num_all_normal_ref = all_normal_samples_pca_ref.shape[0]
            logger.info(f"[GlobalRef] Total normal samples in full data (PCA space): {num_all_normal_ref}")

            if num_all_normal_ref > 1:
                sample_size_ref = int(num_all_normal_ref * 0.90)
                if sample_size_ref == 0 and num_all_normal_ref > 0: sample_size_ref = 1
                
                np.random.seed(42) # Ensure reproducibility
                random_indices_ref = np.random.choice(num_all_normal_ref, size=sample_size_ref, replace=False)
                
                # Get the PCA data for the sampled normals
                global_known_normal_samples_pca_for_cni = all_normal_samples_pca_ref[random_indices_ref]
                # Get the ORIGINAL indices for the sampled normals
                known_normal_indices_for_surrogate = all_normal_indices_ref[random_indices_ref]

                logger.info(f"[GlobalRef] Global reference normal samples (90% of all normals, PCA space) created. Shape: {global_known_normal_samples_pca_for_cni.shape}")
                logger.info(f"[GlobalRef] Corresponding original indices created. Shape: {known_normal_indices_for_surrogate.shape}")

            elif num_all_normal_ref == 1:
                global_known_normal_samples_pca_for_cni = all_normal_samples_pca_ref
                known_normal_indices_for_surrogate = all_normal_indices_ref
                logger.info(f"[GlobalRef] Global reference normal samples (1 sample from full data, PCA space) created. Shape: {global_known_normal_samples_pca_for_cni.shape}")
            else:
                logger.warning("[WARN GlobalRef] No normal samples found in the full dataset to create global reference.")
                if ref_X_pca.ndim == 2 and ref_X_pca.shape[1] > 0:
                    global_known_normal_samples_pca_for_cni = np.empty((0, ref_X_pca.shape[1]))
                    known_normal_indices_for_surrogate = np.array([])
                else:
                    global_known_normal_samples_pca_for_cni = np.array([])
                    known_normal_indices_for_surrogate = np.array([])
        else:
            logger.error("[ERROR GlobalRef] PCA data (ref_X_pca) was not generated. Cannot create reference samples.")
            global_known_normal_samples_pca_for_cni = np.array([])
            known_normal_indices_for_surrogate = np.array([])

        del full_data_for_ref
        if 'ref_data_for_processing' in locals(): del ref_data_for_processing
        if 'ref_X' in locals(): del ref_X
        if 'ref_X_processed' in locals(): del ref_X_processed
        if 'ref_data_processed' in locals(): del ref_data_processed
        if 'ref_embedded_df' in locals(): del ref_embedded_df
        if 'ref_group_mapped_df' in locals(): del ref_group_mapped_df
        if 'ref_X_scaled' in locals(): del ref_X_scaled
        if 'all_normal_samples_pca_ref' in locals(): del all_normal_samples_pca_ref
        logger.info("[GlobalRef] Freed memory from temporary full data load.")

    except Exception as e:
        logger.error(f"[ERROR GlobalRef] Failed to create global reference normal samples: {e}. Proceeding without it.")
        import traceback
        traceback.print_exc()
        global_known_normal_samples_pca_for_cni = None
        known_normal_indices_for_surrogate = None

    logger.info(f"Step 0 finished. Time: {time.time() - start_global_ref:.2f}s. Global ref shape: {global_known_normal_samples_pca_for_cni.shape if global_known_normal_samples_pca_for_cni is not None else 'None'}")
    return global_known_normal_samples_pca_for_cni, known_normal_indices_for_surrogate
