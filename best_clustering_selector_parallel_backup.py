#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Best Clustering Algorithm Selector - Parallel Version
Run all clustering algorithms and select the algorithm with the highest Jaccard coefficient to save the results
Used with run_paral.sh to process multiple datasets simultaneously
Based on Data_Labeling.py with all core features included
"""

# Essential imports
import argparse
import numpy as np
import pandas as pd
import time
import math
import os
import csv
import sys
import logging
import psutil
import gc
import multiprocessing
from datetime import datetime

# Set NUMEXPR_MAX_THREADS to prevent threading issues
os.environ['NUMEXPR_MAX_THREADS'] = '128'

# Core functionality imports
from sklearn.preprocessing import MinMaxScaler
from Dataset_Choose_Rule.choose_amount_dataset import file_path_line_nonnumber, file_cut
from definition.Anomal_Judgment import anomal_judgment_nonlabel, anomal_judgment_label
from utils.time_transfer import time_scalar_transfer
from Modules.Heterogeneous_module import choose_heterogeneous_method
from Heterogeneous_Method.separate_group_mapping import map_intervals_to_groups
from Modules.PCA import pca_func
from utils.cluster_adjust_mapping import cluster_mapping
from Clustering_Method.clustering_score import evaluate_clustering, evaluate_clustering_wos

# Conditional imports - loaded when needed
def get_autotune_imports():
    """Lazy import for autotune functionality"""
    try:
        print("[DEBUG] Attempting to import Modules.Clustering_Algorithm_Autotune...")
        from Modules.Clustering_Algorithm_Autotune import choose_clustering_algorithm
        print("[DEBUG] Successfully imported choose_clustering_algorithm")
        
        print("[DEBUG] Attempting to import Modules.Clustering_Algorithm_Nonautotune...")
        from Modules.Clustering_Algorithm_Nonautotune import choose_clustering_algorithm_Non_optimization
        print("[DEBUG] Successfully imported choose_clustering_algorithm_Non_optimization")
        
        return choose_clustering_algorithm, choose_clustering_algorithm_Non_optimization
    except ImportError as e:
        print(f"[ERROR] Import error in get_autotune_imports: {e}")
        print(f"[ERROR] Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise e

def get_jaccard_elbow_imports():
    """Lazy import for Jaccard elbow functionality"""
    from Modules.Jaccard_Elbow_Method import test_all_algorithms_with_jaccard_elbow
    return test_all_algorithms_with_jaccard_elbow

def get_save_imports():
    """Lazy import for save functionality"""
    from utils.minmaxscaler import apply_minmax_scaling_and_save_scalers
    from Dataset_Choose_Rule.save_csv import csv_compare_clustering_ex, csv_compare_matrix_clustering_ex
    from Dataset_Choose_Rule.time_save import time_save_csv_VL_ex
    return apply_minmax_scaling_and_save_scalers, csv_compare_clustering_ex, csv_compare_matrix_clustering_ex, time_save_csv_VL_ex

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of supported clustering algorithms (excluding CANNwKNN)
CLUSTERING_ALGORITHMS = [
    'kmeans', 'kmedians', 'GMM', 'SGMM', 'Gmeans', 'Xmeans', 
    'DBSCAN', 'MShift', 'FCM', 'CK', 'NeuralGas'
]

def get_system_resources():
    """Get system resource information."""
    cpu_count = os.cpu_count()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    logger.info(f"System Resources:")
    logger.info(f"  CPU Cores: {cpu_count}")
    logger.info(f"  Total RAM: {memory.total / (1024**3):.2f} GB")
    logger.info(f"  Available RAM: {memory.available / (1024**3):.2f} GB")
    logger.info(f"  Total Disk: {disk.total / (1024**3):.2f} GB")
    logger.info(f"  Available Disk: {disk.free / (1024**3):.2f} GB")
    
    return cpu_count, memory, disk

def apply_labeling_logic(data, file_type):
    """Apply labeling logic based on file type - reusable function"""
    if file_type in ['MiraiBotnet', 'NSL-KDD', 'NSL_KDD']:
        data['label'], _ = anomal_judgment_nonlabel(file_type, data)
    elif file_type == 'netML':
        data['label'] = data['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
    elif file_type == 'DARPA98':
        data['label'] = data['Class'].apply(lambda x: 0 if str(x).strip() == '-' else 1)
    elif file_type in ['CICIDS2017', 'CICIDS']:
        if 'Label' in data.columns:
            data['label'] = data['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
        else:
            logger.error("'Label' column not found in data")
            data['label'] = 0
    elif file_type in ['CICModbus23', 'CICModbus']:
        data['label'] = data['Attack'].apply(lambda x: 0 if str(x).strip() == 'Baseline Replay: In position' else 1)
    elif file_type in ['IoTID20', 'IoTID']:
        data['label'] = data['Label'].apply(lambda x: 0 if str(x).strip() == 'Normal' else 1)
    elif file_type == 'Kitsune':
        data['label'] = data['Label']
    elif file_type in ['CICIoT', 'CICIoT2023']:
        data['label'] = data['attack_flag']
    else:
        logger.warning(f"Using generic anomal_judgment_label for {file_type}")
        data['label'] = anomal_judgment_label(data)
    
    return data

def cleanup_memory():
    """Clean up memory by forcing garbage collection"""
    gc.collect()

def create_global_reference_normal_samples(file_type, file_number, heterogeneous_method):
    """Create global reference normal samples PCA for CNI function (Step 0 from Data_Labeling.py)"""
    logger.info("Step 0: Creating Global Reference Normal Samples PCA...")
    start_global_ref = time.time()
    global_known_normal_samples_pca_for_cni = None
    
    try:
        file_path_for_global_ref, _ = file_path_line_nonnumber(file_type, file_number)
        
        # Load FULL data for reference normal selection
        logger.info("[DEBUG GlobalRef] Loading full data for reference normal selection...")
        full_data_for_ref = file_cut(file_type, file_path_for_global_ref, 'all')
        full_data_for_ref.columns = full_data_for_ref.columns.str.strip()
        if logger.isEnabledFor(logging.DEBUG):
            logger.info(f"[DEBUG GlobalRef] Full data for ref loaded. Shape: {full_data_for_ref.shape}")

        # Apply labeling logic using reusable function
        full_data_for_ref = apply_labeling_logic(full_data_for_ref, file_type)
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.info(f"[DEBUG GlobalRef] Full data labeled. Label counts: {full_data_for_ref['label'].value_counts(dropna=False)}")

        # Apply same embedding and group mapping
        full_data_for_ref_processed = time_scalar_transfer(full_data_for_ref.copy(), file_type)
        ref_embedded_df, _, ref_cat_map, ref_data_list = choose_heterogeneous_method(full_data_for_ref_processed, file_type, heterogeneous_method, 'N')
        ref_group_mapped_df, _ = map_intervals_to_groups(ref_embedded_df, ref_cat_map, ref_data_list, 'N')
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.info(f"[DEBUG GlobalRef] Full data group mapped. Shape: {ref_group_mapped_df.shape}")
            logger.info(f"[DEBUG GlobalRef] Group mapped columns: {list(ref_group_mapped_df.columns)}")
        
        # Store labels before scaling to maintain index alignment
        ref_labels_before_scaling = full_data_for_ref['label'].values
        logger.info(f"[DEBUG GlobalRef] Labels before scaling shape: {ref_labels_before_scaling.shape}")
        logger.info(f"[DEBUG GlobalRef] Normal samples count before scaling: {np.sum(ref_labels_before_scaling == 0)}")
        
        # Apply MinMax scaling
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
        logger.info(f"[DEBUG GlobalRef] Full data scaled. Shape: {ref_X_scaled.shape}")
        logger.info(f"[DEBUG GlobalRef] Scaled columns: {list(ref_X_scaled.columns)}")
        
        # NaN Check (After Scaling)
        if not ref_X_scaled.empty:
            nan_check = ref_X_scaled.isnull().sum()
            logger.info(f"[DEBUG GlobalRef] NaN count per column after scaling: {nan_check.to_dict()}")

        # Apply PCA
        ref_pca_want = 'N' if file_type in ['DARPA98', 'DARPA', 'CICModbus23', 'CICModbus'] else 'Y'
        if ref_pca_want == 'Y':
            logger.info("[DEBUG GlobalRef] Applying PCA to scaled full data for reference...")
            ref_X_pca = pca_func(ref_X_scaled) 
        else:
            logger.info("[DEBUG GlobalRef] Skipping PCA for reference normal generation based on file_type.")
            ref_X_pca = ref_X_scaled.to_numpy() if hasattr(ref_X_scaled, 'to_numpy') else ref_X_scaled
        logger.info(f"[DEBUG GlobalRef] Full data PCA applied. Shape: {ref_X_pca.shape}")

        # Create global reference from normal samples
        all_normal_samples_pca_ref = ref_X_pca[full_data_for_ref['label'] == 0]
        num_all_normal_ref = all_normal_samples_pca_ref.shape[0]
        logger.info(f"[DEBUG GlobalRef] Total normal samples in full data (PCA space): {num_all_normal_ref}")
        logger.info(f"[DEBUG GlobalRef] Using correct labels for normal sample selection")

        if num_all_normal_ref > 1:
            sample_size_ref = int(num_all_normal_ref * 0.90)
            if sample_size_ref == 0 and num_all_normal_ref > 0: 
                sample_size_ref = 1
            if sample_size_ref > num_all_normal_ref: 
                sample_size_ref = num_all_normal_ref 
            
            if sample_size_ref > 0:
                random_indices_ref = np.random.choice(num_all_normal_ref, size=sample_size_ref, replace=False)
                global_known_normal_samples_pca_for_cni = all_normal_samples_pca_ref[random_indices_ref]
                logger.info(f"[DEBUG GlobalRef] Global reference normal samples (90% of all normals in full data, PCA space) created. Shape: {global_known_normal_samples_pca_for_cni.shape}")
                logger.info(f"[DEBUG GlobalRef] Global ref data type: {type(global_known_normal_samples_pca_for_cni)}")
                logger.info(f"[DEBUG GlobalRef] Global ref contains NaN: {np.any(np.isnan(global_known_normal_samples_pca_for_cni)) if isinstance(global_known_normal_samples_pca_for_cni, np.ndarray) else 'N/A'}")
                
                # NaN 위치 상세 분석
                if isinstance(global_known_normal_samples_pca_for_cni, np.ndarray):
                    nan_count = np.isnan(global_known_normal_samples_pca_for_cni).sum()
                    total_elements = global_known_normal_samples_pca_for_cni.size
                    logger.info(f"[DEBUG GlobalRef] NaN count: {nan_count} out of {total_elements} elements ({nan_count/total_elements*100:.2f}%)")
                    
                    # NaN이 있는 행 찾기
                    nan_rows = np.any(np.isnan(global_known_normal_samples_pca_for_cni), axis=1)
                    nan_row_count = np.sum(nan_rows)
                    logger.info(f"[DEBUG GlobalRef] Rows with NaN: {nan_row_count} out of {global_known_normal_samples_pca_for_cni.shape[0]} rows ({nan_row_count/global_known_normal_samples_pca_for_cni.shape[0]*100:.2f}%)")
                    
                    # NaN이 있는 열 찾기
                    nan_cols = np.any(np.isnan(global_known_normal_samples_pca_for_cni), axis=0)
                    nan_col_count = np.sum(nan_cols)
                    logger.info(f"[DEBUG GlobalRef] Columns with NaN: {nan_col_count} out of {global_known_normal_samples_pca_for_cni.shape[1]} columns")
                    if nan_col_count > 0:
                        nan_col_indices = np.where(nan_cols)[0]
                        logger.info(f"[DEBUG GlobalRef] NaN column indices: {nan_col_indices}")
                
                logger.info(f"[DEBUG GlobalRef] Global ref sample values (first 3 rows, first 5 cols): {global_known_normal_samples_pca_for_cni[:3, :5] if global_known_normal_samples_pca_for_cni.size > 0 else 'Empty array'}")
            else:
                logger.warning("[WARN GlobalRef] Sample size for global reference normals is 0. No global reference created.")
                global_known_normal_samples_pca_for_cni = np.array([])

        elif num_all_normal_ref == 1:
            global_known_normal_samples_pca_for_cni = all_normal_samples_pca_ref
            logger.info("[DEBUG GlobalRef] Global reference normal samples (1 sample from full data, PCA space) created.")
            logger.info(f"[DEBUG GlobalRef] Global ref data type: {type(global_known_normal_samples_pca_for_cni)}")
            logger.info(f"[DEBUG GlobalRef] Global ref shape: {global_known_normal_samples_pca_for_cni.shape}")
        else:
            logger.warning("[WARN GlobalRef] No normal samples found in the full dataset to create global reference.")
            global_known_normal_samples_pca_for_cni = np.array([])
            if ref_X_pca.ndim == 2 and ref_X_pca.shape[1] > 0:
                 global_known_normal_samples_pca_for_cni = np.empty((0, ref_X_pca.shape[1]))

        del full_data_for_ref, full_data_for_ref_processed, ref_embedded_df, ref_group_mapped_df, ref_X_scaled, ref_X_pca, all_normal_samples_pca_ref
        logger.info("[DEBUG GlobalRef] Freed memory from temporary full data load.")

    except Exception as e:
        logger.error(f"[ERROR GlobalRef] Failed to create global reference normal samples: {e}. Proceeding without it.")
        global_known_normal_samples_pca_for_cni = None

    logger.info(f"Step 0 finished. Time: {time.time() - start_global_ref:.2f}s. Global ref shape: {global_known_normal_samples_pca_for_cni.shape if global_known_normal_samples_pca_for_cni is not None else 'None'}")
    return global_known_normal_samples_pca_for_cni

def run_clustering_with_cni(data, data_for_clustering, original_labels, algorithm, 
                           global_known_normal_samples_pca, autotune_enabled, 
                           num_processes_for_clustering_algo, threshold_candidates, 
                           file_type=None, file_number=None, stable_cluster_bounds=None):
    """Run clustering with CNI threshold optimization (based on Data_Labeling.py chunking logic)"""
    
    # Define chunking parameters
    chunk_size = 30000
    num_samples = data_for_clustering.shape[0]
    num_chunks = math.ceil(num_samples / chunk_size)
    logger.info(f"Total samples: {num_samples}, Chunk size: {chunk_size}, Number of chunks: {num_chunks}")

    # Check if threshold optimization results already exist
    optimization_results = None
    if file_type and file_number:
        optimization_results = load_threshold_optimization_results(file_type, file_number, algorithm)
        if optimization_results:
            logger.info(f"Found existing threshold optimization results for {algorithm}: optimal_threshold={optimization_results['optimal_threshold']}")
            logger.info(f"Results from: {optimization_results['timestamp']}")
            # Use existing results and skip to Phase 3
            optimal_cni_threshold = optimization_results['optimal_threshold']
            return run_full_dataset_with_optimal_threshold(
                data, data_for_clustering, original_labels, algorithm,
                global_known_normal_samples_pca, autotune_enabled,
                num_processes_for_clustering_algo, optimal_cni_threshold
            )

    # Early termination parameters (define first)
    min_chunks_for_statistics = 5  # Minimum chunks needed for reliable statistics
    max_chunks_for_early_termination = 6  # Maximum chunks to process before checking for early termination (reduced for faster processing)
    convergence_threshold = 0.01  # Jaccard score convergence threshold

    # Load existing threshold optimization progress if available
    threshold_jaccard_scores_across_chunks = {thresh: [] for thresh in threshold_candidates}
    threshold_cluster_counts_across_chunks = {thresh: [] for thresh in threshold_candidates}
    completed_optimization_chunks = set()
    
    if file_type and file_number:
        # Load existing threshold optimization progress
        existing_jaccard_scores, existing_cluster_counts, completed_optimization_chunks = load_threshold_optimization_progress(file_type, file_number, algorithm)
        
        if existing_jaccard_scores or existing_cluster_counts:
            logger.info(f"Loaded existing threshold optimization progress: {len(completed_optimization_chunks)} chunks completed")
            threshold_jaccard_scores_across_chunks.update(existing_jaccard_scores)
            threshold_cluster_counts_across_chunks.update(existing_cluster_counts)
        else:
            logger.info("No existing threshold optimization progress found")

    # Select random chunks for threshold optimization
    selected_chunks_for_optimization = select_random_chunks_for_optimization(num_chunks, max_chunks_for_early_termination)
    logger.info(f"Selected {len(selected_chunks_for_optimization)} random chunks for threshold optimization: {selected_chunks_for_optimization}")

    # Store temporary labels for reuse
    chunk_threshold_labels_temp_storage = {}

    # Phase 1: Iterate through chunks and thresholds to collect Jaccard scores
    logger.info("Phase 1: Collecting Jaccard scores and temporary labels for each chunk and threshold...")
    
    # Pre-load clustering functions to avoid repeated imports
    try:
        choose_clustering_algorithm, choose_clustering_algorithm_Non_optimization = get_autotune_imports()
    except Exception as e:
        logger.error(f"Error loading clustering functions: {e}")
        raise e
    
    processed_chunks = 0
    for i in selected_chunks_for_optimization:
        # Skip chunks that are already completed for threshold optimization
        if i in completed_optimization_chunks:
            logger.info(f"  Skipping Chunk {i+1}/{num_chunks} (threshold optimization already completed)")
            continue
        start_chunk_time = time.time()
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_samples)
        
        current_chunk_data_np = data_for_clustering[start_idx:end_idx]
        current_chunk_original_labels_np = original_labels[start_idx:end_idx]

        logger.info(f"  Processing Chunk {i+1}/{num_chunks} (Samples {start_idx}-{end_idx-1}), Shape: {current_chunk_data_np.shape}")
        
        for current_threshold_in_chunk_loop in threshold_candidates:
            # Optimize logging frequency - only log every 2nd threshold
            if len(threshold_candidates) <= 2 or current_threshold_in_chunk_loop == threshold_candidates[0] or current_threshold_in_chunk_loop == threshold_candidates[-1]:
                logger.info(f"    Chunk {i+1}, Testing CNI threshold: {current_threshold_in_chunk_loop}")
            gmm_type_for_this_run = None 
            
            try:
                if autotune_enabled: 
                    temp_chunk_clustering_result, gmm_type_for_this_run = choose_clustering_algorithm(
                        data, current_chunk_data_np, current_chunk_original_labels_np, algorithm, 
                        global_known_normal_samples_pca=global_known_normal_samples_pca,
                        threshold_value=current_threshold_in_chunk_loop,
                        num_processes_for_algo=num_processes_for_clustering_algo
                    )
                else: 
                    temp_chunk_clustering_result = choose_clustering_algorithm_Non_optimization(
                        data, current_chunk_data_np, current_chunk_original_labels_np, algorithm, 
                        global_known_normal_samples_pca=global_known_normal_samples_pca,
                        threshold_value=current_threshold_in_chunk_loop,
                        num_processes_for_algo=num_processes_for_clustering_algo
                    )
            except NameError as e:
                if "choose_clustering_algorithm" in str(e):
                    logger.error(f"choose_clustering_algorithm not defined. autotune_enabled={autotune_enabled}")
                    logger.error(f"Available functions: choose_clustering_algorithm={choose_clustering_algorithm is not None}, choose_clustering_algorithm_Non_optimization={choose_clustering_algorithm_Non_optimization is not None}")
                    raise e
                else:
                    raise e

            if isinstance(temp_chunk_clustering_result, dict) and 'Cluster_labeling' in temp_chunk_clustering_result and temp_chunk_clustering_result['Cluster_labeling'] is not None:
                y_pred_chunk_current_thresh = temp_chunk_clustering_result['Cluster_labeling']

                if y_pred_chunk_current_thresh.size == current_chunk_original_labels_np.size and y_pred_chunk_current_thresh.size > 0:
                    chunk_threshold_labels_temp_storage[(i, current_threshold_in_chunk_loop)] = y_pred_chunk_current_thresh
                    chunk_metrics = evaluate_clustering_wos(current_chunk_original_labels_np, y_pred_chunk_current_thresh)
                    
                    micro_metrics_dict = chunk_metrics.get('average=micro', {})
                    current_jaccard_micro_chunk = micro_metrics_dict.get('jaccard', -1.0)

                    logger.info(f"      INFO: Thresh {current_threshold_in_chunk_loop} - Calculated Jaccard (micro): {current_jaccard_micro_chunk}")

                    if current_jaccard_micro_chunk != -1.0:
                         threshold_jaccard_scores_across_chunks[current_threshold_in_chunk_loop].append(current_jaccard_micro_chunk)
                         logger.info(f"        DEBUG: Thresh {current_threshold_in_chunk_loop} - Stored Jaccard. Current list for this thresh: {threshold_jaccard_scores_across_chunks[current_threshold_in_chunk_loop]}")
                         
                         # Store cluster count for convergence analysis
                         num_clusters = len(np.unique(y_pred_chunk_current_thresh))
                         threshold_cluster_counts_across_chunks[current_threshold_in_chunk_loop].append(num_clusters)
                         logger.info(f"        DEBUG: Thresh {current_threshold_in_chunk_loop} - Stored cluster count: {num_clusters}")
                         
                         # Save progress immediately after each threshold processing
                         if file_type and file_number:
                             save_threshold_optimization_progress(
                                 file_type, file_number, algorithm, i, 
                                 current_threshold_in_chunk_loop, current_jaccard_micro_chunk, num_clusters
                             )
                    else:
                        logger.info(f"        DEBUG: Thresh {current_threshold_in_chunk_loop} - Jaccard score is -1.0, not storing.")
                else: 
                    logger.warning(f"      WARN: Thresh {current_threshold_in_chunk_loop} - Label size mismatch or empty labels. No Jaccard calculated or stored for this run.")
            else: 
                logger.warning(f"      WARN: Thresh {current_threshold_in_chunk_loop} - 'Cluster_labeling' missing, None, or result not a dict.")
        
        end_chunk_time = time.time()
        logger.info(f"  Chunk {i+1} (threshold sweep) processed in {end_chunk_time - start_chunk_time:.2f}s.")
        
        # Mark this chunk as completed for threshold optimization
        completed_optimization_chunks.add(i)
        
        # Save general chunk progress
        if file_type and file_number:
            chunk_jaccard_scores = {}
            for thresh, scores in threshold_jaccard_scores_across_chunks.items():
                if scores:
                    chunk_jaccard_scores[thresh] = scores[-1]  # Get the last score for this chunk
            save_progress(file_type, file_number, algorithm, i, 'completed', chunk_jaccard_scores)
        
        processed_chunks += 1
        
        # Clean up memory after each chunk
        cleanup_memory()
        
        # Check for early termination after processing enough chunks
        if processed_chunks >= min_chunks_for_statistics and processed_chunks >= max_chunks_for_early_termination:
            logger.info(f"  Checking for early termination after {processed_chunks} chunks...")
            
            # Calculate current optimal threshold based on processed chunks
            current_optimal_threshold = 0.3
            current_best_jaccard = -1.0
            threshold_stability = {}
            
            for thresh_val, jaccard_list in threshold_jaccard_scores_across_chunks.items():
                if len(jaccard_list) >= min_chunks_for_statistics:
                    scores_np = np.array(jaccard_list)
                    q1 = np.percentile(scores_np, 25)
                    q3 = np.percentile(scores_np, 75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    filtered_scores = scores_np[(scores_np >= lower_bound) & (scores_np <= upper_bound)]
                    if filtered_scores.size > 0:
                        robust_avg = np.mean(filtered_scores)
                        robust_std = np.std(filtered_scores)
                        threshold_stability[thresh_val] = {
                            'mean': robust_avg,
                            'std': robust_std,
                            'cv': robust_std / robust_avg if robust_avg > 0 else float('inf')  # Coefficient of Variation
                        }
                        
                        if robust_avg > current_best_jaccard:
                            current_best_jaccard = robust_avg
                            current_optimal_threshold = thresh_val
            
            # Check for convergence: if the optimal threshold is stable
            if len(threshold_stability) >= len(threshold_candidates):
                optimal_cv = threshold_stability[current_optimal_threshold]['cv']
                logger.info(f"  Current optimal threshold: {current_optimal_threshold} (Jaccard: {current_best_jaccard:.4f}, CV: {optimal_cv:.4f})")
                
                # Check cluster count convergence for optimal threshold
                optimal_cluster_counts = threshold_cluster_counts_across_chunks.get(current_optimal_threshold, [])
                if len(optimal_cluster_counts) >= min_chunks_for_statistics:
                    cluster_counts_np = np.array(optimal_cluster_counts)
                    cluster_cv = np.std(cluster_counts_np) / np.mean(cluster_counts_np) if np.mean(cluster_counts_np) > 0 else float('inf')
                    cluster_range = np.max(cluster_counts_np) - np.min(cluster_counts_np)
                    
                    logger.info(f"  Cluster count convergence: mean={np.mean(cluster_counts_np):.1f}, std={np.std(cluster_counts_np):.1f}, CV={cluster_cv:.4f}, range={cluster_range}")
                    
                    # Check if cluster counts are stable
                    cluster_stable = cluster_cv < 0.1 and cluster_range <= 5  # CV < 10% and range <= 5
                    
                    if cluster_stable:
                        logger.info(f"  Cluster counts are stable! Will use constrained range for remaining chunks.")
                        # Store stable cluster range for later use
                        stable_cluster_mean = int(np.mean(cluster_counts_np))
                        stable_cluster_range = max(3, int(np.std(cluster_counts_np) * 2))  # ±2σ range, minimum 3
                        stable_cluster_bounds = (max(2, stable_cluster_mean - stable_cluster_range), 
                                               stable_cluster_mean + stable_cluster_range)
                        logger.info(f"  Stable cluster bounds: {stable_cluster_bounds}")
                    else:
                        stable_cluster_bounds = None
                        logger.info(f"  Cluster counts still unstable. Will use full range for remaining chunks.")
                else:
                    stable_cluster_bounds = None
                    logger.info(f"  Not enough cluster count data yet ({len(optimal_cluster_counts)} < {min_chunks_for_statistics})")
                
                # Check if all thresholds have low coefficient of variation (stable)
                all_stable = all(stats['cv'] < 0.05 for stats in threshold_stability.values())  # CV < 5%
                
                if all_stable:
                    logger.info(f"  All thresholds show stable results (CV < 5%). Early termination.")
                    # Store stable cluster bounds for Phase 3
                    if 'stable_cluster_bounds' in locals() and stable_cluster_bounds is not None:
                        logger.info(f"  Will use cluster bounds {stable_cluster_bounds} for remaining chunks.")
                    break
                else:
                    logger.info(f"  Some thresholds still unstable. Continuing...")
            
            # Fallback: if we have enough data, proceed anyway
            total_scores = sum(len(scores) for scores in threshold_jaccard_scores_across_chunks.values())
            if total_scores >= min_chunks_for_statistics * len(threshold_candidates):
                logger.info(f"  Sufficient data collected ({total_scores} scores). Proceeding to threshold optimization...")
                break
            else:
                logger.info(f"  Not enough data yet ({total_scores} scores). Continuing...")
        
        # Also check if we have processed enough chunks for basic statistics
        elif processed_chunks >= min_chunks_for_statistics:
            total_scores = sum(len(scores) for scores in threshold_jaccard_scores_across_chunks.values())
            if total_scores >= min_chunks_for_statistics * len(threshold_candidates):
                logger.info(f"  Sufficient data collected after {processed_chunks} chunks ({total_scores} scores). Checking for convergence...")
                
                # Advanced convergence check: trend analysis and stability
                if processed_chunks >= 10:  # Need at least 10 chunks for convergence check
                    recent_scores = {}
                    convergence_metrics = {}
                    
                    for thresh_val, jaccard_list in threshold_jaccard_scores_across_chunks.items():
                        if len(jaccard_list) >= 8:  # Need at least 8 recent scores for trend analysis
                            recent_scores[thresh_val] = jaccard_list[-8:]  # Last 8 scores
                            
                            # Calculate trend (slope of linear regression)
                            x = np.arange(len(recent_scores[thresh_val]))
                            y = np.array(recent_scores[thresh_val])
                            slope = np.polyfit(x, y, 1)[0]  # Linear trend
                            
                            # Calculate stability metrics
                            recent_std = np.std(recent_scores[thresh_val])
                            recent_cv = recent_std / np.mean(recent_scores[thresh_val]) if np.mean(recent_scores[thresh_val]) > 0 else float('inf')
                            
                            convergence_metrics[thresh_val] = {
                                'slope': slope,
                                'std': recent_std,
                                'cv': recent_cv,
                                'trend_stable': abs(slope) < 0.001  # Very small trend
                            }
                    
                    if len(convergence_metrics) == len(threshold_candidates):
                        # Check convergence conditions
                        all_trends_stable = all(metrics['trend_stable'] for metrics in convergence_metrics.values())
                        all_cv_low = all(metrics['cv'] < 0.03 for metrics in convergence_metrics.values())  # CV < 3%
                        
                        logger.info(f"  Convergence check: trends_stable={all_trends_stable}, cv_low={all_cv_low}")
                        for thresh_val, metrics in convergence_metrics.items():
                            logger.info(f"    Threshold {thresh_val}: slope={metrics['slope']:.6f}, CV={metrics['cv']:.4f}")
                        
                        if all_trends_stable and all_cv_low:
                            logger.info(f"  Strong convergence detected after {processed_chunks} chunks. Early termination.")
                            break
                        elif all_cv_low:  # At least stable variance
                            logger.info(f"  Moderate convergence detected after {processed_chunks} chunks. Early termination.")
                            break

    # Phase 2: Determine Optimal CNI Threshold (IQR Outlier Removal + Mean)
    logger.info("Phase 2: Determining Optimal CNI Threshold...")
    logger.info(f"  Processed {processed_chunks} chunks out of {num_chunks} total chunks ({processed_chunks/num_chunks*100:.1f}%)")
    optimal_cni_threshold = 0.3  # Default if all else fails or no scores
    best_robust_average_jaccard = -1.0
    stable_cluster_bounds = None  # Initialize stable cluster bounds

    for thresh_val, jaccard_list in threshold_jaccard_scores_across_chunks.items():
        if not jaccard_list:
            logger.info(f"  Threshold {thresh_val}: No Jaccard scores recorded.")
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
            logger.info(f"  Threshold {thresh_val}: Robust Avg Jaccard (micro) = {robust_average_jaccard:.4f} (from {filtered_scores.size}/{scores_np.size} scores after outlier removal)")
            if robust_average_jaccard > best_robust_average_jaccard:
                best_robust_average_jaccard = robust_average_jaccard
                optimal_cni_threshold = thresh_val
        else:
            # If all scores were outliers, fall back to mean of original scores
            original_mean = np.mean(scores_np)
            logger.info(f"  Threshold {thresh_val}: All scores considered outliers. Original mean Jaccard = {original_mean:.4f} (from {scores_np.size} scores)")
            if original_mean > best_robust_average_jaccard:
                 best_robust_average_jaccard = original_mean
                 optimal_cni_threshold = thresh_val
                 logger.info(f"    (Using original mean as it's currently the best overall: {original_mean:.4f}) ")

    logger.info(f"Optimal CNI Threshold selected: {optimal_cni_threshold} with best robust average Jaccard (micro): {best_robust_average_jaccard:.4f}")
    
    # Save threshold optimization results
    if file_type and file_number:
        save_threshold_optimization_results(
            file_type, file_number, algorithm, optimal_cni_threshold,
            threshold_jaccard_scores_across_chunks, threshold_cluster_counts_across_chunks,
            selected_chunks_for_optimization
        )
        logger.info(f"Saved threshold optimization results for {algorithm}")
        
        # Clean up progress file since optimization is complete
        progress_file = get_threshold_optimization_progress_file_path(file_type, file_number, algorithm)
        if os.path.exists(progress_file):
            try:
                os.remove(progress_file)
                logger.info(f"Cleaned up threshold optimization progress file: {progress_file}")
            except Exception as e:
                logger.warning(f"Could not remove progress file {progress_file}: {e}")

    # Phase 3: Re-process chunks with Optimal CNI Threshold to get final labels
    logger.info(f"Phase 3: Assembling final predictions by re-processing chunks with optimal_cni_threshold = {optimal_cni_threshold}...")
    final_predict_results_list = []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_samples)
        current_chunk_data_np = data_for_clustering[start_idx:end_idx]
        current_chunk_original_labels_np = original_labels[start_idx:end_idx]
        
        final_gmm_type_for_chunk = None
        final_params_for_chunk = {}
        labels_for_this_chunk_optimal = None

        # Check if we already processed this chunk with the optimal threshold
        chunk_key = (i, optimal_cni_threshold)
        if chunk_key in chunk_threshold_labels_temp_storage:
            # Reuse existing labels from Phase 1
            labels_for_this_chunk_optimal = chunk_threshold_labels_temp_storage[chunk_key]
            logger.info(f"  Chunk {i+1}: Reusing labels from Phase 1 (optimal threshold {optimal_cni_threshold})")
        else:
            # Process this chunk for the first time with optimal threshold
            logger.info(f"  Chunk {i+1}: Processing with optimal threshold {optimal_cni_threshold} for the first time")
            
            # Use stable cluster bounds if available
            if stable_cluster_bounds is not None:
                logger.info(f"  Chunk {i+1}: Using stable cluster bounds {stable_cluster_bounds} for faster processing")
                # Modify the clustering algorithm to use constrained cluster range
                # This would require modifying the choose_clustering_algorithm function
                # For now, we'll log the information
                
            if autotune_enabled:
                final_chunk_clustering_result, final_gmm_type_for_chunk = choose_clustering_algorithm(
                    data, current_chunk_data_np, current_chunk_original_labels_np, algorithm,
                    global_known_normal_samples_pca=global_known_normal_samples_pca, threshold_value=optimal_cni_threshold,
                    num_processes_for_algo=num_processes_for_clustering_algo
                )
            else:
                final_chunk_clustering_result = choose_clustering_algorithm_Non_optimization(
                    data, current_chunk_data_np, current_chunk_original_labels_np, algorithm,
                    global_known_normal_samples_pca=global_known_normal_samples_pca, threshold_value=optimal_cni_threshold,
                    num_processes_for_clustering_algo=num_processes_for_clustering_algo
                )
        
            if 'Cluster_labeling' in final_chunk_clustering_result and final_chunk_clustering_result['Cluster_labeling'] is not None:
                labels_for_this_chunk_optimal = final_chunk_clustering_result['Cluster_labeling']
                final_params_for_chunk = final_chunk_clustering_result.get('Best_parameter_dict', {})
            else:
                logger.error(f"    [ERROR] Chunk {i+1} processing with optimal threshold failed to produce labels. Using empty array for this chunk.")
                labels_for_this_chunk_optimal = np.array([]) 

        final_predict_results_list.append(labels_for_this_chunk_optimal)

    if final_predict_results_list and not all(arr.size == 0 for arr in final_predict_results_list):
        valid_results_to_concat = [arr for arr in final_predict_results_list if arr.size > 0]
        if valid_results_to_concat:
            final_predict_results = np.concatenate(valid_results_to_concat)
            if len(final_predict_results) != num_samples:
                 logger.warning(f"[WARNING Phase 3] Length of concatenated final labels ({len(final_predict_results)}) does not match total samples ({num_samples}) after potentially excluding failed chunks.")
        else:
            logger.error("[ERROR Phase 3] All chunks failed to produce labels in Phase 3. final_predict_results will be empty.")
            final_predict_results = np.array([])
    else:
        logger.error("[ERROR Phase 3] No valid prediction results to concatenate after Phase 3.")
        final_predict_results = np.array([])

    return final_predict_results, optimal_cni_threshold, stable_cluster_bounds

def evaluate_thresholds_for_best_algorithm(data, data_for_clustering, original_labels, best_algorithm,
                                         global_known_normal_samples_pca, autotune_enabled,
                                         num_processes_for_clustering_algo, file_type, file_number):
    """Evaluate multiple thresholds (0.3, 0.4, 0.5, 0.6) for the best algorithm"""
    logger.info(f"Evaluating thresholds for best algorithm: {best_algorithm}")
    
    threshold_candidates = [0.3, 0.4, 0.5, 0.6]
    threshold_results = {}
    
    for threshold in threshold_candidates:
        logger.info(f"Testing threshold: {threshold}")
        try:
            cluster_labels, _, _ = run_clustering_with_cni(
                data, data_for_clustering, original_labels, best_algorithm,
                global_known_normal_samples_pca, autotune_enabled,
                num_processes_for_clustering_algo, [threshold],  # Only test this threshold
                file_type, file_number  # Pass for progress tracking
            )
            
            if cluster_labels is not None and len(cluster_labels) == len(original_labels):
                metrics = evaluate_clustering_wos(original_labels, cluster_labels)
                micro_metrics = metrics.get('average=micro', {})
                jaccard_score = micro_metrics.get('jaccard', -1.0)
                
                threshold_results[threshold] = {
                    'jaccard_score': jaccard_score,
                    'cluster_labels': cluster_labels,
                    'metrics': metrics
                }
                
                logger.info(f"  Threshold {threshold}: Jaccard Score = {jaccard_score:.4f}")
            else:
                logger.warning(f"  Threshold {threshold}: Invalid cluster labels")
                
        except Exception as e:
            logger.error(f"  Threshold {threshold}: Error - {str(e)}")
            continue
    
    return threshold_results

def ensure_dataset_ex_directory():
    """Ensure Dataset_ex directory structure exists (similar to Dataset_Paral)"""
    # Create main Dataset_ex directory
    dataset_ex_dir = "../Dataset_ex"
    if not os.path.exists(dataset_ex_dir):
        os.makedirs(dataset_ex_dir)
        logger.info(f"Created main directory: {dataset_ex_dir}")
    
    # Create subdirectories similar to Dataset_Paral structure
    subdirs = [
        "Data_Label",  # Changed from save_dataset to Data_Label
        "time_log/virtual_labeling",
        "time_log/virtual_labeling_ex",  # For our specific time logs
        "signature",
        "mapping_info",  # For storing mapping information
        "progress_tracking"  # For tracking chunk processing progress
    ]
    
    for subdir in subdirs:
        full_path = os.path.join(dataset_ex_dir, subdir)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            logger.info(f"Created subdirectory: {full_path}")
    
    return dataset_ex_dir

def reset_intermediate_files(file_type, file_number):
    """Delete all intermediate progress files for a fresh start"""
    import glob
    
    # Patterns for different types of progress files
    patterns_to_delete = [
        # Elbow method progress files
        f"../Dataset_ex/progress_tracking/elbow_*_*_progress.csv",
        # Grid search progress files  
        f"../Dataset_ex/progress_tracking/grid_search_*_*_progress.csv",
        # Jaccard Elbow method progress files
        f"../Dataset_ex/progress_tracking/jaccard_elbow_*_*_progress.csv",
        # Threshold optimization files
        f"../Dataset_ex/progress_tracking/*_threshold_optimization.csv",
        # Algorithm results files
        f"../Dataset_ex/progress_tracking/*_results.csv",
        # Summary files
        f"../Dataset_ex/progress_tracking/*_summary.txt",
        # Mapping files (to force regeneration with new feature configuration)
        f"../Dataset_ex/mapping_info/*{file_type}*_Interval_inverse_mapping.pkl",
        f"../Dataset_ex/mapping_info/*{file_type}*_Interval_scalers.pkl"
    ]
    
    deleted_files = []
    
    for pattern in patterns_to_delete:
        matching_files = glob.glob(pattern)
        for file_path in matching_files:
            try:
                os.remove(file_path)
                deleted_files.append(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Warning: Could not delete {file_path}: {e}")
    
    if deleted_files:
        print(f"Successfully deleted {len(deleted_files)} intermediate files")
    else:
        print("No intermediate files found to delete")

def get_progress_file_path(file_type, file_number, algorithm):
    """Get the progress tracking file path for a specific dataset and algorithm"""
    return f"../Dataset_ex/progress_tracking/{file_type}_{file_number}_{algorithm}_progress.csv"

def load_progress(file_type, file_number, algorithm):
    """Load existing progress from CSV file"""
    progress_file = get_progress_file_path(file_type, file_number, algorithm)
    completed_chunks = set()
    
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    if len(row) >= 2:
                        chunk_idx = int(row[0])
                        status = row[1]
                        if status == 'completed':
                            completed_chunks.add(chunk_idx)
        except Exception as e:
            print(f"Warning: Could not load progress file {progress_file}: {e}")
    
    return completed_chunks

def save_progress(file_type, file_number, algorithm, chunk_idx, status, jaccard_scores=None):
    """Save progress to CSV file"""
    progress_file = get_progress_file_path(file_type, file_number, algorithm)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)
    
    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(progress_file)
    
    try:
        with open(progress_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header if file is new
            if not file_exists:
                writer.writerow(['chunk_idx', 'status', 'timestamp', 'jaccard_scores'])
            
            # Write progress data
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            jaccard_str = str(jaccard_scores) if jaccard_scores else ''
            writer.writerow([chunk_idx, status, timestamp, jaccard_str])
            
    except Exception as e:
        print(f"Warning: Could not save progress to {progress_file}: {e}")

def get_next_chunk_to_process(completed_chunks, total_chunks):
    """Get the next chunk index that needs to be processed"""
    for i in range(total_chunks):
        if i not in completed_chunks:
            return i
    return None  # All chunks completed

def get_threshold_optimization_file_path(file_type, file_number, algorithm):
    """Get the threshold optimization results file path"""
    return f"../Dataset_ex/progress_tracking/{file_type}_{file_number}_{algorithm}_threshold_optimization.csv"

def get_threshold_optimization_progress_file_path(file_type, file_number, algorithm):
    """Get the threshold optimization progress file path"""
    return f"../Dataset_ex/progress_tracking/{file_type}_{file_number}_{algorithm}_threshold_progress.csv"

def load_threshold_optimization_results(file_type, file_number, algorithm):
    """Load existing threshold optimization results"""
    optimization_file = get_threshold_optimization_file_path(file_type, file_number, algorithm)
    
    if os.path.exists(optimization_file):
        try:
            with open(optimization_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header and len(header) >= 6:
                    row = next(reader, None)
                    if row and len(row) >= 6:
                        return {
                            'optimal_threshold': float(row[1]),
                            'jaccard_scores': eval(row[2]) if row[2] else {},
                            'cluster_counts': eval(row[3]) if row[3] else {},
                            'selected_chunks': eval(row[4]) if row[4] else [],
                            'timestamp': row[5]
                        }
        except Exception as e:
            print(f"Warning: Could not load threshold optimization results from {optimization_file}: {e}")
    
    return None

def load_threshold_optimization_progress(file_type, file_number, algorithm):
    """Load existing threshold optimization progress"""
    progress_file = get_threshold_optimization_progress_file_path(file_type, file_number, algorithm)
    
    threshold_jaccard_scores = {}
    threshold_cluster_counts = {}
    completed_chunks = set()
    
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header and len(header) >= 5:
                    for row in reader:
                        if len(row) >= 5:
                            chunk_idx = int(row[0])
                            threshold = float(row[1])
                            jaccard_score = float(row[2])
                            cluster_count = int(row[3])
                            
                            # Initialize threshold dictionaries if needed
                            if threshold not in threshold_jaccard_scores:
                                threshold_jaccard_scores[threshold] = []
                            if threshold not in threshold_cluster_counts:
                                threshold_cluster_counts[threshold] = []
                            
                            # Add scores
                            threshold_jaccard_scores[threshold].append(jaccard_score)
                            threshold_cluster_counts[threshold].append(cluster_count)
                            completed_chunks.add(chunk_idx)
                            
        except Exception as e:
            print(f"Warning: Could not load threshold optimization progress from {progress_file}: {e}")
    
    return threshold_jaccard_scores, threshold_cluster_counts, completed_chunks

def save_threshold_optimization_progress(file_type, file_number, algorithm, chunk_idx, threshold, jaccard_score, cluster_count):
    """Save threshold optimization progress for a single chunk-threshold combination"""
    progress_file = get_threshold_optimization_progress_file_path(file_type, file_number, algorithm)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)
    
    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(progress_file)
    
    try:
        with open(progress_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header if file is new
            if not file_exists:
                writer.writerow(['chunk_idx', 'threshold', 'jaccard_score', 'cluster_count', 'timestamp'])
            
            # Write progress data
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            writer.writerow([chunk_idx, threshold, jaccard_score, cluster_count, timestamp])
            
    except Exception as e:
        print(f"Warning: Could not save threshold optimization progress to {progress_file}: {e}")

def save_threshold_optimization_results(file_type, file_number, algorithm, optimal_threshold, 
                                      jaccard_scores, cluster_counts, selected_chunks):
    """Save threshold optimization results to CSV file"""
    optimization_file = get_threshold_optimization_file_path(file_type, file_number, algorithm)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(optimization_file), exist_ok=True)
    
    try:
        with open(optimization_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['algorithm', 'optimal_threshold', 'jaccard_scores', 'cluster_counts', 'selected_chunks', 'timestamp'])
            
            # Write results
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            writer.writerow([
                algorithm,
                optimal_threshold,
                str(jaccard_scores),
                str(cluster_counts),
                str(selected_chunks),
                timestamp
            ])
            
    except Exception as e:
        print(f"Warning: Could not save threshold optimization results to {optimization_file}: {e}")

def select_random_chunks_for_optimization(total_chunks, num_chunks_to_select):
    """Select random chunks for threshold optimization"""
    if num_chunks_to_select >= total_chunks:
        return list(range(total_chunks))
    
    selected_chunks = sorted(np.random.choice(total_chunks, num_chunks_to_select, replace=False))
    return selected_chunks

def run_full_dataset_with_optimal_threshold(data, data_for_clustering, original_labels, algorithm,
                                           global_known_normal_samples_pca, autotune_enabled,
                                           num_processes_for_clustering_algo, optimal_cni_threshold):
    """Run full dataset clustering with pre-determined optimal threshold"""
    logger.info(f"Running full dataset clustering with optimal threshold: {optimal_cni_threshold}")
    
    chunk_size = 30000
    num_samples = data_for_clustering.shape[0]
    num_chunks = math.ceil(num_samples / chunk_size)
    logger.info(f"Total samples: {num_samples}, Chunk size: {chunk_size}, Number of chunks: {num_chunks}")
    
    final_predict_results_list = []
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_samples)
        current_chunk_data_np = data_for_clustering[start_idx:end_idx]
        current_chunk_original_labels_np = original_labels[start_idx:end_idx]
        
        logger.info(f"  Processing Chunk {i+1}/{num_chunks} with optimal threshold {optimal_cni_threshold}")
        
        if autotune_enabled:
            chunk_clustering_result, _ = choose_clustering_algorithm(
                data, current_chunk_data_np, current_chunk_original_labels_np, algorithm,
                global_known_normal_samples_pca=global_known_normal_samples_pca, 
                threshold_value=optimal_cni_threshold,
                num_processes_for_algo=num_processes_for_clustering_algo
            )
        else:
            chunk_clustering_result = choose_clustering_algorithm_Non_optimization(
                data, current_chunk_data_np, current_chunk_original_labels_np, algorithm,
                global_known_normal_samples_pca=global_known_normal_samples_pca, 
                threshold_value=optimal_cni_threshold,
                num_processes_for_clustering_algo=num_processes_for_clustering_algo
            )
        
        if 'Cluster_labeling' in chunk_clustering_result and chunk_clustering_result['Cluster_labeling'] is not None:
            labels_for_this_chunk = chunk_clustering_result['Cluster_labeling']
        else:
            logger.error(f"    [ERROR] Chunk {i+1} processing failed. Using empty array for this chunk.")
            labels_for_this_chunk = np.array([])
        
        final_predict_results_list.append(labels_for_this_chunk)
    
    if final_predict_results_list and not all(arr.size == 0 for arr in final_predict_results_list):
        valid_results_to_concat = [arr for arr in final_predict_results_list if arr.size > 0]
        if valid_results_to_concat:
            final_predict_results = np.concatenate(valid_results_to_concat)
            if len(final_predict_results) != num_samples:
                logger.warning(f"[WARNING] Length of concatenated final labels ({len(final_predict_results)}) does not match total samples ({num_samples})")
        else:
            logger.error("[ERROR] All chunks failed to produce labels.")
            final_predict_results = np.array([])
    else:
        logger.error("[ERROR] No valid prediction results to concatenate.")
        final_predict_results = np.array([])
    
    return final_predict_results, optimal_cni_threshold, None



def main():
    parser = argparse.ArgumentParser(description='Best Clustering Algorithm Selector - Parallel Version')
    parser.add_argument('--file_type', type=str, default="MiraiBotnet", help='Data file type')
    parser.add_argument('--file_number', type=int, default=1, help='File number')
    parser.add_argument('--train_test', type=int, default=0, help='Train = 0, test = 1')
    parser.add_argument('--heterogeneous', type=str, default="Interval_inverse", help='Heterogeneous method')
    parser.add_argument('--autotune', type=str, default="y", help='Enable autotuning (y/n)')
    parser.add_argument('--max_workers', type=int, default=0, help='Max parallel workers (0 = auto)')
    parser.add_argument('--eval_clustering_silhouette', type=str, default="n", help="Evaluate with silhouette score (y/n)")
    parser.add_argument('--reset', action='store_true', help='Reset all intermediate progress files and start fresh')
    
    args = parser.parse_args()
    
    # Handle --reset flag: Delete all intermediate progress files
    if args.reset:
        reset_intermediate_files(args.file_type, args.file_number)
        print("All intermediate progress files have been deleted. Starting fresh...")
    
    total_start_time = time.time()
    timing_info = {}  # For step-by-step time recording
    
    # Check system resources
    cpu_count, memory, disk = get_system_resources()
    
    # Set number of processes for internal algorithms - use all available cores
    if args.max_workers > 0:
        num_processes = min(args.max_workers, cpu_count)
    else:
        num_processes = cpu_count  # Use all available cores for each algorithm
    
    logger.info(f"Using {num_processes} processes (all cores) for each algorithm sequentially")
    
    # Define candidate threshold values for CNI (only 0.4 for algorithm selection)
    algorithm_selection_threshold = [0.4]  # Only use 0.4 for algorithm selection
    final_evaluation_thresholds = [0.3, 0.4, 0.5, 0.6]  # For final evaluation of best algorithm
    
    # Parse arguments
    file_type = args.file_type
    file_number = args.file_number
    train_test = args.train_test
    heterogeneous_method = args.heterogeneous
    eval_clustering_silhouette_flag = args.eval_clustering_silhouette.lower() == 'y'
    autotune_enabled = args.autotune.lower() == 'y'
    
    # Ensure Dataset_ex directory exists
    dataset_ex_dir = ensure_dataset_ex_directory()
    
    # 0. Create Global Reference Normal Samples PCA (for CNI function)
    global_known_normal_samples_pca_for_cni = create_global_reference_normal_samples(
        file_type, file_number, heterogeneous_method
    )
    
    # 1. Load data
    logger.info("Step 1: Loading data...")
    start_time = time.time()
    
    file_path, file_number = file_path_line_nonnumber(file_type, file_number)
    
    '''
    if file_type in ['DARPA98', 'DARPA', 'NSL-KDD', 'NSL_KDD', 'CICModbus23', 'CICModbus', 'MitM', 'Kitsune', 'ARP']:
        cut_type = 'random'
    else:
        cut_type = 'all'
    '''
    cut_type = 'all'
    
    data = file_cut(file_type, file_path, cut_type)
    data.columns = data.columns.str.strip()
    
    logger.info(f"Data loaded. Shape: {data.shape}")
    timing_info['1_load_data'] = time.time() - start_time
    logger.info(f"Step 1 completed in {timing_info['1_load_data']:.2f}s")
    
    # 2. Process labels
    logger.info("Step 2: Processing labels...")
    start_time = time.time()
    
    if file_type in ['MiraiBotnet', 'NSL-KDD', 'NSL_KDD']:
        data['label'], _ = anomal_judgment_nonlabel(file_type, data)
    elif file_type == 'netML':
        data['label'] = data['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
    elif file_type == 'DARPA98':
        data['label'] = data['Class'].apply(lambda x: 0 if str(x).strip() == '-' else 1)
    elif file_type in ['CICIDS2017', 'CICIDS']:
        if 'Label' in data.columns:
            data['label'] = data['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
        else:
            logger.error("'Label' column not found in data")
            data['label'] = 0
    elif file_type in ['CICModbus23', 'CICModbus']:
        data['label'] = data['Attack'].apply(lambda x: 0 if str(x).strip() == 'Baseline Replay: In position' else 1)
    elif file_type in ['IoTID20', 'IoTID']:
        data['label'] = data['Label'].apply(lambda x: 0 if str(x).strip() == 'Normal' else 1)
    elif file_type == 'Kitsune':
        data['label'] = data['Label']
    elif file_type in ['CICIoT', 'CICIoT2023']:
        data['label'] = data['attack_flag']
    else:
        logger.warning(f"Using generic anomal_judgment_label for {file_type}")
        data['label'] = anomal_judgment_label(data)
    
    timing_info['2_label_check'] = time.time() - start_time
    logger.info(f"Step 2 completed in {timing_info['2_label_check']:.2f}s")
    
    # 3. Feature embedding and preprocessing
    logger.info("Step 3: Feature embedding and preprocessing...")
    start_time = time.time()
    
    data = time_scalar_transfer(data, file_type)
    regul = 'N'
    
    # Try to load existing mapping first, if not found, create new mapping
    existing_mapping = None
    try:
        # Look for existing mapping in Dataset_ex folder
        mapping_path = f"../Dataset_ex/mapping_info/{file_type}_{file_number}_{heterogeneous_method}_mapping.pkl"
        if os.path.exists(mapping_path):
            import pickle
            with open(mapping_path, 'rb') as f:
                existing_mapping = pickle.load(f)
            logger.info(f"Loaded existing mapping from: {mapping_path}")
        else:
            logger.info(f"No existing mapping found at: {mapping_path}. Will create new mapping.")
    except Exception as e:
        logger.warning(f"Failed to load existing mapping: {e}. Will create new mapping.")
    
    embedded_dataframe, feature_list, category_mapping, data_list = choose_heterogeneous_method(
        data, file_type, heterogeneous_method, regul, existing_mapping=existing_mapping
    )
    
    group_mapped_df, mapped_info_df = map_intervals_to_groups(
        embedded_dataframe, category_mapping, data_list, regul
    )
    
    # Save newly created mapping to Dataset_ex folder if it was created
    if existing_mapping is None and category_mapping is not None:
        try:
            # Create mapping_info directory in Dataset_ex
            mapping_dir = f"../Dataset_ex/mapping_info"
            os.makedirs(mapping_dir, exist_ok=True)
            
            # Save the mapping
            mapping_path = f"{mapping_dir}/{file_type}_{file_number}_{heterogeneous_method}_mapping.pkl"
            import pickle
            with open(mapping_path, 'wb') as f:
                pickle.dump(category_mapping, f)
            logger.info(f"Saved new mapping to: {mapping_path}")
        except Exception as e:
            logger.warning(f"Failed to save mapping: {e}")
    
    timing_info['3_embedding'] = time.time() - start_time
    logger.info(f"Step 3 completed in {timing_info['3_embedding']:.2f}s")
    
    # 3.5 MinMax scaling
    logger.info("Step 3.5: Applying MinMax scaling...")
    start_time = time.time()
    
    # Get save functions via lazy import
    apply_minmax_scaling_and_save_scalers, _, _, _ = get_save_imports()
    
    X_scaled_for_pca, saved_scaler_path = apply_minmax_scaling_and_save_scalers(
        group_mapped_df, file_type, file_number, heterogeneous_method
    )
    
    logger.info(f"Step 3.5 completed in {time.time() - start_time:.2f}s")
    
    # 4. PCA processing
    logger.info("Step 4: PCA processing...")
    start_time = time.time()
    
    X = X_scaled_for_pca
    
    # Handle NaN values before PCA
    if isinstance(X, pd.DataFrame):
        # Fill NaN values with 0 or mean
        X_cleaned = X.fillna(0)
        logger.info(f"NaN values filled with 0. Original shape: {X.shape}")
    else:
        # For numpy arrays, replace NaN with 0
        X_cleaned = np.nan_to_num(X, nan=0.0)
        logger.info(f"NaN values replaced with 0 in numpy array. Original shape: {X.shape}")
    
    if file_type in ['DARPA98', 'DARPA', 'CICModbus23', 'CICModbus']:
        pca_want = 'N'
    else:
        pca_want = 'Y'
    
    if pca_want in ['Y', 'y']:
        X_processed = pca_func(X_cleaned)
        logger.info("PCA applied")
    else:
        X_processed = X_cleaned
        logger.info("PCA skipped")
    
    # Prepare data for clustering
    data_for_clustering = X_processed.to_numpy() if hasattr(X_processed, 'to_numpy') else X_processed
    
    # Final data validation
    if isinstance(data_for_clustering, np.ndarray):
        # Check for any remaining NaN or infinite values
        if np.any(np.isnan(data_for_clustering)) or np.any(np.isinf(data_for_clustering)):
            logger.warning("Found NaN or infinite values in clustering data. Replacing with 0.")
            data_for_clustering = np.nan_to_num(data_for_clustering, nan=0.0, posinf=0.0, neginf=0.0)
    
    original_labels = data['label'].to_numpy()
    
    logger.info(f"Final data validation - Shape: {data_for_clustering.shape}, Contains NaN: {np.any(np.isnan(data_for_clustering)) if isinstance(data_for_clustering, np.ndarray) else 'N/A'}")
    
    timing_info['4_pca_time'] = time.time() - start_time
    logger.info(f"Step 4 completed in {timing_info['4_pca_time']:.2f}s")
    logger.info(f"Data for clustering shape: {data_for_clustering.shape}")
    
    # 5. Run Jaccard-based Elbow method for algorithm selection
    logger.info("Step 5: Running Jaccard-based Elbow method for algorithm selection...")
    start_time = time.time()
    
    # Convert algorithm names to match the Jaccard Elbow method format
    # Only include algorithms that are supported by the Jaccard Elbow method
    jaccard_algorithms = []
    for algo in CLUSTERING_ALGORITHMS:
        if algo == 'kmeans':
            jaccard_algorithms.append('K-Means')
        elif algo == 'GMM':
            jaccard_algorithms.append('GMM')
        elif algo == 'SGMM':
            jaccard_algorithms.append('SGMM')
        elif algo == 'Gmeans':
            jaccard_algorithms.append('G-Means')
        elif algo == 'Xmeans':
            jaccard_algorithms.append('X-Means')
        elif algo == 'DBSCAN':
            jaccard_algorithms.append('DBSCAN')
        elif algo == 'MShift':
            jaccard_algorithms.append('Mean Shift')
        elif algo == 'FCM':
            jaccard_algorithms.append('FCM')
        elif algo == 'CK':
            jaccard_algorithms.append('CK')
        elif algo == 'NeuralGas':
            jaccard_algorithms.append('Neural Gas')
        # Skip kmedians as it's not implemented in the Jaccard Elbow method
    
    # Run Jaccard-based Elbow method
    try:
        # Get Jaccard elbow function via lazy import
        test_all_algorithms_with_jaccard_elbow = get_jaccard_elbow_imports()
        
        # === DATA CONSISTENCY FIX ===
        # To resolve potential floating-point discrepancies between the global reference and the main dataset,
        # we will find the indices of the normal samples in the main dataset and re-create the global reference
        # directly from it. This ensures 100% data consistency before passing to downstream functions.
        logger.info("[FIX] Ensuring data consistency between source data and known normal samples...")
        try:
            # Re-create the full normal samples set from the main processed data
            all_normal_samples_main = data_for_clustering[original_labels == 0]
            
            # Re-sample 90% from this consistent set
            num_all_normal = all_normal_samples_main.shape[0]
            if num_all_normal > 1:
                sample_size = int(num_all_normal * 0.90)
                if sample_size == 0 and num_all_normal > 0: sample_size = 1
                
                np.random.seed(42) # Use the same seed for reproducibility
                random_indices = np.random.choice(num_all_normal, size=sample_size, replace=False)
                
                # This is the new, 100% consistent reference set
                consistent_known_normal_samples_pca = all_normal_samples_main[random_indices]
                logger.info(f"[FIX] Re-created a consistent known_normal_samples set. Shape: {consistent_known_normal_samples_pca.shape}")
            else:
                logger.warning("[FIX] Not enough normal samples in the main data to create a consistent reference set.")
                consistent_known_normal_samples_pca = global_known_normal_samples_pca_for_cni # Fallback
        except Exception as e:
            logger.error(f"[FIX] Error during data consistency fix: {e}. Using original global reference.")
            consistent_known_normal_samples_pca = global_known_normal_samples_pca_for_cni
        # === END FIX ===

        jaccard_results = test_all_algorithms_with_jaccard_elbow(
            data_for_clustering, original_labels, jaccard_algorithms, file_type, file_number, consistent_known_normal_samples_pca, num_processes_for_algo=None
        )
        
        # Find best algorithm from Jaccard results
        best_algorithm = None
        best_jaccard_score = -1.0
        
        for algorithm, result in jaccard_results.items():
            jaccard_score = result['best_jaccard']
            if jaccard_score > best_jaccard_score:
                best_jaccard_score = jaccard_score
                best_algorithm = algorithm
        
        # Convert back to original algorithm name format
        if best_algorithm:
            if best_algorithm == 'K-Means':
                best_algorithm = 'kmeans'
            elif best_algorithm == 'GMM':
                best_algorithm = 'GMM'
            elif best_algorithm == 'SGMM':
                best_algorithm = 'SGMM'
            elif best_algorithm == 'G-Means':
                best_algorithm = 'Gmeans'
            elif best_algorithm == 'X-Means':
                best_algorithm = 'Xmeans'
            elif best_algorithm == 'DBSCAN':
                best_algorithm = 'DBSCAN'
            elif best_algorithm == 'Mean Shift':
                best_algorithm = 'MShift'
            elif best_algorithm == 'FCM':
                best_algorithm = 'FCM'
            elif best_algorithm == 'CK':
                best_algorithm = 'CK'
            elif best_algorithm == 'Neural Gas':
                best_algorithm = 'NeuralGas'
        
        logger.info(f"Jaccard Elbow method selected: {best_algorithm} with Jaccard score: {best_jaccard_score:.4f}")
        
        # For compatibility, we still need to run the selected algorithm with CNI
        # to get the actual cluster labels for the rest of the pipeline
        if best_algorithm:
            logger.info(f"Running selected algorithm {best_algorithm} with CNI for final labels...")
            best_cluster_labels, _, _ = run_clustering_with_cni(
                data, data_for_clustering, original_labels, best_algorithm,
                global_known_normal_samples_pca_for_cni, autotune_enabled,
                num_processes, algorithm_selection_threshold,
                file_type, file_number
            )
        else:
            best_cluster_labels = None
            logger.warning("No algorithm was selected by Jaccard Elbow method")
            
    except Exception as e:
        logger.error(f"Error in Jaccard Elbow method: {str(e)}")
        logger.info("Falling back to original algorithm selection method...")
        
        # Fallback to original method
        best_algorithm = None
        best_jaccard_score = -1.0
        best_cluster_labels = None
        
        # Pre-calculate total for logging optimization
        total_algorithms = len(CLUSTERING_ALGORITHMS)
        
        for idx, algorithm in enumerate(CLUSTERING_ALGORITHMS, 1):
            logger.info(f"Testing algorithm {idx}/{total_algorithms}: {algorithm}")
            algorithm_start_time = time.time()
            
            try:
                cluster_labels, _, _ = run_clustering_with_cni(
                    data, data_for_clustering, original_labels, algorithm,
                    global_known_normal_samples_pca_for_cni, autotune_enabled,
                    num_processes, algorithm_selection_threshold,
                    file_type, file_number
                )
                
                # Optimize condition checking
                if cluster_labels is not None and len(cluster_labels) == len(original_labels):
                    metrics = evaluate_clustering_wos(original_labels, cluster_labels)
                    micro_metrics = metrics.get('average=micro', {})
                    jaccard_score = micro_metrics.get('jaccard', -1.0)
                    
                    algorithm_time = time.time() - algorithm_start_time
                    logger.info(f"  {algorithm}: Jaccard Score = {jaccard_score:.4f} (Time: {algorithm_time:.2f}s)")
                    
                    # Optimize best algorithm checking
                    if jaccard_score > best_jaccard_score:
                        best_jaccard_score = jaccard_score
                        best_algorithm = algorithm
                        best_cluster_labels = cluster_labels
                        logger.info(f"    -> New best algorithm: {best_algorithm} with score {best_jaccard_score:.4f}")
                else:
                    logger.warning(f"  {algorithm}: Invalid cluster labels")
                    
            except Exception as e:
                logger.error(f"  {algorithm}: Error - {str(e)}")
                continue
    
    timing_info['5_clustering_time'] = time.time() - start_time
    logger.info(f"Step 5 completed in {timing_info['5_clustering_time']:.2f}s")
    
    # 6. Evaluate multiple thresholds for the best algorithm
    if best_algorithm:
        logger.info(f"Step 6: Evaluating multiple thresholds for best algorithm: {best_algorithm}")
        start_time = time.time()
        
        threshold_results = evaluate_thresholds_for_best_algorithm(
            data, data_for_clustering, original_labels, best_algorithm,
            global_known_normal_samples_pca_for_cni, autotune_enabled,
            num_processes, file_type, file_number
        )
        
        # Check if threshold evaluation failed
        if not threshold_results:
            logger.error("Threshold evaluation failed. Skipping final clustering.")
            best_cluster_labels = None
            best_jaccard_score = -1.0
        else:
            # Find best threshold
            best_threshold = 0.4  # Default
            best_threshold_jaccard = -1.0
            for threshold, result in threshold_results.items():
                if result['jaccard_score'] > best_threshold_jaccard:
                    best_threshold_jaccard = result['jaccard_score']
                    best_threshold = threshold
            
            logger.info(f"Best threshold: {best_threshold} with Jaccard score: {best_threshold_jaccard:.4f}")
            
            # Use the best threshold result
            if best_threshold in threshold_results:
                best_cluster_labels = threshold_results[best_threshold]['cluster_labels']
                best_jaccard_score = best_threshold_jaccard
            else:
                logger.error(f"Best threshold {best_threshold} not found in threshold_results. Available: {list(threshold_results.keys())}")
                # Fallback to first available threshold
                first_threshold = list(threshold_results.keys())[0]
                best_cluster_labels = threshold_results[first_threshold]['cluster_labels']
                best_jaccard_score = threshold_results[first_threshold]['jaccard_score']
                best_threshold = first_threshold
                logger.info(f"Using fallback threshold: {best_threshold} with Jaccard score: {best_jaccard_score:.4f}")
        
        timing_info['6_threshold_evaluation'] = time.time() - start_time
        logger.info(f"Step 6 completed in {timing_info['6_threshold_evaluation']:.2f}s")
    
    # 7. Save results
    if best_algorithm and best_cluster_labels is not None:
        logger.info(f"Best algorithm: {best_algorithm} with Jaccard score: {best_jaccard_score:.4f}")
        
        # Add cluster labels to original data
        data['cluster'] = best_cluster_labels
        
        # Create adjusted_cluster by inverting cluster labels (0->1, 1->0)
        if 'cluster' in data.columns:
            logger.info("Populating 'adjusted_cluster' by inverting 'data['cluster']' (0->1, 1->0).")
            if not isinstance(data['cluster'], pd.Series):
                current_cluster_series = pd.Series(data['cluster'], index=data.index if hasattr(data, 'index') else None)
            else:
                current_cluster_series = data['cluster']
            
            data['adjusted_cluster'] = 1 - current_cluster_series
        
        # Output result summary
        logger.info("Algorithm Performance Summary:")
        if 'jaccard_results' in locals():
            for alg, result in sorted(jaccard_results.items(), key=lambda x: x[1]['best_jaccard'], reverse=True):
                logger.info(f"  {alg}: {result['best_jaccard']:.4f}")
        else:
            logger.info(f"  {best_algorithm}: {best_jaccard_score:.4f}")
        
        # Standardize file_type to ensure consistent output directory and filename
        if file_type in ['CICIDS2017', 'CICIDS']:
            output_file_type = "CICIDS2017"
        elif file_type in ['DARPA', 'DARPA98']:
            output_file_type = "DARPA98"
        elif file_type in ['NSL-KDD', 'NSL_KDD']:
            output_file_type = "NSL-KDD"
        elif file_type in ['CICModbus23', 'CICModbus']:
            output_file_type = "CICModbus23"
        elif file_type in ['IoTID20', 'IoTID']:
            output_file_type = "IoTID20"
        elif file_type in ['CICIoT', 'CICIoT2023']:
            output_file_type = "CICIoT2023"
        # Explicitly handle non-alias types for clarity and consistency
        elif file_type in ['MiraiBotnet', 'Kitsune', 'netML']:
            output_file_type = file_type
        else:
            # Fallback for any other type
            output_file_type = file_type

        # Save basic CSV file to Dataset_ex in the new structure
        output_dir = os.path.join(dataset_ex_dir, "load_dataset", output_file_type)
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, f"best_clustering_{output_file_type}_{file_number}.csv")
        data.to_csv(output_filename, index=False)
        logger.info(f"Results saved to: {output_filename}")
        
        # 8. Save detailed results for best algorithm only
        start = time.time()
        
        # Determine GMM type
        determined_gmm_type = None 
        if best_algorithm.upper().startswith("GMM"):
            parts = best_algorithm.split('_')
            if len(parts) == 1 and parts[0].upper() == 'GMM': 
                determined_gmm_type = "normal"
            elif len(parts) == 2 and parts[0].upper() == 'GMM' and parts[1].lower() in ['normal', 'full', 'tied', 'diag']:
                determined_gmm_type = parts[1].lower()
        
        timing_info['7_save_time_start_hook'] = time.time() - start
        
        # Save time information to Dataset_ex
        apply_minmax_scaling_and_save_scalers, csv_compare_clustering_ex, csv_compare_matrix_clustering_ex, time_save_csv_VL_ex = get_save_imports()
        time_save_csv_VL_ex(file_type, file_number, best_algorithm, timing_info)
        
        # Save detailed comparison results for best algorithm only
        if 'cluster' in data.columns and len(data['cluster']) == len(original_labels):
            # Get save functions via lazy import
            _, csv_compare_clustering_ex, _, _ = get_save_imports()
            
            csv_compare_clustering_ex(
                file_type, best_algorithm, file_number, data,
                GMM_type=determined_gmm_type,
                optimal_cni_threshold=best_threshold
            )
            
            # Calculate metrics for both original and adjusted clusters
            y_true = data['label'].to_numpy()
            y_pred_original = data['cluster'].to_numpy()
            y_pred_adjusted = data['adjusted_cluster'].to_numpy()
            
            if eval_clustering_silhouette_flag:
                metrics_original = evaluate_clustering(y_true, y_pred_original, data_for_clustering)
                metrics_adjusted = evaluate_clustering(y_true, y_pred_adjusted, data_for_clustering)
            else:
                metrics_original = evaluate_clustering_wos(y_true, y_pred_original)
                metrics_adjusted = evaluate_clustering_wos(y_true, y_pred_adjusted)
            
            # Get save functions via lazy import
            _, _, csv_compare_matrix_clustering_ex, _ = get_save_imports()
            
            csv_compare_matrix_clustering_ex(
                file_type, file_number, best_algorithm,
                metrics_original, metrics_adjusted,
                GMM_type=determined_gmm_type,
                optimal_cni_threshold=best_threshold
            )
        else:
            logger.warning("[WARN Save] 'cluster' column not available or length mismatch with y_true. Skipping CSV result comparison saving.")
        
        # Save performance results for all algorithms (only matrix comparison)
        logger.info("Saving performance results for all algorithms...")
        if 'jaccard_results' in locals():
            for alg, result in jaccard_results.items():
                # Convert algorithm name back to original format
                original_alg = alg
                if alg == 'K-Means':
                    original_alg = 'kmeans'
                elif alg == 'GMM':
                    original_alg = 'GMM'
                elif alg == 'SGMM':
                    original_alg = 'SGMM'
                elif alg == 'G-Means':
                    original_alg = 'Gmeans'
                elif alg == 'X-Means':
                    original_alg = 'Xmeans'
                elif alg == 'DBSCAN':
                    original_alg = 'DBSCAN'
                elif alg == 'Mean Shift':
                    original_alg = 'MShift'
                elif alg == 'FCM':
                    original_alg = 'FCM'
                elif alg == 'CK':
                    original_alg = 'CK'
                elif alg == 'Neural Gas':
                    original_alg = 'NeuralGas'
                
                if original_alg != best_algorithm:  # Skip best algorithm as it's already saved above
                    try:
                        # Run the algorithm to get cluster labels
                        temp_cluster_labels, _, _ = run_clustering_with_cni(
                            data, data_for_clustering, original_labels, original_alg,
                            global_known_normal_samples_pca_for_cni, autotune_enabled,
                            num_processes, algorithm_selection_threshold,
                            file_type, file_number
                        )
                        
                        if temp_cluster_labels is not None:
                            # Create temporary data with this algorithm's results
                            temp_data = data.copy()
                            temp_data['cluster'] = temp_cluster_labels
                            temp_data['adjusted_cluster'] = 1 - temp_data['cluster']
                            
                            # Save only matrix comparison for other algorithms
                            y_true = temp_data['label'].to_numpy()
                            y_pred_original = temp_data['cluster'].to_numpy()
                            y_pred_adjusted = temp_data['adjusted_cluster'].to_numpy()
                            
                            if eval_clustering_silhouette_flag:
                                metrics_original = evaluate_clustering(y_true, y_pred_original, data_for_clustering)
                                metrics_adjusted = evaluate_clustering(y_true, y_pred_adjusted, data_for_clustering)
                            else:
                                metrics_original = evaluate_clustering_wos(y_true, y_pred_original)
                                metrics_adjusted = evaluate_clustering_wos(y_true, y_pred_adjusted)
                            
                            # Get save functions via lazy import
                            _, _, csv_compare_matrix_clustering_ex, _ = get_save_imports()
                            
                            csv_compare_matrix_clustering_ex(
                                file_type, file_number, original_alg,
                                metrics_original, metrics_adjusted,
                                GMM_type=None,
                                optimal_cni_threshold=0.4  # Use 0.4 for other algorithms
                            )
                        
                    except Exception as e:
                        logger.error(f"Failed to save results for algorithm {original_alg}: {e}")
        
        timing_info['7_save_time'] = time.time() - start
        logger.info(f"Step 7 finished. Save Time: {timing_info['7_save_time']:.2f}s")
        
        # Create result summary file in Dataset_ex
        summary_filename = os.path.join(dataset_ex_dir, f"clustering_summary_{file_type}_{file_number}.txt")
        with open(summary_filename, 'w', encoding='utf-8') as f:
            f.write(f"Best Clustering Algorithm Selection Results\n")
            f.write(f"==========================================\n")
            f.write(f"File Type: {file_type}\n")
            f.write(f"File Number: {file_number}\n")
            f.write(f"Best Algorithm: {best_algorithm}\n")
            f.write(f"Best Jaccard Score: {best_jaccard_score:.4f}\n")
            f.write(f"Best Threshold: {best_threshold}\n")
            f.write(f"Total Time: {time.time() - total_start_time:.2f}s\n")
            f.write(f"Internal Processes: {num_processes}\n")
            f.write(f"System CPU Cores: {cpu_count}\n")
            f.write(f"Available RAM: {memory.available / (1024**3):.2f} GB\n\n")
            
            f.write("All Algorithm Results (Jaccard Elbow Method):\n")
            f.write("--------------------------------------------\n")
            if 'jaccard_results' in locals():
                for alg, result in sorted(jaccard_results.items(), key=lambda x: x[1]['best_jaccard'], reverse=True):
                    f.write(f"{alg}: {result['best_jaccard']:.4f}\n")
            else:
                f.write(f"{best_algorithm}: {best_jaccard_score:.4f}\n")
            
            if threshold_results:
                f.write(f"\nThreshold Evaluation for {best_algorithm}:\n")
                f.write("----------------------------------------\n")
                for threshold, result in sorted(threshold_results.items()):
                    f.write(f"Threshold {threshold}: {result['jaccard_score']:.4f}\n")
        
        logger.info(f"Summary saved to: {summary_filename}")
        
    else:
        logger.error("No valid clustering results found")
    
    # Calculate total time
    total_end_time = time.time()
    timing_info['0_total_time'] = total_end_time - total_start_time
    
    # Save time information as a CSV
    apply_minmax_scaling_and_save_scalers, csv_compare_clustering_ex, csv_compare_matrix_clustering_ex, time_save_csv_VL_ex = get_save_imports()
    time_save_csv_VL_ex(file_type, file_number, best_algorithm if best_algorithm else "none", timing_info)
    
    total_time = time.time() - total_start_time
    logger.info(f"Total execution time: {total_time:.2f}s")
    
    # Print final system resource usage
    final_memory = psutil.virtual_memory()
    logger.info(f"Final memory usage: {final_memory.percent:.1f}%")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nThe job was stopped by the user.")
        print("Check the interim saved files:")
        print("- .csv files in ../Dataset_ex/progress_tracking/ folder")
        print("- Time log files in ../Dataset_ex/time_log/ folder")
        print("\nWhen restarting, it will resume from the point where it was interrupted.")
        sys.exit(0)
