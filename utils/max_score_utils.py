#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for Max Score based hyperparameter optimization.
This module is created to house the logic for the --max_score flag
in best_clustering_selector_parallel.py, keeping the main script cleaner
and avoiding modifications to the stable Jaccard_Elbow_Method.py.
"""

import os
import csv
import time
import numpy as np
import multiprocessing
from datetime import datetime
from tqdm import tqdm
import logging
import hashlib

# Module imports
from Tuning_hyperparameter.jaccard_run_single_clustering import run_single_clustering
from Tuning_hyperparameter.algorithm_jaccard_elbow import dbscan_eps_elbow_method, mean_shift_quantile_elbow_method
from Modules.Jaccard_Elbow_Method import comprehensive_algorithm_optimization
# from Dataset_Choose_Rule.save_jaccard_elbow import save_jaccard_elbow_progress_parallel


# Platform-specific import for file locking
import sys
IS_LINUX = sys.platform == "linux"
if IS_LINUX:
    try:
        import fcntl
    except ImportError:
        IS_LINUX = False
        print("[WARN] fcntl module not found, file locking in parallel writes will be disabled. This is expected on non-Linux systems.")

logger = logging.getLogger(__name__)

# Helper function must be defined at the top level for multiprocessing to work.
def _evaluate_single_k_max_score(args_tuple):
    """Helper function to evaluate a single k value in parallel for the Max Score method."""
    from Tuning_hyperparameter.Surrogate_score import compute_surrogate_score

    k, algorithm, X, original_labels, global_known_normal_samples_pca, num_processes_for_algo, data_hash, data_for_clustering, known_normal_idx, file_type = args_tuple

    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    score = 0.0
    jaccard_for_log = 0.0
    ratio_distribution = {}
    nan_count = 0

    try:
        params = {}
        if algorithm in ['Kmeans', 'Kmedoids', 'CLARA']:
            params['n_clusters'] = k
            params['random_state'] = 42
        elif algorithm in ['GMM', 'SGMM']:
            params['n_components'] = k
            params['random_state'] = 42
        elif algorithm in ['FCM', 'CK', 'Xmeans', 'Gmeans']:
            params['max_clusters'] = k
        elif algorithm == 'NeuralGas':
            params['max_nodes'] = k

        #'''
        if file_type in ['CICIoT2023', 'CICIoT'] and algorithm == 'CK':
            num_processes_for_algo = 1
        #'''

        print(f"[MaxScore Worker] [{algorithm}] Starting k={k} evaluation...")
        clusters = run_single_clustering(
            algorithm, X, params,
            aligned_original_labels=original_labels,
            global_known_normal_samples_pca=global_known_normal_samples_pca,
            num_processes_for_algo=num_processes_for_algo,
            file_type=file_type
        )
        print(f"[MaxScore Worker] [{algorithm}] Clustering for k={k} completed. Starting surrogate score calculation...")

        if clusters is not None:
            unique_labels = np.unique(clusters)
            if len(unique_labels) > 1 or (len(unique_labels) == 1 and -1 not in unique_labels):
                # Check for NaN values in input data before computing surrogate score
                nan_mask = np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1)
                nan_count = nan_mask.sum()
                 
                score = compute_surrogate_score(X, clusters, known_normal_idx, 
                                                file_type=file_type, 
                                                algorithm=algorithm, 
                                                data_hash=data_hash, 
                                                k=k)
                print(f"[MaxScore Worker] [{algorithm}] Surrogate score for k={k}: {score:.4f}. Starting CNI...")
            else:
                score = 0.0
                print(f"[MaxScore Worker] [{algorithm}] k={k}: Invalid clusters, skipping CNI.")

            from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify
            _, jaccard_for_log, final_results_df = clustering_nomal_identify(
                data_features_for_clustering=X, clusters_assigned=clusters,
                original_labels_aligned=original_labels, global_known_normal_samples_pca=global_known_normal_samples_pca,
                threshold_value=0.3, num_processes_for_algo=1,
                data_for_clustering=X, known_normal_idx=known_normal_idx
            )
            print(f"[MaxScore Worker] [{algorithm}] CNI for k={k} completed. Jaccard: {jaccard_for_log:.4f}. Saving progress...")
            if final_results_df is not None and not final_results_df.empty:
                ratio_distribution = final_results_df['normal_ratio'].value_counts(normalize=True).to_dict()
        else:
            score = 0.0
            jaccard_for_log = 0.0

    except Exception as e:
        print(f"Error evaluating {algorithm} with k={k}: {e}")
        score = 0.0
        jaccard_for_log = 0.0

    _save_jaccard_elbow_progress_parallel_local(algorithm, data_hash, k, score, jaccard_for_log, ratio_distribution, nan_count)
    return k, score, jaccard_for_log

def _save_jaccard_elbow_progress_parallel_local(algorithm, data_hash, k, surrogate_score, jaccard_score, ratio_distribution, nan_count=0, max_retries=3):
    """Local, platform-aware version of save_jaccard_elbow_progress_parallel."""
    from Dataset_Choose_Rule.save_jaccard_elbow import get_jaccard_elbow_progress_file_path
    from Clustering_Method.cluster_normal_analyze import create_ratio_summary_10_bins
    
    progress_file = get_jaccard_elbow_progress_file_path(algorithm, data_hash)
    
    # Create fixed 10-bin summary for ratio distribution
    ratio_summary = create_ratio_summary_10_bins(ratio_distribution)
    
    # Define fixed header with 10 ratio bins and nan_count
    header = ['param_value', 'surrogate_score', 'jaccard_score', 'nan_count'] + list(ratio_summary.keys()) + ['timestamp']
    row_data = [k, surrogate_score, jaccard_score, nan_count] + list(ratio_summary.values()) + [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]

    for attempt in range(max_retries):
        try:
            os.makedirs(os.path.dirname(progress_file), exist_ok=True)
            with open(progress_file, 'a', newline='', encoding='utf-8') as f:
                if IS_LINUX: fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.seek(0, os.SEEK_END)
                    if f.tell() == 0:
                        writer = csv.writer(f)
                        writer.writerow(header)
                    writer = csv.writer(f)
                    writer.writerow(row_data)
                    f.flush()
                finally:
                    if IS_LINUX: fcntl.flock(f, fcntl.LOCK_UN)
            return
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))
            else:
                print(f"Error saving progress for {algorithm} k={k} after {max_retries} attempts: {e}")

def cluster_count_max_score_method(X, true_labels, algorithm, max_k=300, file_type=None, file_number=None, global_known_normal_samples_pca=None, num_processes_for_algo=None, data_for_clustering=None, known_normal_idx=None, is_chunk_processing=False):
    """Selects k based on the maximum surrogate score."""
    from utils.generate_data_hash import generate_stable_data_hash, generate_temp_chunk_hash
    from Dataset_Choose_Rule.save_jaccard_elbow import load_jaccard_elbow_progress
    
    # Hash logic is now conditional
    if is_chunk_processing:
        data_hash = generate_temp_chunk_hash(file_type, file_number)
    else:
        data_hash = generate_stable_data_hash(file_type, file_number, X.shape)
        
    #k_values = list(range(2, max_k + 1, 5))
    #if algorithm == 'CK' and file_type in ['CICIoT2023', 'CICIoT', 'CICIDS2017', 'CICIDS']:
    if algorithm == 'CK' and file_type in ['CICIDS2017', 'CICIDS', 'NSL-KDD']:
        k_values = list(range(72, max_k + 1, 5))
    else:
        k_values = list(range(2, max_k + 1, 5))
    completed_k_values, existing_scores = load_jaccard_elbow_progress(algorithm, data_hash)
    k_values_to_test = [k for k in k_values if k not in completed_k_values]

    if not k_values_to_test:
        print(f"[{algorithm}] All k-values already processed.")
        scores_final = [existing_scores.get(k, {}).get('surrogate_score', 0.0) for k in k_values]
    else:
        print(f"[{algorithm}] Testing {len(k_values_to_test)} k values for Max Score method...")
        num_processes = min(num_processes_for_algo, len(k_values_to_test), multiprocessing.cpu_count())
        tasks = [(k, algorithm, X, true_labels, global_known_normal_samples_pca, num_processes_for_algo, data_hash, data_for_clustering, known_normal_idx, file_type) for k in k_values_to_test]
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap(_evaluate_single_k_max_score, tasks), total=len(tasks), desc=f"{algorithm} Max Score"))
        
        new_scores = {k: {'surrogate_score': score, 'jaccard_score': j_score} for k, score, j_score in results}
        existing_scores.update(new_scores)
        scores_final = [existing_scores.get(k, {}).get('surrogate_score', 0.0) for k in k_values]

    # if not scores_final or all(s == 0 for s in scores_final):
    if not scores_final or all(s == 0 or np.isnan(s) for s in scores_final):
        logger.warning(f"[{algorithm}] No valid surrogate scores calculated. Defaulting to k=20.")
        return 20
    
    # Use nanargmax (insteed argmax) to ignore NaN values when finding the best score
    best_score_idx = np.nanargmax(scores_final)
    optimal_k = k_values[best_score_idx]
    best_score = scores_final[best_score_idx]
    
    logger.info(f"[{algorithm}] Optimal k={optimal_k} found by Max Score method with score={best_score:.4f}.")
    return optimal_k

def comprehensive_optimization_max_score(algorithm, X, true_labels, file_type, file_number, global_known_normal_samples_pca, num_processes_for_algo=None, known_normal_idx=None, is_chunk_processing=False):
    """Wrapper function to perform max-score based optimization for any algorithm."""
    from Modules.Jaccard_Elbow_Method import comprehensive_algorithm_optimization #test_all_algorithms_with_jaccard_elbow
    from sklearn.cluster import estimate_bandwidth
    from utils.generate_data_hash import generate_stable_data_hash, generate_temp_chunk_hash
    from Dataset_Choose_Rule.save_jaccard_elbow import load_jaccard_elbow_progress, save_jaccard_elbow_progress_parallel
    
    logger.info(f"===== Comprehensive Max Score Optimization for {algorithm} =====")
    
    # ===== UNIFIED CACHE CHECK (ALL ALGORITHMS) =====
    # Check for existing progress before running optimization
    if is_chunk_processing:
        # For chunk processing, use the temporary hash function
        data_hash = generate_temp_chunk_hash(file_type, file_number)
    else:
        # For main processing, hash includes shape to create a unique cache for the dataset size
        data_hash = generate_stable_data_hash(file_type, file_number, X.shape)
        
    completed_k_values, existing_scores = load_jaccard_elbow_progress(algorithm, data_hash)
    
    if algorithm in ['Gmeans', 'Xmeans']:
        # --- Special Chunking Path for DARPA98 Xmeans ---
        if algorithm == 'Xmeans' and file_type in ['DARPA98', 'DARPA'] and not is_chunk_processing:
            max_clusters_for_run = 50
            params = {'max_clusters': max_clusters_for_run, 'random_state': 42}
            
            # 1. Check cache before running
            if max_clusters_for_run in existing_scores:
                logger.info(f"[Xmeans-DARPA98] Found cached chunked result for max_clusters={max_clusters_for_run}")
                best_jaccard = existing_scores[max_clusters_for_run].get('jaccard_score', 0.0)
                logger.info(f"[Xmeans-DARPA98] Using cached Jaccard score: {best_jaccard:.4f}")
                return {'best_jaccard': best_jaccard, 'best_params': params, 'total_time': 0.0}

            # 2. If no cache, run the chunking logic
            import math
            from sklearn.metrics import jaccard_score
            from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify
            logger.info("[Xmeans-DARPA98 Special Path] No cache found. Starting chunked processing...")
            
            chunk_size = 30000
            n_samples = X.shape[0]
            num_chunks = math.ceil(n_samples / chunk_size)
            all_virtual_labels = np.array([], dtype=int)

            for i in tqdm(range(num_chunks), desc="Xmeans-DARPA98 Chunked Selection"):
                # ... (chunking and processing logic as before) ...
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, n_samples)
                chunk_data, chunk_labels = X[start_idx:end_idx], true_labels[start_idx:end_idx]
                chunk_kn_indices_local = None
                if known_normal_idx is not None:
                    mask = (known_normal_idx >= start_idx) & (known_normal_idx < end_idx)
                    chunk_kn_indices_local = known_normal_idx[mask] - start_idx

                clusters = run_single_clustering('Xmeans', chunk_data, params, aligned_original_labels=chunk_labels, global_known_normal_samples_pca=global_known_normal_samples_pca, file_type=file_type)
                
                if clusters is None:
                    virtual_labels = np.full(chunk_data.shape[0], -1, dtype=int)
                else:
                    virtual_labels, _, _ = clustering_nomal_identify(data_features_for_clustering=chunk_data, clusters_assigned=clusters, original_labels_aligned=chunk_labels, global_known_normal_samples_pca=global_known_normal_samples_pca, threshold_value=0.3, num_processes_for_algo=1, data_for_clustering=chunk_data, known_normal_idx=chunk_kn_indices_local)
                
                all_virtual_labels = np.concatenate([all_virtual_labels, virtual_labels])
            
            final_jaccard = jaccard_score(true_labels, all_virtual_labels, average='weighted') if len(all_virtual_labels) == len(true_labels) else 0.0
            
            # 3. Save the final result to cache
            logger.info(f"[Xmeans-DARPA98 Special Path] Saving result to cache. Final Jaccard: {final_jaccard:.4f}")
            save_jaccard_elbow_progress_parallel(algorithm, data_hash, max_clusters_for_run, 
                                                 surrogate_score=0.0, # Placeholder
                                                 jaccard_score=final_jaccard, 
                                                 ratio_distribution={}) # Placeholder
            
            return {'best_params': params, 'best_jaccard': final_jaccard}

        # --- Original Logic for Gmeans and other Xmeans cases ---
        else:
            max_clusters_for_run = 250

            # These auto-tune k, their optimization is simpler.
            ## Chunking path (DARPA98, Xmeans) and separation
            if max_clusters_for_run in existing_scores:
                logger.info(f"[{algorithm}] Found cached result for max_clusters={max_clusters_for_run}")
                best_jaccard = existing_scores[max_clusters_for_run].get('jaccard_score', 0.0)
                best_params = {'max_clusters': max_clusters_for_run, 'random_state': 42}
                logger.info(f"[{algorithm}] Using cached Jaccard: {best_jaccard:.4f}")
                logger.info(f"[{algorithm}] Comprehensive optimization completed (from cache) in 0.00s")
                logger.info(f"[{algorithm}] Final best_jaccard: {best_jaccard}, Final best_params: {best_params}")
                return {'best_jaccard': best_jaccard, 'best_params': best_params, 'total_time': 0.0}
            else:
                logger.info(f"[{algorithm}] No cached result found, running optimization...")
                # We call the single-algorithm optimization function directly from Jaccard_Elbow_Method
                # This avoids the premature "FINAL RESULTS" printout.
                return comprehensive_algorithm_optimization(
                    algorithm, X, true_labels, file_type, file_number, 
                    global_known_normal_samples_pca, 
                    num_processes_for_algo=num_processes_for_algo,
                    known_normal_idx=known_normal_idx
                )

    optimal_param = None
    if algorithm == 'DBSCAN':
        optimal_param, _ = dbscan_eps_elbow_method(X=X, true_labels=true_labels, file_type=file_type, file_number=file_number, global_known_normal_samples_pca=global_known_normal_samples_pca, num_processes_for_algo=num_processes_for_algo, known_normal_idx=known_normal_idx)
    elif algorithm == 'MShift':
        optimal_param, _ = mean_shift_quantile_elbow_method(X=X, true_labels=true_labels, file_type=file_type, file_number=file_number, global_known_normal_samples_pca=global_known_normal_samples_pca, num_processes_for_algo=num_processes_for_algo, known_normal_idx=known_normal_idx)
    elif algorithm == 'CK':
        optimal_param = cluster_count_max_score_method(X=X, true_labels=true_labels, algorithm=algorithm, max_k=275, file_type=file_type, file_number=file_number, global_known_normal_samples_pca=global_known_normal_samples_pca, num_processes_for_algo=num_processes_for_algo, data_for_clustering=X, known_normal_idx=known_normal_idx, is_chunk_processing=is_chunk_processing)
    else:
        # For all k-based algorithms
        optimal_param = cluster_count_max_score_method(X=X, true_labels=true_labels, algorithm=algorithm, max_k=300, file_type=file_type, file_number=file_number, global_known_normal_samples_pca=global_known_normal_samples_pca, num_processes_for_algo=num_processes_for_algo, data_for_clustering=X, known_normal_idx=known_normal_idx, is_chunk_processing=is_chunk_processing)

    # Construct the best parameters dictionary based on the optimal param found
    best_params = {}
    if algorithm == 'Kmeans': best_params = {'n_clusters': optimal_param, 'random_state': 42}
    elif algorithm in ['FCM', 'CK']: best_params = {'max_clusters': optimal_param}
    # elif algorithm == 'CLARA': best_params = {'k': optimal_param}
    elif algorithm == 'CLARA': best_params = {'n_clusters': optimal_param, 'random_state': 42}  # FIXED: CLARA uses n_clusters, not k
    elif algorithm == 'GMM': best_params = {'n_components': optimal_param, 'covariance_type': 'diag', 'reg_covar': 1e-5, 'random_state': 42}
    elif algorithm == 'SGMM': best_params = {'n_components': optimal_param, 'covariance_type': 'spherical', 'reg_covar': 1e-5, 'random_state': 42}
    elif algorithm == 'NeuralGas': best_params = {'n_start_nodes': 2, 'max_nodes': optimal_param, 'step': 0.2, 'max_edge_age': 50}
    elif algorithm == 'DBSCAN': best_params = {'eps': optimal_param, 'min_samples': 5}
    elif algorithm == 'MShift':
        ### Bandwidth calculation is now handled here, with a special path for DARPA98 ###
        if file_type in ['DARPA98', 'DARPA']:
            from utils.dynamic_bandwidth import estimate_bandwidth_with_timeout
            logger.info("[MShift] Applying dynamic bandwidth estimation with timeout for DARPA98.")
            bandwidth = estimate_bandwidth_with_timeout(
                X,
                initial_quantile=optimal_param,
                n_samples=min(10000, X.shape[0]),
                procs_to_use=num_processes_for_algo
            )
        else:
            # Original logic for all other datasets
            bandwidth = estimate_bandwidth(X, quantile=optimal_param) if optimal_param > 0 else None

        if bandwidth and bandwidth > 0:
            best_params = {'bandwidth': bandwidth}
        else:
            logger.warning(f"[{algorithm}] Invalid bandwidth for quantile {optimal_param}. Clustering may fail.")
            best_params = {}
    
    # Get the best Jaccard score from already completed results
    best_jaccard = 0.0
    if best_params and algorithm in ['Kmeans', 'CLARA', 'GMM', 'SGMM', 'FCM', 'CK', 'NeuralGas', 'Gmeans', 'Xmeans', 'DBSCAN']:
        # For k-based algorithms, we already calculated the Jaccard score during optimization
        # Get it from the existing results instead of re-running clustering
        from utils.generate_data_hash import generate_stable_data_hash, generate_temp_chunk_hash
        from Dataset_Choose_Rule.save_jaccard_elbow import load_jaccard_elbow_progress
        
        # Hash logic must be consistent with how it was generated above
        if is_chunk_processing:
            data_hash = generate_temp_chunk_hash(file_type, file_number)
        else:
            data_hash = generate_stable_data_hash(file_type, file_number, X.shape)
            
        completed_k_values, existing_scores = load_jaccard_elbow_progress(algorithm, data_hash)
        
        if algorithm in ['Kmeans', 'CLARA']:
            k_param = best_params.get('n_clusters')
        elif algorithm in ['GMM', 'SGMM']:
            k_param = best_params.get('n_components')
        elif algorithm in ['FCM', 'CK']:
            k_param = best_params.get('max_clusters')
        elif algorithm == 'NeuralGas':
            k_param = best_params.get('max_nodes')
        elif algorithm == 'DBSCAN':
            k_param = best_params.get('eps')  # Use eps parameter for DBSCAN
        elif algorithm in ['Gmeans', 'Xmeans']:
            k_param = best_params.get('max_clusters')  # Use max_clusters parameter for Gmeans and Xmeans
        
        if k_param and k_param in existing_scores:
            best_jaccard = existing_scores[k_param].get('jaccard_score', 0.0)
            logger.info(f"[{algorithm}] Using cached Jaccard score: {best_jaccard:.4f} for k={k_param}")
        else:
            logger.warning(f"[{algorithm}] No cached Jaccard score found for k={k_param}, using 0.0")
    elif best_params:
        # For MShift and other algorithms not in the cache list, we still need to calculate Jaccard score
        logger.info(f"[{algorithm}] Calculating Jaccard score for algorithm not in cache...")
        from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify
        clusters = run_single_clustering(algorithm, X, best_params, aligned_original_labels=true_labels, global_known_normal_samples_pca=global_known_normal_samples_pca, num_processes_for_algo=num_processes_for_algo, file_type=file_type)
        if clusters is not None:
             _, best_jaccard, _ = clustering_nomal_identify(
                data_features_for_clustering=X, clusters_assigned=clusters,
                original_labels_aligned=true_labels, global_known_normal_samples_pca=global_known_normal_samples_pca,
                threshold_value=0.3, num_processes_for_algo=1, data_for_clustering=X,
                known_normal_idx=known_normal_idx
            )
    
    logger.info(f"[{algorithm}] Max Score Result: Jaccard={best_jaccard:.4f} with params={best_params}")
    return {'best_params': best_params, 'best_jaccard': best_jaccard}
