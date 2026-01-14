#!/usr/bin/env python3
"""
Jaccard-based Elbow Method for Clustering Algorithm Selection

This module provides Jaccard-based Elbow methods for different clustering algorithms
to find optimal hyperparameters and compare algorithm performance.
"""


# 250912: set -> numpy unique optimization


import numpy as np
import time
import sys
import gc
import fcntl
import multiprocessing
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

# Force the start method to 'spawn' for more robust parallel processing,
# especially with libraries like scikit-learn that use their own threading.
# This helps prevent deadlocks that can occur with the default 'fork' method on Linux
# when multiple pools are created sequentially.
try:
    if multiprocessing.get_start_method() != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
        print("[INFO] Multiprocessing start method set to 'spawn'.")
except RuntimeError:
    print("[INFO] Multiprocessing start method could not be changed.")

from Tuning_hyperparameter.Grid_search import grid_search_neural_gas_custom
from sklearn.metrics import jaccard_score, calinski_harabasz_score, silhouette_score
from tqdm import tqdm
import importlib
import csv
import os
import hashlib
from datetime import datetime
import pandas as pd
import joblib # Added for KMeans parallel backend
import itertools # Added for itertools.product
from kneed import KneeLocator # Added for KneeLocator
import logging # Added for logging

# Import clustering_nomal_identify once at the top level
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify
from sklearn.cluster import MeanShift#, estimate_bandwidth # Added for MeanShift
from sklearn_extra.cluster import KMedoids
from Clustering_Method.clustering_CLARA import pre_clustering_CLARA
from Tuning_hyperparameter.Surrogate_score import compute_surrogate_score
from utils.generate_data_hash import generate_stable_data_hash
from utils.dynamic_import import dynamic_import_jaccard_elbow as dynamic_import
from Dataset_Choose_Rule.save_jaccard_elbow import load_jaccard_elbow_progress, save_jaccard_elbow_progress_parallel
from Tuning_hyperparameter.algorithm_jaccard_elbow import find_elbow_point, dbscan_eps_elbow_method, mean_shift_quantile_elbow_method
from Tuning_hyperparameter.gridsearch_jaccard_elbow import grid_search_with_unsupervised_score_custom, grid_search_with_unsupervised_score
from Tuning_hyperparameter.jaccard_run_single_clustering import run_single_clustering
from Evaluation.evaluate_jaccard_elbow import _evaluate_unsupervised_score


# Setup logger
logger = logging.getLogger(__name__)


# Cache for dynamic imports to avoid repeated imports
_import_cache = {}

def cleanup_memory():
    """Clean up memory by forcing garbage collection"""
    gc.collect()

def get_algorithm_config(algorithm):
    """Get algorithm configuration to reduce repeated condition checks"""
    configs = {
        'Kmeans': {
            'module': 'sklearn.cluster',
            'class': 'KMeans',
            'needs_cni': True,
            'needs_grid_search': True
        },
        'GMM': {
            'module': 'sklearn.mixture',
            'class': 'GaussianMixture',
            'needs_cni': True,
            'needs_grid_search': True
        },
        'SGMM': {
            'module': 'Clustering_Method.clustering_SGMM',
            'class': 'SphericalGaussianMixture',
            'needs_cni': True,
            'needs_grid_search': False
        },
        'FCM': {
            'module': 'Clustering_Method.clustering_FCM',
            'class': 'FuzzyCMeans',
            'needs_cni': True,
            'needs_grid_search': False
        },
        'CK': {
            'module': 'Clustering_Method.clustering_CK',
            'class': 'CKMeans',
            'needs_cni': True,
            'needs_grid_search': False
        },
        'Xmeans': {
            'module': 'Clustering_Method.clustering_Xmeans',
            'class': 'XMeansWrapper',
            'needs_cni': True,
            'needs_grid_search': False
        },
        'Gmeans': {
            'module': 'Clustering_Method.clustering_Gmeans',
            'class': 'GMeans',
            'needs_cni': True,
            'needs_grid_search': False
        },
        'DBSCAN': {
            'module': 'sklearn.cluster',
            'class': 'DBSCAN',
            'needs_cni': False,
            'needs_grid_search': True
        },
        'MShift': {
            'module': 'Clustering_Method.clustering_MShift',
            'class': 'MeanShiftWithDynamicBandwidth',
            'needs_cni': True,
            'needs_grid_search': True
        },
        'NeuralGas': {
            'module': 'Clustering_Method.clustering_NeuralGas',
            'class': 'NeuralGas',
            'needs_cni': True,
            'needs_grid_search': True
        },
        'CLARA': {
            'module': 'Clustering_Method.clustering_CLARA',
            'class': 'CLARA',
            'needs_cni': True,
            'needs_grid_search': False
        }
    }
    return configs.get(algorithm, {})


def cluster_count_elbow_method(X, true_labels, algorithm, max_k=300, file_type=None, file_number=None, global_known_normal_samples_pca=None, num_processes_for_algo=None, data_for_clustering=None, known_normal_idx=None):
    """Jaccard-based Elbow method for cluster count algorithms (Parallel Version)"""
    # Generate data hash for progress tracking
    data_hash = generate_stable_data_hash(file_type, file_number, X.shape)
    
    k_values = list(range(2, max_k + 1, 5))
    scores_final = []
    
    # Load existing progress (this function returns a SET of completed k_values)
    completed_k_values, existing_scores = load_jaccard_elbow_progress(algorithm, data_hash)

    if completed_k_values:
        print(f"[{algorithm}] Found existing progress: {len(completed_k_values)} k-values already completed")
    
    k_values_to_test = [k for k in k_values if k not in completed_k_values]
    
    if not k_values_to_test:
        print(f"[{algorithm}] All k-values already processed.")
        scores_final = [existing_scores.get(k, {}).get('surrogate_score', 0.0) for k in k_values]
    else:
        print(f"[{algorithm}] Testing {len(k_values_to_test)} k values from {min(k_values_to_test)} to {max(k_values_to_test)}...")
        
        # Set up parallel processing
        if num_processes_for_algo is None:
            num_processes_for_algo = multiprocessing.cpu_count()
        
        num_processes = min(num_processes_for_algo, len(k_values_to_test), multiprocessing.cpu_count())
        print(f"[{algorithm}] Using {num_processes} processes for Elbow method.")

        # Prepare arguments for each task
        # MODIFIED: Add data_hash and data_for_clustering to the arguments tuple
        tasks = [
            (k, algorithm, X, true_labels, global_known_normal_samples_pca, num_processes_for_algo, data_hash, data_for_clustering, known_normal_idx)
            for k in k_values_to_test
        ]

        # Use multiprocessing Pool
        try:
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = list(tqdm(
                    pool.imap(_evaluate_single_elbow_k, tasks),
                    total=len(tasks),
                    desc=f"{algorithm} Elbow"
                ))
        except (AssertionError, Exception) as e:
            print(f"\n!!!!!! PARALLEL ELBOW METHOD FAILED for {algorithm} !!!!!!")
            print(f"ERROR TYPE: {type(e).__name__}")
            print(f"ERROR: {e}")
            print("This usually means a worker process tried to start a sub-process (nested parallelism), which is not allowed.")
            print("This is often caused by a clustering wrapper (e.g., clustering_FCM.py) not receiving the required hyperparameter (like 'k' or 'max_clusters') and falling back to its internal auto-tuning logic, which creates its own process pool.")
            print("Now falling back to sequential execution...")
            # Fallback to sequential execution
            results = []
            for task in tqdm(tasks, desc=f"Elbow for {algorithm} (Sequential)"):
                results.append(_evaluate_single_elbow_k(task))

        # Process new results
        new_scores = {k: {'surrogate_score': ch_score, 'jaccard_score': jaccard_score} for k, ch_score, jaccard_score in results}
        existing_scores.update(new_scores)
        
        # Reconstruct the full list of scores in order
        scores_final = [existing_scores.get(k, {}).get('surrogate_score', 0.0) for k in k_values]

    if not scores_final:
        print(f"[{algorithm}] No scores were calculated.")
        return 2, 0.0

    # Find elbow point
    optimal_k = find_elbow_point(k_values, scores_final)
    best_score = max(scores_final) if scores_final else 0.0
    
    print(f"[{algorithm}] Optimal k={optimal_k} found by Elbow method.")
    
    return optimal_k, best_score


def comprehensive_algorithm_optimization(algorithm, X, true_labels, file_type, file_number, global_known_normal_samples_pca, num_processes_for_algo=None, known_normal_idx=None):
    """
    Performs a comprehensive optimization for a given clustering algorithm.
    It separates algorithms that auto-tune k (G-Means, X-Means) from others.
    """
    logger.info(f"===== Comprehensive Optimization for {algorithm} =====")
    start_time = time.time()
    
    best_params = None
    best_jaccard = 0.0

    # --- Branch for algorithms that auto-tune k ---
    if algorithm in ['Gmeans', 'Xmeans']:
        logger.info(f"[{algorithm}] Running as a self-tuning algorithm (single run).")
        
        # --- NEW: Conditional max_clusters for DARPA98 Xmeans ---
        max_clusters_for_run = 250
        if algorithm == 'Xmeans' and file_type in ['DARPA98', 'DARPA']:
            max_clusters_for_run = 50
            logger.info(f"[INFO] For DARPA98 dataset, Xmeans max_clusters is limited to {max_clusters_for_run}.")
        
        params = {'max_clusters': max_clusters_for_run, 'random_state': 42}
        
        # Run clustering once.
        clusters = run_single_clustering(
            algorithm, X, params,
            aligned_original_labels=true_labels,
            global_known_normal_samples_pca=global_known_normal_samples_pca,
            file_type=file_type
        )
        
        if clusters is not None:
            # Evaluate with CH score (for consistency in logs, though not used for tuning here)
            ch_score = _evaluate_unsupervised_score(X, clusters, known_normal_idx, file_type=file_type)
            logger.info(f"[{algorithm}] Found {len(np.unique(clusters))} clusters with CH Score: {ch_score:.2f}")
            
            # Evaluate final Jaccard score and get ratio_distribution in one call
            _, jaccard_score, cni_results_df = clustering_nomal_identify(
                data_features_for_clustering=X,
                clusters_assigned=clusters,
                original_labels_aligned=true_labels,
                global_known_normal_samples_pca=global_known_normal_samples_pca,
                threshold_value=0.3, # Use default threshold
                num_processes_for_algo=1,
                data_for_clustering=X,
                known_normal_idx=known_normal_idx
            )
            best_jaccard = jaccard_score
            best_params = params # Store the params used
            logger.info(f"[{algorithm}] Result Jaccard: {best_jaccard:.4f} with params: {best_params}")
            
            # Save progress for Gmeans/Xmeans (single run, but still need to save)
            try:
                from Tuning_hyperparameter.Surrogate_score import compute_surrogate_score
                # Note: file_type and file_number are passed as parameters to this function
                # For Gmeans/Xmeans, we don't pass diagnostic params (algorithm, data_hash, k) since it's a single run
                print(f"[DEBUG] About to calculate surrogate score. file_type={file_type}, file_number={file_number}")
                surrogate_score = compute_surrogate_score(X, clusters, known_normal_idx, file_type=file_type)
                print(f"[DEBUG] Surrogate score calculated: {surrogate_score}")
                
                # Get ratio_distribution from the CNI results we already have
                ratio_distribution = {}
                if cni_results_df is not None and not cni_results_df.empty:
                    ratio_distribution = cni_results_df['normal_ratio'].value_counts(normalize=True).to_dict()
                
                # Save with max_clusters as the key
                data_hash = generate_stable_data_hash(file_type, file_number, X.shape)
                save_jaccard_elbow_progress_parallel(algorithm, data_hash, params['max_clusters'], 
                                                     surrogate_score, jaccard_score, ratio_distribution)
                logger.info(f"[{algorithm}] Progress saved for max_clusters={params['max_clusters']}")
            except Exception as save_error:
                logger.warning(f"[{algorithm}] Failed to save progress: {save_error}")
        else:
            logger.warning(f"[{algorithm}] Single run failed.")
            best_jaccard = 0.0
            best_params = None

    # --- Branch for all other algorithms that require tuning ---
    else:
        # --- Step 1: Elbow method for finding optimal hyperparameter ---
        logger.info(f"[{algorithm}] Finding optimal hyperparameter using Surrogate-Score-based Elbow method...")

        # Branch handling: Call the appropriate Elbow function for the algorithm.
        if algorithm == 'DBSCAN':
            # DBSCAN needs to tune eps.
            optimal_k, elbow_ch_score = dbscan_eps_elbow_method(
                X=X, 
                true_labels=true_labels, 
                file_type=file_type, 
                file_number=file_number,
                global_known_normal_samples_pca=global_known_normal_samples_pca,
                num_processes_for_algo=num_processes_for_algo,
                known_normal_idx=known_normal_idx
            )
        elif algorithm == 'MShift':
            # Mean Shift needs to tune quantile. (This could also be a problem, so fix it together)
            optimal_k, elbow_ch_score = mean_shift_quantile_elbow_method(
                X=X, 
                true_labels=true_labels, 
                file_type=file_type, 
                file_number=file_number,
                global_known_normal_samples_pca=global_known_normal_samples_pca,
                num_processes_for_algo=num_processes_for_algo,
                known_normal_idx=known_normal_idx
            )
        else:        
            optimal_k, elbow_ch_score = cluster_count_elbow_method(
                X=X,
                true_labels=true_labels,
                algorithm=algorithm,
                max_k=300,  # Explicitly set max_k
                file_type=file_type,
                file_number=file_number,
                global_known_normal_samples_pca=global_known_normal_samples_pca,
                num_processes_for_algo=num_processes_for_algo,
                data_for_clustering=X,
                known_normal_idx=known_normal_idx
            )

        # --- Step 1b: Evaluate Jaccard for Elbow Result ---
        elbow_jaccard_score = 0.0
        elbow_best_params = {}
        if optimal_k is not None:
            logger.info(f"[{algorithm}] Evaluating Jaccard score for Elbow method's best params (k={optimal_k})...")
            # Reconstruct elbow best_params based on algorithm
            if algorithm == 'Kmeans':
                elbow_best_params = {'n_clusters': optimal_k, 'random_state': 42}
            elif algorithm == 'FCM' or algorithm == 'CK':
                elbow_best_params = {'max_clusters': optimal_k}
            elif algorithm == 'Kmedoids':
                elbow_best_params = {'n_clusters': optimal_k, 'random_state': 42, 'method': 'pam'}
            elif algorithm == 'GMM':
                elbow_best_params = {'n_components': optimal_k, 'covariance_type': 'diag', 'reg_covar': 1e-5, 'random_state': 42}
            elif algorithm == 'SGMM':
                elbow_best_params = {'n_components': optimal_k, 'covariance_type': 'spherical', 'reg_covar': 1e-5, 'random_state': 42}
            elif algorithm == 'CK':
                elbow_best_params = {'k': optimal_k, 'random_state': 42}
            elif algorithm == 'NeuralGas':
                elbow_best_params = {'n_start_nodes': 2, 'max_nodes': optimal_k, 'step': 0.2, 'max_edge_age': 50}
            elif algorithm == 'DBSCAN':
                elbow_best_params = {'eps': optimal_k, 'min_samples': 5}
            elif algorithm == 'MShift':
                elbow_best_params = {'bandwidth': optimal_k}
            elif algorithm == 'CLARA':
                elbow_best_params = {'k': optimal_k}
            
            if elbow_best_params:
                elbow_clusters = run_single_clustering(algorithm, X, elbow_best_params,
                                                        aligned_original_labels=true_labels, 
                                                        global_known_normal_samples_pca=global_known_normal_samples_pca,
                                                        file_type=file_type)

                if elbow_clusters is not None:
                    _, elbow_jaccard_score, _ = clustering_nomal_identify(
                        data_features_for_clustering=X, clusters_assigned=elbow_clusters,
                        original_labels_aligned=true_labels, global_known_normal_samples_pca=global_known_normal_samples_pca,
                        threshold_value=0.3, num_processes_for_algo=1, data_for_clustering=X,
                        known_normal_idx=known_normal_idx
                    )
                    logger.info(f"[{algorithm}] Elbow Result Jaccard: {elbow_jaccard_score:.4f} with params: {elbow_best_params}")
                else:
                    logger.warning(f"[{algorithm}] Clustering failed for Elbow params.")
            else:
                logger.warning(f"[{algorithm}] Could not reconstruct params for Elbow method.")
        else:
            logger.warning(f"[{algorithm}] Elbow method failed to find an optimal k.")

        # --- Step 2: Grid Search for fine-tuning ---
        grid_search_jaccard_score = 0.0
        best_params_from_grid = None
        best_ch_score_from_grid = -1.0
        if optimal_k is not None:
            logger.info(f"[{algorithm}] Grid Search: Testing parameter combinations around optimal k={optimal_k}...")
            
            logger.info(f"[{algorithm}] Calling Grid Search with the initial, globally-defined normal samples.")
            if global_known_normal_samples_pca is not None:
                 logger.info(f"[{algorithm}] Normal samples for Grid Search shape: {global_known_normal_samples_pca.shape}")

            # Select the correct algorithm and run grid search
            if algorithm == 'Kmeans':
                param_grid = {
                    'n_clusters': [optimal_k],
                    'random_state': [42],
                    'n_init': [10, 20, 50, 80]
                }
                best_params_from_grid, best_ch_score_from_grid = grid_search_with_unsupervised_score(X, algorithm, param_grid, file_type, file_number, num_processes_for_algo, true_labels=true_labels, global_known_normal_samples_pca=global_known_normal_samples_pca, known_normal_idx=known_normal_idx)
            
            elif algorithm == 'Kmedoids':
                param_grid = {
                    'n_clusters': [optimal_k],
                    'random_state': [42],
                    'method': ['pam', 'alternate']
                }
                best_params_from_grid, best_ch_score_from_grid = grid_search_with_unsupervised_score(X, algorithm, param_grid, file_type, file_number, num_processes_for_algo, true_labels=true_labels, global_known_normal_samples_pca=global_known_normal_samples_pca, known_normal_idx=known_normal_idx)

            elif algorithm == 'GMM':
                param_grid = {
                    'n_components': [optimal_k],
                    'random_state': [42]
                }
                param_grid['covariance_type'] = ['full', 'tied', 'diag', 'spherical']
                param_grid['reg_covar'] = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

                # Generate all combinations of parameters using itertools.product
                keys, values = zip(*param_grid.items())
                all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
                
                # Use the custom grid search function that takes a list of dicts
                best_params_from_grid, best_ch_score_from_grid = grid_search_with_unsupervised_score_custom(X, algorithm, all_combinations, file_type, file_number, num_processes_for_algo, true_labels=true_labels, global_known_normal_samples_pca=global_known_normal_samples_pca, known_normal_idx=known_normal_idx)

            elif algorithm == 'SGMM':
                param_grid = {
                    'n_components': [optimal_k],
                    'random_state': [42],
                    'covariance_type': ['spherical'], # SGMM is GMM with spherical covariance
                    'reg_covar': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
                }
                best_params_from_grid, best_ch_score_from_grid = grid_search_with_unsupervised_score(X, algorithm, param_grid, file_type, file_number, num_processes_for_algo, true_labels=true_labels, global_known_normal_samples_pca=global_known_normal_samples_pca, known_normal_idx=known_normal_idx)
                
            elif algorithm == 'DBSCAN':
                # DBSCAN: eps and min_samples need Grid Search
                # We use the 'optimal_k' from elbow as the starting 'eps' value
                param_grid = {
                    'eps': [max(0.1, optimal_k - 0.1), optimal_k, optimal_k + 0.1],
                    'min_samples': [2, 4, 6, 8, 10]
                }
                best_params_from_grid, best_ch_score_from_grid = grid_search_with_unsupervised_score(X, algorithm, param_grid, file_type, file_number, num_processes_for_algo, true_labels=true_labels, global_known_normal_samples_pca=global_known_normal_samples_pca, known_normal_idx=known_normal_idx)

            elif algorithm == 'MShift':
                # Mean Shift: bandwidth needs Grid Search
                # We use 'optimal_k' from elbow as the starting 'bandwidth' value
                param_grid = {'bandwidth': [optimal_k * 0.1, optimal_k * 0.5, optimal_k, optimal_k * 1.5]}
                best_params_from_grid, best_ch_score_from_grid = grid_search_with_unsupervised_score(X, algorithm, param_grid, file_type, file_number, num_processes_for_algo, true_labels=true_labels, global_known_normal_samples_pca=global_known_normal_samples_pca, known_normal_idx=known_normal_idx)

            elif algorithm == 'CLARA':
                param_grid = []
                # Searches centered around best_k found in Elbow.
                best_k = elbow_best_params.get('k', 20) # If Elbow fails, use default value 20
                k_values = sorted(list(set([k for k in [best_k - 2, best_k, best_k + 2] if k > 1])))

                # Defines test sample and iteration combinations (reduced to speed up)
                sample_multipliers = [1.0, 1.5] # Multiplier for the base (40 + 2*k)
                iteration_values = [3, 5]
                
                # Generate parameter combinations
                for k_val in k_values:
                    base_samples = 40 + 2 * k_val
                    for multiplier in sample_multipliers:
                        num_samples = int(base_samples * multiplier)
                        if num_samples > X.shape[0]:
                            num_samples = X.shape[0]

                        for num_iter in iteration_values:
                            param_grid.append({
                                'k': k_val, 
                                'number_samples': num_samples, 
                                'number_iterations': num_iter
                            })
                
                # Run Grid Search with generated combinations
                best_params_from_grid, best_ch_score_from_grid = grid_search_with_unsupervised_score_custom(
                    X, algorithm, param_grid, file_type, file_number, num_processes_for_algo,
                    true_labels=true_labels, global_known_normal_samples_pca=global_known_normal_samples_pca,
                    known_normal_idx=known_normal_idx
                )

            elif algorithm in ['FCM', 'CK']:
                # For these algorithms, Elbow result is sufficient for minimal change.
                if algorithm == 'FCM':
                    best_params_from_grid = {'n_clusters': optimal_k, 'random_state': 42}
                elif algorithm == 'CK':
                    best_params_from_grid = {'k': optimal_k, 'random_state': 42}
                best_ch_score_from_grid = elbow_ch_score

            elif algorithm == 'NeuralGas':
                # Use custom grid search for Neural Gas
                param_combinations = [
                    {'n_start_nodes': 2, 'max_nodes': optimal_k, 'step': 0.1, 'max_edge_age': 50},
                    {'n_start_nodes': 2, 'max_nodes': optimal_k, 'step': 0.2, 'max_edge_age': 50},
                    {'n_start_nodes': 2, 'max_nodes': optimal_k, 'step': 0.2, 'max_edge_age': 80},
                    {'n_start_nodes': 5, 'max_nodes': optimal_k, 'step': 0.2, 'max_edge_age': 50},
                ]
                best_params_from_grid, best_ch_score_from_grid = grid_search_with_unsupervised_score_custom(X, algorithm, param_combinations, file_type, file_number, num_processes_for_algo, true_labels=true_labels, global_known_normal_samples_pca=global_known_normal_samples_pca, known_normal_idx=known_normal_idx)

            # --- Step 2b: Evaluate Jaccard for Grid Search Result ---
            if best_params_from_grid:
                logger.info(f"[{algorithm}] Evaluating Jaccard score for Grid Search's best params...")
                grid_search_clusters = run_single_clustering(
                    algorithm, X, best_params_from_grid,
                    aligned_original_labels=true_labels,
                    global_known_normal_samples_pca=global_known_normal_samples_pca,
                    file_type=file_type
                )
                if grid_search_clusters is not None:
                    _, grid_search_jaccard_score, _ = clustering_nomal_identify(
                        data_features_for_clustering=X, clusters_assigned=grid_search_clusters,
                        original_labels_aligned=true_labels, global_known_normal_samples_pca=global_known_normal_samples_pca,
                        threshold_value=0.3, num_processes_for_algo=1, data_for_clustering=X,
                        known_normal_idx=known_normal_idx
                    )
                    logger.info(f"[{algorithm}] Grid Search Result Jaccard: {grid_search_jaccard_score:.4f} with params: {best_params_from_grid}")
                else:
                    logger.warning(f"[{algorithm}] Clustering failed for Grid Search params.")
            else:
                logger.info(f"[{algorithm}] Grid Search did not find any valid parameters.")
        
        # --- Step 3: Final Selection ---
        # Ensure scores are floats for comparison
        final_elbow_score = float(elbow_ch_score) if elbow_ch_score is not None else -1.0
        final_grid_score = float(best_ch_score_from_grid) if best_ch_score_from_grid is not None else -1.0

        # Gracefully handle cases where jaccard score might not be a float
        elbow_jaccard_score = float(elbow_jaccard_score) if isinstance(elbow_jaccard_score, (int, float)) else 0.0
        grid_search_jaccard_score = float(grid_search_jaccard_score) if isinstance(grid_search_jaccard_score, (int, float)) else 0.0

        if final_grid_score > final_elbow_score:
            logger.info(f"[{algorithm}] Selecting Grid Search params based on Surrogate Score ({final_grid_score:.2f} > {final_elbow_score:.2f})")
            best_params = best_params_from_grid
            best_jaccard = grid_search_jaccard_score
        else:
            logger.info(f"[{algorithm}] Selecting Elbow params based on Surrogate Score ({final_elbow_score:.2f} >= {final_grid_score:.2f})")
            best_params = elbow_best_params
            best_jaccard = elbow_jaccard_score

    end_time = time.time()
    logger.info(f"[{algorithm}] Comprehensive optimization completed in {end_time - start_time:.2f}s")
    logger.info(f"[{algorithm}] Final best_jaccard: {best_jaccard}, Final best_params: {best_params}")

    return {'best_params': best_params, 'best_jaccard': best_jaccard}

def apply_jaccard_elbow_method(algorithm, X, true_labels, file_type=None, file_number=None, global_known_normal_samples_pca=None, num_processes_for_algo=None, known_normal_idx=None):
    """Apply comprehensive optimization (Jaccard Elbow + Grid Search) for each algorithm"""
    return comprehensive_algorithm_optimization(algorithm, X, true_labels, file_type, file_number, global_known_normal_samples_pca, num_processes_for_algo, known_normal_idx=known_normal_idx)

def test_all_algorithms_with_jaccard_elbow(X, true_labels, algorithms=None, file_type=None, file_number=None, global_known_normal_samples_pca=None, num_processes_for_algo=None, known_normal_idx=None):
    """
    Tests all specified clustering algorithms using the Jaccard Elbow method and returns the results.
    """
    
    # --- HOTFIX: Force replacement of Kmedoids with CLARA ---
    # This ensures CLARA is run regardless of what the calling script passes.
    if algorithms is not None and 'Kmedoids' in algorithms:
        logger.warning("HOTFIX: Found 'Kmedoids' in algorithm list, forcefully replacing with 'CLARA'.")
        algorithms = [algo if algo != 'Kmedoids' else 'CLARA' for algo in algorithms]
    # --- END HOTFIX ---

    results = {}
    
    if algorithms is None:
        algorithms = ['Kmeans', 'CLARA', 'DBSCAN', 'Xmeans', 'Gmeans', 'MShift', 
                      'NeuralGas', 'FCM', 'CK', 'GMM', 'SGMM']  # removed Kmedoids
    
    print("=" * 60)
    print("JACCARD-BASED ELBOW METHOD FOR ALGORITHM SELECTION")
    print("=" * 60)
    
    total_start_time = time.time()
    
    try:
        for algorithm in algorithms:
            # Note: DBSCAN and MShift now use internal chunking for large datasets
            # No need for pre-sampling here - they will handle memory management internally
            
            result_dict = apply_jaccard_elbow_method(algorithm, X, true_labels, file_type, file_number, global_known_normal_samples_pca, num_processes_for_algo, known_normal_idx=known_normal_idx)
            results[algorithm] = result_dict
    except KeyboardInterrupt:
        print(f"\nThe entire algorithm test was stopped by the user.")
        print(f"Returning results from {len(results)} algorithms completed so far.")
        # Continue with current results
    
    total_elapsed_time = time.time() - total_start_time
    
    # Sort results by Jaccard score, handling potential non-float values gracefully
    sorted_results = sorted(
        results.items(), 
        key=lambda x: float(x[1].get('best_jaccard', 0.0) or 0.0), 
        reverse=True
    )
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS (Sorted by Jaccard Score)")
    print("=" * 60)
    
    for i, (algorithm, result) in enumerate(sorted_results, 1):
        # Ensure the jaccard score is a float before printing
        best_jaccard_val = result.get('best_jaccard', 0.0)
        jaccard_val = float(best_jaccard_val) if isinstance(best_jaccard_val, (int, float)) else 0.0
        print(f"{i:2d}. {algorithm:12s}: Jaccard={jaccard_val:.4f}, "
              f"Best Params={result['best_params']}")
    
    print(f"\nTotal time: {total_elapsed_time:.2f}s")
    
    # Performance gap analysis
    if len(sorted_results) >= 2:
        best_algorithm = sorted_results[0][0]
        best_jaccard = float(sorted_results[0][1].get('best_jaccard', 0.0) or 0.0)
        second_best_jaccard = float(sorted_results[1][1].get('best_jaccard', 0.0) or 0.0)
        gap = best_jaccard - second_best_jaccard
        
        print(f"\nPerformance Gap Analysis:")
        print(f"Best Algorithm: {best_algorithm}")
        print(f"Gap between 1st and 2nd: {gap:.4f} ({gap*100:.1f}%)")
        
        if gap >= 0.5:
            print("Status: DRAMATIC GAP - Clear winner!")
        elif gap >= 0.2:
            print("Status: Significant gap - Good winner")
        else:
            print("Status: Small gap - Unclear winner")
    
    return results

if __name__ == "__main__":
    try:
        # Example usage
        print("Jaccard-based Elbow Method Module")
        print("Import this module in best_clustering_selector_parallel.py")
    except KeyboardInterrupt:
        print("\n\nThe job was stopped by the user.")
        print("Check the interim saved files:")
        print("- Dataset_ex/progress_tracking/jaccard_elbow_*_progress.csv files")
        print("\nWhen restarting, it will resume from the point where it was interrupted.")
        sys.exit(0)
