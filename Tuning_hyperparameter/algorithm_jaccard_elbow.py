import numpy as np
import time
import sys
import gc
import fcntl
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
import multiprocessing

from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth # Added for MeanShift
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import jaccard_score
from multiprocessing import Process, Queue, TimeoutError

# module imports
from utils.generate_data_hash import generate_data_hash, generate_stable_data_hash, get_existing_hash_for_file_type
from utils.dynamic_import import dynamic_import_jaccard_elbow as dynamic_import
from Dataset_Choose_Rule.save_jaccard_elbow import load_jaccard_elbow_progress, save_jaccard_elbow_progress_parallel, save_jaccard_elbow_progress
from Evaluation.evaluate_jaccard_elbow import _evaluate_unsupervised_score
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify
from Clustering_Method.clustering_CLARA import pre_clustering_CLARA
from utils.dynamic_bandwidth import estimate_bandwidth_with_timeout


# Setup logger
logger = logging.getLogger(__name__)


def find_elbow_point(values, scores):
    """Find elbow point using second derivative method"""
    if len(scores) < 3:
        return values[0] if values else 2
    
    # Calculate first derivative (improvements)
    improvements = []
    for i in range(1, len(scores)):
        improvement = scores[i] - scores[i-1]
        improvements.append(improvement)
    
    # Calculate second derivative (rate of change in improvements)
    if len(improvements) >= 2:
        second_derivatives = []
        for i in range(1, len(improvements)):
            second_derivative = improvements[i] - improvements[i-1]
            second_derivatives.append(second_derivative)
        
        # Find the point with maximum second derivative (elbow point)
        elbow_idx = np.argmax(second_derivatives) + 2  # +2 because of two diffs
        if elbow_idx < len(values):
            return values[elbow_idx]
    
    # Fallback: return the value with maximum score
    max_score_idx = np.argmax(scores)
    return values[max_score_idx]


def dbscan_eps_elbow_method(X, true_labels, min_samples=5, file_type=None, file_number=None, global_known_normal_samples_pca=None, num_processes_for_algo=None, known_normal_idx=None):
    """
    Elbow method for DBSCAN eps parameter, now tuned with Surrogate-Score.
    """
    data_hash = generate_stable_data_hash(file_type, file_number, X.shape)
    
    eps_values = np.linspace(0.1, 2.0, 20) # Test 20 eps values from 0.1 to 2.0
    scores_final = []
    
    # Define procs_to_use at the beginning of the function
    procs_to_use = num_processes_for_algo if num_processes_for_algo is not None else -1
    
    completed_eps_values, existing_scores = load_jaccard_elbow_progress("DBSCAN", data_hash)

    if completed_eps_values:
        print(f"[DBSCAN] Found existing progress: {len(completed_eps_values)} eps values already completed")
    
    eps_values_to_test = [eps for eps in eps_values if eps not in completed_eps_values]
    
    if not eps_values_to_test:
        print(f"[DBSCAN] All eps-values already processed.")
        scores_final = [existing_scores.get(eps, {}).get('surrogate_score', 0.0) for eps in eps_values]
    else:
        print(f"[DBSCAN] Testing {len(eps_values_to_test)} eps values with min_samples={min_samples}...")
        
        for eps in tqdm(eps_values_to_test, desc="DBSCAN Elbow"):
            try:
                # Use chunked DBSCAN for large datasets
                from Clustering_Method.clustering_algorithm_chunked import dbscan_with_chunking
                n_samples = len(X)
                chunk_size = 30000
                
                if n_samples > chunk_size:
                    clusters = dbscan_with_chunking(X, eps=eps, min_samples=min_samples, chunk_size=chunk_size, overlap_ratio=0.1, n_jobs=procs_to_use)
                else:
                    clusters = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=procs_to_use).fit_predict(X)
                
                surrogate_score = _evaluate_unsupervised_score(X, clusters, known_normal_idx, file_type=file_type)
                
                # For logging, get the Jaccard score
                _, jaccard_for_log, final_results_df = clustering_nomal_identify(
                    data_features_for_clustering=X, clusters_assigned=clusters,
                    original_labels_aligned=true_labels, global_known_normal_samples_pca=global_known_normal_samples_pca,
                    threshold_value=0.3, num_processes_for_algo=1, data_for_clustering=X,
                    known_normal_idx=known_normal_idx
                )
                ratio_distribution = final_results_df['normal_ratio'].value_counts(normalize=True).to_dict() if final_results_df is not None and not final_results_df.empty else {}
                
                save_jaccard_elbow_progress_parallel("DBSCAN", data_hash, eps, surrogate_score, jaccard_for_log, ratio_distribution)
                existing_scores[eps] = {'surrogate_score': surrogate_score, 'jaccard_score': jaccard_for_log}

            except Exception as e:
                print(f"Error with DBSCAN eps={eps}: {e}")
                save_jaccard_elbow_progress_parallel("DBSCAN", data_hash, eps, 0.0, 0.0, {})
                existing_scores[eps] = {'surrogate_score': 0.0, 'jaccard_score': 0.0}
        
        scores_final = [existing_scores.get(eps, {}).get('surrogate_score', 0.0) for eps in eps_values]

    if not scores_final:
        print("[DBSCAN] No scores were calculated.")
        return 0.5, 0.0 # Return a default eps

    # Find the eps that gives the highest CH score (not an elbow, but best value)
    best_score_idx = np.argmax(scores_final)
    optimal_eps = eps_values[best_score_idx]
    best_surrogate_score = scores_final[best_score_idx]
    
    print(f"[DBSCAN] Optimal eps={optimal_eps:.2f} found with Surrogate-Score={best_surrogate_score:.2f}")
    
    return optimal_eps, best_surrogate_score

def max_clusters_elbow_method(X, true_labels, algorithm, max_clusters_values=None, file_type=None, file_number=None, global_known_normal_samples_pca=None, num_processes_for_algo=None, known_normal_idx=None):
    """Jaccard-based Elbow method for max_clusters parameter"""
    # Generate data hash for progress tracking
    if file_type and file_number:
        data_hash = get_existing_hash_for_file_type(file_type, file_number)
    else:
        data_hash = generate_data_hash(X)
    
    if max_clusters_values is None:
        max_clusters_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]
    
    jaccard_scores = []
    
    # Load existing progress
    completed_max_clusters = load_jaccard_elbow_progress(algorithm, data_hash)
    if completed_max_clusters:
        print(f"[{algorithm}] Found existing progress: {len(completed_max_clusters)} max_clusters values already completed")
    
    print(f"[{algorithm}] Testing {len(max_clusters_values)} max_clusters values...")
    
    try:
        for max_clusters in tqdm(max_clusters_values, desc=f"{algorithm} Elbow"):
            # Skip if already completed
            if max_clusters in completed_max_clusters:
                continue
            try:
                if algorithm == 'Xmeans':
                    # Use X-Means with CNI using dynamic_import structure
                    try:
                        # Step 1: Use dynamic_import to get X-Means class
                        XMeans = dynamic_import("Clustering_Method.clustering_Xmeans", "XMeansWrapper")
                        if XMeans:
                            model = XMeans(max_clusters=max_clusters, random_state=42)
                            clusters = model.fit_predict(X)
                            
                            # Step 2: Apply CNI to get final labels
                            final_labels = clustering_nomal_identify(
                                X, true_labels, clusters, len(np.unique(clusters)), 
                                global_known_normal_samples_pca=global_known_normal_samples_pca, 
                                threshold_value=0.3, 
                                num_processes_for_algo=1 # Prevent nested parallelism
                            )
                            
                            # Step 3: Calculate Jaccard score
                            jaccard = jaccard_score(true_labels, final_labels, average='weighted')
                            jaccard_scores.append(jaccard)
                            save_jaccard_elbow_progress(algorithm, max_clusters, jaccard, data_hash)
                            continue
                        else:
                            print(f"Warning: X-Means not available, skipping {algorithm}")
                            continue
                    except Exception as e:
                        print(f"Error with X-Means + CNI: {e}, falling back to sklearn only")
                        model = KMeans(n_clusters=max_clusters, random_state=42, n_init=10)
                elif algorithm == 'Gmeans':
                    # Use G-Means with CNI using dynamic_import structure
                    try:
                        # Step 1: Use dynamic_import to get G-Means class
                        GMeans = dynamic_import("Clustering_Method.clustering_Gmeans", "GMeans")
                        if GMeans:
                            model = GMeans(max_clusters=max_clusters, random_state=42)
                            clusters = model.fit_predict(X)
                            
                            # Step 2: Apply CNI to get final labels
                            final_labels = clustering_nomal_identify(
                                X, true_labels, clusters, len(np.unique(clusters)), 
                                global_known_normal_samples_pca=global_known_normal_samples_pca, 
                                threshold_value=0.3, 
                                num_processes_for_algo=1 # Prevent nested parallelism
                            )
                            
                            # Step 3: Calculate Jaccard score
                            jaccard = jaccard_score(true_labels, final_labels, average='weighted')
                            jaccard_scores.append(jaccard)
                            save_jaccard_elbow_progress(algorithm, max_clusters, jaccard, data_hash)
                            continue
                        else:
                            print(f"Warning: G-Means not available, skipping {algorithm}")
                            continue
                    except Exception as e:
                        print(f"Error with G-Means + CNI: {e}, falling back to sklearn only")
                        model = KMeans(n_clusters=max_clusters, random_state=42, n_init=10)
                else:
                    print(f"Warning: Unknown algorithm {algorithm}")
                    continue
            
                pred_labels = model.fit_predict(X)
                
                # Check if clustering was successful
                if len(np.unique(pred_labels)) > 1:  # More than 1 cluster
                    jaccard = jaccard_score(true_labels, pred_labels, average='weighted')
                else:
                    jaccard = 0.0  # Single cluster or failed clustering
                    
                jaccard_scores.append(jaccard)
                
                # Save progress after each max_clusters value
                save_jaccard_elbow_progress(algorithm, max_clusters, jaccard, data_hash)
                continue
                
            except Exception as e:
                print(f"Error with {algorithm} max_clusters={max_clusters}: {e}")
                jaccard_scores.append(0.0)
                # Save error case as well
                save_jaccard_elbow_progress(algorithm, max_clusters, 0.0, data_hash)
    
    except KeyboardInterrupt:
        print(f"\n[{algorithm}] The job was stopped by the user.")
        print(f"[{algorithm}] Interim saved file: Dataset_ex/progress_tracking/jaccard_elbow_{data_hash}_{algorithm}_progress.csv")
        print(f"[{algorithm}] When restarting, it will resume from the point where it was interrupted.")
        # Return current best results
        if jaccard_scores:
            optimal_max_clusters = find_elbow_point(max_clusters_values, jaccard_scores)
            best_jaccard = max(jaccard_scores)
            return optimal_max_clusters, best_jaccard
        else:
            return 50, 0.0
    
    # Find elbow point
    optimal_max_clusters = find_elbow_point(max_clusters_values, jaccard_scores)
    best_jaccard = max(jaccard_scores) if jaccard_scores else 0.0
    
    print(f"[{algorithm}] Optimal max_clusters={optimal_max_clusters}, Best Jaccard={best_jaccard:.4f}")
    
    return optimal_max_clusters, best_jaccard

def _estimate_bandwidth_worker(X, quantile, n_samples, n_jobs, result_queue):
    """Worker function to run estimate_bandwidth and put the result in a queue."""
    try:
        bandwidth = estimate_bandwidth(X, quantile=quantile, n_samples=n_samples, n_jobs=n_jobs)
        result_queue.put(bandwidth)
    except Exception as e:
        result_queue.put(e)

def mean_shift_quantile_elbow_method(X, true_labels, n_samples=500, file_type=None, file_number=None, global_known_normal_samples_pca=None, num_processes_for_algo=None, known_normal_idx=None):
    """
    Elbow method for Mean Shift quantile parameter, now tuned with Surrogate-Score.
    """
    data_hash = generate_stable_data_hash(file_type, file_number, X.shape)
    
    quantile_values = np.linspace(0.05, 0.5, 10) # Test 10 quantile values
    scores_final = []

    # Define procs_to_use at the beginning of the function
    procs_to_use = num_processes_for_algo if num_processes_for_algo is not None else -1

    completed_quantiles, existing_scores = load_jaccard_elbow_progress("MShift", data_hash)

    if completed_quantiles:
        print(f"[MShift] Found existing progress: {len(completed_quantiles)} quantile values already completed")

    quantiles_to_test = [q for q in quantile_values if q not in completed_quantiles]
    
    if not quantiles_to_test:
        print(f"[MShift] All quantile-values already processed.")
        scores_final = [existing_scores.get(q, {}).get('surrogate_score', 0.0) for q in quantile_values]
    else:
        print(f"[MShift] Testing {len(quantiles_to_test)} quantile values with n_samples={n_samples}...")
        
        # Sub-sample for bandwidth estimation to speed up the process
        X_sample = X[np.random.choice(X.shape[0], min(n_samples, X.shape[0]), replace=False)]

        for quantile in tqdm(quantiles_to_test, desc="MShift Elbow"):
            try:
                bandwidth = estimate_bandwidth(X_sample, quantile=quantile, n_samples=n_samples)
                if bandwidth > 0:
                    # Note: MeanShift chunking is available but currently disabled
                    # If enabled in clustering_MShift.py, similar logic would apply here
                    # For now, using standard MeanShift
                    '''
                    # --- NEW: Chunking logic for DARPA98 ---
                    if file_type in ['DARPA98', 'DARPA'] and X.shape[0] > 30000:
                        import math
                        from tqdm import trange
                        logger.info(f"  [MShift-DARPA98] Applying chunked prediction for quantile={quantile:.2f}")
                        
                        chunk_size = 30000
                        n_samples_total = X.shape[0]
                        num_chunks = math.ceil(n_samples_total / chunk_size)
                        
                        model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                        # Fit the model on a sample to define cluster centers
                        model.fit(X_sample) 
                        
                        # Predict in chunks
                        all_clusters = np.array([], dtype=int)
                        for i in trange(num_chunks, desc="  Chunked Predict", leave=False):
                            start_idx = i * chunk_size
                            end_idx = min((i + 1) * chunk_size, n_samples_total)
                            chunk_data = X[start_idx:end_idx]
                            chunk_clusters = model.predict(chunk_data)
                            all_clusters = np.concatenate([all_clusters, chunk_clusters])
                        clusters = all_clusters
                    else:
                        # Original logic for other datasets
                        clusters = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit_predict(X)
                    '''
                    clusters = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit_predict(X)  # Commented out chunking logic
                else: # Handle case where bandwidth is zero
                    clusters = np.zeros(X.shape[0], dtype=int)

                surrogate_score = _evaluate_unsupervised_score(X, clusters, known_normal_idx, file_type=file_type)

                # For logging, get the Jaccard score
                _, jaccard_for_log, final_results_df = clustering_nomal_identify(
                    data_features_for_clustering=X, clusters_assigned=clusters,
                    original_labels_aligned=true_labels, global_known_normal_samples_pca=global_known_normal_samples_pca,
                    threshold_value=0.3, num_processes_for_algo=1, data_for_clustering=X,
                    known_normal_idx=known_normal_idx
                )
                ratio_distribution = final_results_df['normal_ratio'].value_counts(normalize=True).to_dict() if final_results_df is not None and not final_results_df.empty else {}
                
                save_jaccard_elbow_progress_parallel("MShift", data_hash, quantile, surrogate_score, jaccard_for_log, ratio_distribution)
                existing_scores[quantile] = {'surrogate_score': surrogate_score, 'jaccard_score': jaccard_for_log}

            except Exception as e:
                print(f"Error with MShift quantile={quantile}: {e}")
                save_jaccard_elbow_progress_parallel("MShift", data_hash, quantile, 0.0, 0.0, {})
                existing_scores[quantile] = {'surrogate_score': 0.0, 'jaccard_score': 0.0}

        scores_final = [existing_scores.get(q, {}).get('surrogate_score', 0.0) for q in quantile_values]

    if not scores_final:
        print("[MShift] No scores were calculated.")
        return 0.2, 0.0 # Return a default quantile

    # Find the quantile that gives the highest CH score
    best_score_idx = np.argmax(scores_final)
    optimal_quantile = quantile_values[best_score_idx]
    best_surrogate_score = scores_final[best_score_idx]
    
    print(f"[MShift] Optimal quantile={optimal_quantile:.2f} found with Surrogate-Score={best_surrogate_score:.2f}")

    # The function -> in max_score_utils.py will handle the final bandwidth calculation.
    return optimal_quantile, best_surrogate_score

def neural_gas_max_nodes_elbow_method(X, true_labels, n_start_nodes=2, file_type=None, file_number=None, num_processes_for_algo=None, global_known_normal_samples_pca=None):
    """Jaccard-based Elbow method for Neural Gas max_nodes parameter"""
    # Set default value for num_processes_for_algo if None
    if num_processes_for_algo is None:
        num_processes_for_algo = multiprocessing.cpu_count()
    
    # Generate data hash for progress tracking
    if file_type and file_number:
        data_hash = get_existing_hash_for_file_type(file_type, file_number)
    else:
        data_hash = generate_data_hash(X)
    
    max_nodes_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    jaccard_scores = []
    
    # Load existing progress
    completed_max_nodes = load_jaccard_elbow_progress("Neural Gas", data_hash)
    if completed_max_nodes:
        print(f"[Neural Gas] Found existing progress: {len(completed_max_nodes)} max_nodes values already completed")
    
    print(f"[Neural Gas] Testing {len(max_nodes_values)} max_nodes values with n_start_nodes={n_start_nodes}...")
    
    try:
        for max_nodes in tqdm(max_nodes_values, desc="Neural Gas Elbow"):
            # Skip if already completed
            if max_nodes in completed_max_nodes:
                continue
            try:
                # Use Neural Gas with CNI for Jaccard Elbow Method
                from Clustering_Method.clustering_NeuralGas import clustering_NeuralGas_clustering
                try:
                    # Step 1: Perform Neural Gas clustering with specific max_nodes
                    # Note: We focus on max_nodes optimization in Elbow Method
                    # Other parameters use reasonable defaults based on Neural Gas literature
                    clusters, num_clusters = clustering_NeuralGas_clustering(
                        None, X, 
                        n_start_nodes=2,        # Standard starting point
                        max_nodes=max_nodes,    # This is the parameter we're optimizing
                        step=0.1,              # Smaller step for better convergence
                        max_edge_age=100,      # Higher edge age for stability
                        num_processes_for_algo=1 # Prevent nested parallelism
                    )
                    
                    # Step 2: Apply CNI to get final labels
                    final_labels = clustering_nomal_identify(
                        X, true_labels, clusters, len(np.unique(clusters)), 
                        global_known_normal_samples_pca=None, 
                        threshold_value=0.3, 
                        num_processes_for_algo=1 # Prevent nested parallelism
                    )
                    
                    # Step 3: Calculate Jaccard score
                    jaccard = jaccard_score(true_labels, final_labels, average='weighted')
                    jaccard_scores.append(jaccard)
                    save_jaccard_elbow_progress("Neural Gas", max_nodes, jaccard, data_hash)
                    continue
                except Exception as e:
                    print(f"Error with Neural Gas + CNI: {e}, falling back to sklearn only")
                    # Fallback to sklearn KMeans if Neural Gas fails
                    from sklearn.cluster import KMeans
                    model = KMeans(n_clusters=10, random_state=42, n_init=10)
                    pred_labels = model.fit_predict(X)
                
                # Check if clustering was successful
                if len(np.unique(pred_labels)) > 1:  # More than 1 cluster
                    jaccard = jaccard_score(true_labels, pred_labels, average='weighted')
                else:
                    jaccard = 0.0  # Single cluster or failed clustering
                    
                jaccard_scores.append(jaccard)
            
                # Save progress after each max_nodes value
                save_jaccard_elbow_progress("Neural Gas", max_nodes, jaccard, data_hash)
                continue
                
            except Exception as e:
                print(f"Error with Neural Gas max_nodes={max_nodes}: {e}")
                jaccard_scores.append(0.0)
                # Save error case as well
                save_jaccard_elbow_progress("Neural Gas", max_nodes, 0.0, data_hash)
    
    except KeyboardInterrupt:
        print(f"\n[Neural Gas] The job was stopped by the user.")
        print(f"[Neural Gas] Interim saved file: Dataset_ex/progress_tracking/jaccard_elbow_{data_hash}_Neural Gas_progress.csv")
        print(f"[Neural Gas] When restarting, it will resume from the point where it was interrupted.")
        # Return current best results
        if jaccard_scores:
            optimal_max_nodes = find_elbow_point(max_nodes_values, jaccard_scores)
            best_jaccard = max(jaccard_scores)
            return optimal_max_nodes, best_jaccard
        else:
            return 50, 0.0
    
    # Find elbow point
    optimal_max_nodes = find_elbow_point(max_nodes_values, jaccard_scores)
    best_jaccard = max(jaccard_scores) if jaccard_scores else 0.0
    
    print(f"[Neural Gas] Optimal max_nodes={optimal_max_nodes}, Best Jaccard={best_jaccard:.4f}")
    
    return optimal_max_nodes, best_jaccard