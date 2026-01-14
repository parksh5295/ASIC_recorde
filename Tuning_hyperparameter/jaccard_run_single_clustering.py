import numpy as np
import time
import sys
import gc
import fcntl
import multiprocessing
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
from tqdm import trange # For chunked prediction progress bar

from sklearn.cluster import KMeans, DBSCAN, MeanShift#, estimate_bandwidth # Added for MeanShift
from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

# module imports
from utils.dynamic_import import dynamic_import_jaccard_elbow as dynamic_import
from utils.darpa_mshift_handler import handle_darpa98_mshift_special


# Setup logger
logger = logging.getLogger(__name__)

# Cache for dynamic imports to avoid repeated imports
_import_cache = {}

def run_single_clustering(algorithm, X, params, aligned_original_labels=None, global_known_normal_samples_pca=None, num_processes_for_algo=None, file_type=None):
    """
    Helper function to run a single clustering algorithm with given parameters.
    Returns the raw cluster labels.
    """
    print(f"DEBUG:[run_single_clustering] Entered for {algorithm} with received params: {params}")
    clusters = None  # Initialize clusters to None

    # Determine the number of processes to use for n_jobs
    procs_to_use = num_processes_for_algo if num_processes_for_algo is not None else -1

    # This dictionary defines the parameters for each algorithm's wrapper function.
    # It ensures that CNI-related parameters are passed correctly if the algorithm needs them.
    base_func_args = {
        'data': None,
        'X': X,
        'aligned_original_labels': aligned_original_labels,
        'global_known_normal_samples_pca': global_known_normal_samples_pca,
        'threshold_value': 0.3,  # A default threshold for CNI
        'num_processes_for_algo': 1,  # Avoid nested parallelism
        # **params is removed here to avoid passing the entire params dict to NeuralGas
    }
    # For NeuralGas, we must not pass the entire params dict.
    # We will handle its specific params inside its dedicated logic block.
    if algorithm != 'NeuralGas':
        base_func_args.update(params)

    try:
        # Group 1: Standard sklearn-compatible models (do not need CNI info directly)
        if algorithm in ['Kmeans', 'GMM', 'SGMM', 'DBSCAN', 'MShift', 'Kmedoids']:
            model = None
            if algorithm == 'Kmeans':
                model = KMeans(**params)
            elif algorithm == 'Kmedoids':
                effective_params = params.copy()
                if 'method' not in effective_params: effective_params['method'] = 'pam'
                if 'n_jobs' not in effective_params: effective_params['n_jobs'] = procs_to_use
                if effective_params['method'] == 'pam':
                    try:
                        model = KMedoids(**effective_params)
                    except MemoryError:
                        logger.warning(f"[Kmedoids] MemoryError with 'pam'. Falling back to 'alternate'.")
                        effective_params['method'] = 'alternate'
                        model = KMedoids(**effective_params)
                else:
                    model = KMedoids(**params)

            # Special handling for GMM/SGMM to retry with increased reg_covar and reduced k
            elif algorithm == 'GMM' or algorithm == 'SGMM':
                original_n_components = params.get('n_components')
                if not original_n_components:
                    logger.error(f"[{algorithm}] 'n_components' not provided in params.")
                    clusters = None
                else:
                    # Outer loop: Reduce n_components if fitting fails completely
                    for k_attempt in range(original_n_components, max(1, original_n_components - 10), -1):
                        reg_covar = params.get('reg_covar', 1e-5) # Start with a small reg_covar for each k_attempt
                        model_class = GaussianMixture if algorithm == 'GMM' else BayesianGaussianMixture
                        
                        # Inner loop: Increase reg_covar on covariance errors
                        for _ in range(7): # 7 retry attempts for reg_covar
                            try:
                                temp_params = params.copy()
                                temp_params['n_components'] = k_attempt
                                temp_params['reg_covar'] = reg_covar
                                if algorithm == 'SGMM' and 'covariance_type' not in temp_params:
                                    temp_params['covariance_type'] = 'spherical'
                                
                                current_model = model_class(**temp_params)
                                clusters = current_model.fit_predict(X)
                                break # Success, exit inner reg_covar loop
                            except ValueError as e:
                                if 'ill-defined empirical covariance' in str(e) or 'singular covariance' in str(e):
                                    logger.warning(f"[{algorithm} k={k_attempt}] Covariance error with reg_covar={reg_covar:.1e}. Retrying with {reg_covar * 10:.1e}.")
                                    reg_covar *= 10
                                else:
                                    logger.error(f"[{algorithm} k={k_attempt}] Non-covariance ValueError: {e}")
                                    clusters = None
                                    break # Exit inner loop for other errors
                        
                        if clusters is not None:
                            # If inner loop succeeded
                            if k_attempt != original_n_components:
                                logger.warning(f"[{algorithm}] Succeeded by automatically reducing n_components from {original_n_components} to {k_attempt}.")
                            break # Exit outer k_attempt loop
                    else:
                        # This 'else' belongs to the outer 'for k_attempt' loop.
                        # It runs if the outer loop completes without a 'break'.
                        logger.error(f"[{algorithm}] Fitting failed after multiple reg_covar adjustments and reducing n_components to {max(1, original_n_components - 10)}.")
                        clusters = None
                
                model = None # Ensure we don't refit below
            
            elif algorithm == 'DBSCAN':
                # Use custom chunked DBSCAN for large datasets
                from Clustering_Method.clustering_algorithm_chunked import dbscan_with_chunking
                n_samples = len(X)
                chunk_size = 30000
                
                if n_samples > chunk_size:
                    logger.info(f"[DBSCAN] Using chunked DBSCAN for {n_samples} samples")
                    eps = params.get('eps', 0.5)
                    min_samples = params.get('min_samples', 5)
                    clusters = dbscan_with_chunking(X, eps=eps, min_samples=min_samples, chunk_size=chunk_size, overlap_ratio=0.1, n_jobs=procs_to_use)
                    model = None  # No model object when using chunking
                else:
                    logger.info(f"[DBSCAN] Using standard DBSCAN for {n_samples} samples")
                    safe_params = params.copy()
                    safe_params['n_jobs'] = procs_to_use
                    model = DBSCAN(**safe_params)
                    
            elif algorithm == 'MShift':
                #'''
                if file_type in ['DARPA98', 'DARPA'] and X.shape[0] > 30000:
                    import math
                    import numpy as np
                    
                    logger.info(f"[MShift-DARPA98] Applying chunked prediction with chunk size 30000.")
                    
                    # 1. Fit the model on a sample to find cluster centers
                    sample_size = min(20000, X.shape[0])    # for fit model on a sample to find cluster centers
                    np.random.seed(42) # for reproducibility
                    X_sample = X[np.random.choice(X.shape[0], sample_size, replace=False)]
                    
                    # Use n_jobs=1 to avoid memory issues even when fitting on a sample
                    safe_params = params.copy()
                    safe_params['n_jobs'] = 1
                    model = MeanShift(**safe_params)

                    logger.info(f"[MShift-DARPA98] Fitting model on a sample of size {sample_size}...")
                    model.fit(X_sample)
                    logger.info(f"[MShift-DARPA98] Model fitting complete. Found {len(model.cluster_centers_)} clusters.")

                    # 2. Predict on the full dataset in chunks
                    chunk_size = 30000
                    n_samples_total = X.shape[0]
                    num_chunks = math.ceil(n_samples_total / chunk_size)
                    
                    all_clusters = np.array([], dtype=int)
                    logger.info(f"[MShift-DARPA98] Predicting on full dataset in {num_chunks} chunks...")
                    for i in trange(num_chunks, desc="[MShift-DARPA98] Chunked Prediction"):
                        start_idx = i * chunk_size
                        end_idx = min((i + 1) * chunk_size, n_samples_total)
                        chunk_data = X[start_idx:end_idx]
                        chunk_clusters = model.predict(chunk_data)
                        all_clusters = np.concatenate([all_clusters, chunk_clusters])
                    
                    clusters = all_clusters
                    model = None # Prevent re-running fit_predict below
                    logger.info("[MShift] Chunked clustering complete.")
                else:
                    safe_params = params.copy()
                    safe_params['n_jobs'] = procs_to_use
                    model = MeanShift(**safe_params)
                '''
                safe_params = params.copy()
                safe_params['n_jobs'] = procs_to_use
                model = MeanShift(**safe_params)
                '''
            
            if model:
                clusters = model.fit_predict(X)
                if algorithm == 'MShift':
                    logger.info("[MShift] Clustering complete.")

        # Group 2: Custom wrappers that might internally use CNI
        elif algorithm in ['FCM', 'CK', 'Xmeans', 'Gmeans', 'NeuralGas', 'CLARA']:
            algo_map = {
                'FCM': 'clustering_FCM', 
                'CK': 'clustering_CK', 
                'Xmeans': 'clustering_Xmeans',
                'Gmeans': 'clustering_Gmeans', 
                'NeuralGas': 'clustering_NeuralGas', 
                'CLARA': 'clustering_CLARA'
            }
            module_name = algo_map[algorithm]
            clustering_func = dynamic_import(f"Clustering_Method.{module_name}", module_name)
            
            if clustering_func:
                logger.debug(f"[{algorithm}] run_single_clustering received params: {params}")
                # Create a safe, mutable copy of the arguments for this specific call.
                func_args = base_func_args.copy()
                # --- FIX: Pass the specific hyperparameters (like 'k') to the wrapper ---
                func_args.update(params)
                logger.debug(f"[{algorithm}] func_args after updating with params: {func_args}")
                
                # For Gmeans and Xmeans, remove 'random_state' as they don't accept it.
                #if algorithm in ['Gmeans', 'Xmeans']:
                if algorithm in ['Gmeans', 'Xmeans', 'FCM', 'CK', 'CLARA']:
                    # Pop from the copied func_args, not the original params, for safety.
                    func_args.pop('random_state', None)
                    logger.debug(f"[{algorithm}] func_args after popping random_state: {func_args}")

                # Standardize the cluster count parameter to the correct name for each wrapper.
                # Grid Search might produce 'n_clusters', so we handle that case here.
                if 'n_clusters' in func_args:
                    '''
                    # If 'n_clusters' exists, rename it to 'k' for consistency.
                    func_args['k'] = func_args.pop('n_clusters')
                    '''
                    # If 'n_clusters' exists, rename it to the correct parameter name for the wrapper
                    cluster_param = func_args.pop('n_clusters')
                    if algorithm in ['FCM', 'CK', 'Xmeans', 'Gmeans']:
                        func_args['max_clusters'] = cluster_param
                    elif algorithm == 'CLARA':
                        func_args['k'] = cluster_param
                    else: # Fallback for other algos like NeuralGas
                        func_args['k'] = cluster_param # This will be handled next if it's NeuralGas
                
                # The elbow method passes 'k' for NeuralGas, which needs to be 'max_nodes'
                if algorithm == 'NeuralGas' and 'k' in func_args:
                    func_args['max_nodes'] = func_args.pop('k')

                # SPECIAL HANDLING FOR NeuralGas to match its specific wrapper signature
                if algorithm == 'NeuralGas':
                    # The wrapper expects specific arguments, not a generic dict.
                    # We extract them from the original 'params' dict passed to this function.
                    neural_gas_args = {
                        'data': func_args.get('data'),
                        'X': func_args.get('X'),
                        'aligned_original_labels': func_args.get('aligned_original_labels'),
                        'global_known_normal_samples_pca': func_args.get('global_known_normal_samples_pca'),
                        'threshold_value': func_args.get('threshold_value'),
                        'num_processes_for_algo': func_args.get('num_processes_for_algo'),
                        # Parameters from grid search or elbow method are in 'params'
                        'n_start_nodes': func_args.get('n_start_nodes', 2),
                        'max_nodes': func_args.get('max_nodes'),
                        'step': func_args.get('step', 0.2),
                        'max_edge_age': func_args.get('max_edge_age', 50)
                    }
                    # Filter out None values, but always keep 'data' even if it's None.
                    neural_gas_args = {k: v for k, v in neural_gas_args.items() if v is not None or k == 'data'}
                    logger.debug(f"[{algorithm}] Final args for NeuralGas call: {neural_gas_args}")
                    result = clustering_func(**neural_gas_args)
                else:
                    logger.debug(f"[{algorithm}] Final args for {algorithm} call: {func_args}")
                    result = clustering_func(**func_args)
                    
                clusters = result.get('raw_cluster_labels')
            else:
                logger.error(f"[run_single_clustering] Failed to import {module_name}")

    except Exception as e:
        logger.error(f"[run_single_clustering] Error running {algorithm} with params {params}: {e}")
        # Ensure clusters is None on failure
        clusters = None

    return clusters