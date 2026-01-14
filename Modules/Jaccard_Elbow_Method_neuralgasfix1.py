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
from sklearn.cluster import MeanShift, estimate_bandwidth # Added for MeanShift
from sklearn_extra.cluster import KMedoids
from Clustering_Method.clustering_CLARA import pre_clustering_CLARA
from Tuning_hyperparameter.Surrogate_score import compute_surrogate_score


# Setup logger
logger = logging.getLogger(__name__)


def save_jaccard_elbow_progress_parallel(algorithm, data_hash, k, surrogate_score, jaccard_score, ratio_distribution, max_retries=3):
    """
    Safely append progress to a CSV file from multiple processes, 
    including ratio distribution, timestamp, and retry logic.
    """
    progress_file = get_jaccard_elbow_progress_file_path(algorithm, data_hash)
    
    # Define header and row data dynamically
    header = ['param_value', 'surrogate_score', 'jaccard_score']
    dist_keys = sorted(ratio_distribution.keys())
    header.extend(dist_keys)
    header.append('timestamp') # Add timestamp to header

    row_data = [k, surrogate_score, jaccard_score]
    row_data.extend([ratio_distribution[key] for key in dist_keys])
    row_data.append(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) # Add timestamp value

    for attempt in range(max_retries):
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(progress_file), exist_ok=True)
            
            with open(progress_file, 'a', newline='', encoding='utf-8') as f:
                # Acquire an exclusive lock
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    # Check if file is empty to write header
                    f.seek(0, os.SEEK_END)
                    file_is_empty = f.tell() == 0
                    
                    writer = csv.writer(f)
                    
                    if file_is_empty:
                        writer.writerow(header)
                    
                    writer.writerow(row_data)
                    f.flush() # Ensure data is written to disk
                finally:
                    # Always release the lock
                    fcntl.flock(f, fcntl.LOCK_UN)
            
            return # Success, exit the loop and function

        except Exception as e:
            if attempt < max_retries - 1:
                # Wait before retrying
                time.sleep(0.1 * (attempt + 1)) 
            else:
                # Log final failure
                print(f"Error saving progress for {algorithm} k={k} after {max_retries} attempts: {e}")
                
# In _evaluate_single_elbow_k, the call to save_jaccard_elbow_progress_parallel does not need to be changed
# as max_retries has a default value.

def _evaluate_unsupervised_score(X, clusters, known_normal_idx):
    """
    @ 250921: Changed from CH-Score to surrogate score
    Evaluates clustering using the custom surrogate score.
    Returns the score. A higher score is better.
    Returns 0 if clustering is invalid (e.g., single cluster).
    """
    try:
        # The new score function can handle single clusters, but we can keep the check for robustness
        unique_labels = np.unique(clusters)
        if len(unique_labels) > 1 or (len(unique_labels) == 1 and -1 not in unique_labels):
            score = compute_surrogate_score(X, clusters, known_normal_idx)
            return score
        else:
            # If only noise clusters are found, or it's otherwise invalid.
            return 0.0
    except Exception as e:
        print(f"Error calculating surrogate score: {e}")
        return 0.0

def _evaluate_single_elbow_k(args_tuple):
    """Helper function to evaluate a single k value in parallel for the Elbow method."""
    # MODIFIED: Correctly unpack the 'data_hash' argument from the tuple
    k, algorithm, X, original_labels, global_known_normal_samples_pca, num_processes_for_algo, data_hash, data_for_clustering, known_normal_idx = args_tuple

    # Set thread-related environment variables to 1 to avoid nested parallelism issues
    # This is crucial when running sklearn models in a multiprocessing pool
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

    score = 0.0 # Changed from jaccard to a generic score
    jaccard_for_log = 0.0
    ratio_distribution = {}
    try:
        clusters = None
        num_clusters = k # k is the parameter being tuned by the elbow method

        # Revert to the stable structure from 6879080 for handling different algorithms
        if algorithm == 'Kmeans':
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = model.fit_predict(X)
        elif algorithm == 'Kmedoids':
            try:
                # First, try the default 'pam' method which is often more accurate
                logger.debug(f"[Kmedoids Elbow k={k}] Attempting with method='pam'")
                model = KMedoids(n_clusters=k, random_state=42, method='pam')
                clusters = model.fit_predict(X)
            except MemoryError:
                # If 'pam' fails due to memory, fall back to the more memory-efficient 'alternate' method
                logger.warning(f"[Kmedoids Elbow k={k}] MemoryError with method='pam'. Falling back to 'alternate'.")
                model = KMedoids(n_clusters=k, random_state=42, method='alternate')
                clusters = model.fit_predict(X)
        elif algorithm == 'GMM':
            model = GaussianMixture(n_components=k, random_state=42, reg_covar=1e-5)
            clusters = model.fit_predict(X)
        elif algorithm == 'SGMM':
            model = BayesianGaussianMixture(n_components=k, random_state=42, covariance_type='spherical', reg_covar=1e-5)
            clusters = model.fit_predict(X)
        elif algorithm in ['FCM', 'CK', 'Gmeans', 'Xmeans', 'MShift', 'NeuralGas', 'CLARA']:
            
            # Map the generic algorithm name to the specific function and parameter name
            algo_map = {
                'FCM': ('clustering_FCM', 'k'),
                'CK': ('clustering_CK', 'k'),
                'Gmeans': ('clustering_Gmeans', 'max_clusters'),
                'Xmeans': ('clustering_Xmeans', 'max_clusters'),
                'MShift': ('clustering_MShift', 'quantile'),
                'NeuralGas': ('clustering_NeuralGas', 'max_nodes'),
                'CLARA': ('clustering_CLARA', 'k')
            }
            
            module_name, param_name = algo_map[algorithm]
            
            # For MeanShift, k is not directly applicable to quantile. We'll use a simple mapping for elbow testing.
            # A small k suggests a smaller quantile for fewer clusters, a large k suggests a larger quantile.
            param_value = (k / 250) * 0.45 + 0.05 if algorithm == 'MShift' else k # Scale k to quantile range [0.05, 0.5]
            
            # Dynamically import and call the function
            clustering_func = dynamic_import(f"Clustering_Method.{module_name}", module_name)
            if clustering_func:
                # Prepare arguments for the clustering function
                func_args = {
                    'data': None,
                    'X': X,
                    'aligned_original_labels': original_labels,
                    'global_known_normal_samples_pca': global_known_normal_samples_pca,
                    'threshold_value': 0.3,
                    'num_processes_for_algo': 1,
                    param_name: param_value
                }
                
                # SPECIAL HANDLING FOR NeuralGas to match its specific wrapper signature
                if algorithm == 'NeuralGas':
                    # First, ensure cluster count parameter is named 'max_nodes'
                    if 'k' in func_args:
                        func_args['max_nodes'] = func_args.pop('k')
                    
                    # The wrapper expects specific arguments, not a generic dict.
                    # We need to extract them from func_args.
                    neural_gas_args = {
                        'data': func_args.get('data'),
                        'X': func_args.get('X'),
                        'aligned_original_labels': func_args.get('aligned_original_labels'),
                        'global_known_normal_samples_pca': func_args.get('global_known_normal_samples_pca'),
                        'threshold_value': func_args.get('threshold_value'),
                        'num_processes_for_algo': func_args.get('num_processes_for_algo'),
                        'n_start_nodes': func_args.get('n_start_nodes', 2),
                        'max_nodes': func_args.get('max_nodes'),
                        'step': func_args.get('step', 0.2),
                        'max_edge_age': func_args.get('max_edge_age', 50)
                    }
                    # Filter out None values to avoid passing them if they are not set
                    neural_gas_args = {k: v for k, v in neural_gas_args.items() if v is not None}
                    result = clustering_func(**neural_gas_args)
                else:
                    result = clustering_func(**func_args)
                    
                clusters = result.get('raw_cluster_labels')

        if clusters is not None:
            # --- CRITICAL CHANGE: Use unsupervised score for tuning ---
            score = _evaluate_unsupervised_score(X, clusters, known_normal_idx)
            
            # We still run CNI to get the ratio distribution for analysis, but NOT for tuning.
            final_labels, jaccard_for_log, final_results_df = clustering_nomal_identify(
                data_features_for_clustering=X,
                clusters_assigned=clusters,
                original_labels_aligned=original_labels,
                global_known_normal_samples_pca=global_known_normal_samples_pca,
                threshold_value=0.3,
                num_processes_for_algo=1,
                data_for_clustering=X,
                known_normal_idx=known_normal_idx
            )
            if final_results_df is not None and not final_results_df.empty:
                ratio_distribution = final_results_df['normal_ratio'].value_counts(normalize=True).to_dict()

    except Exception as e:
        print(f"Error with {algorithm} k={k}: {e}")
        # Return 0 score for this k value if an error occurs
        score = 0.0
        jaccard_for_log = 0.0
    
    # Save progress using the new generic score
    save_jaccard_elbow_progress_parallel(algorithm, data_hash, k, score, jaccard_for_log, ratio_distribution)
    return k, score, jaccard_for_log


# Cache for dynamic imports to avoid repeated imports
_import_cache = {}

def cleanup_memory():
    """Clean up memory by forcing garbage collection"""
    gc.collect()

def safe_save_grid_progress(progress_file, param_str, jaccard_score, max_retries=3):
    """Safely save Grid Search progress with file locking (Linux)"""
    for attempt in range(max_retries):
        try:
            # Check if file exists to determine if we need to write header
            file_exists = os.path.exists(progress_file)
            
            with open(progress_file, 'a') as f:
                # Apply exclusive lock (Linux)
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    # Write header if file doesn't exist
                    if not file_exists:
                        f.write('param_str,jaccard_score,timestamp\n')
                    
                    # Write progress data
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f'"{param_str}",{jaccard_score},{timestamp}\n')
                    f.flush()  # Force write to disk
                finally:
                    # Lock is automatically released when file is closed
                    pass
            return True
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                print(f"[Grid Search] Retry {attempt + 1}/{max_retries} for saving progress: {e}")
            else:
                print(f"[Grid Search] Failed to save progress after {max_retries} attempts: {e}")
                return False
    return False

def _evaluate_single_param_combination(args):
    """
    Helper function for grid search pool. Evaluates a single parameter combination.
    Now uses unsupervised score instead of Jaccard.
    """
    params, algorithm, X_local, completed_params, existing_scores, progress_file, data_hash, true_labels, global_known_normal_samples_pca, known_normal_idx = args

    # Set thread-related environment variables to 1 to avoid nested parallelism issues
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

    # Check if this combination has been completed before
    param_str = str(sorted(params.items()))
    if param_str in completed_params:
        return params, existing_scores.get(param_str, 0.0), True

    try:
        # clusters = None
        # Run clustering with the given parameters using the central helper function
        # MODIFIED: Pass through the labels and normal samples for CNI to work inside custom wrappers
        clusters = run_single_clustering(
            algorithm, X_local, params,
            aligned_original_labels=true_labels,
            global_known_normal_samples_pca=global_known_normal_samples_pca
        )

        # --- CRITICAL CHANGE: Use unsupervised score for tuning ---
        if clusters is not None:
            score = _evaluate_unsupervised_score(X_local, clusters, known_normal_idx)
        else:
            score = 0.0

        # Save progress
        progress_file = get_grid_search_progress_file_path(algorithm, data_hash)
        save_grid_search_progress(progress_file, param_str, score)
        return params, score, False

    except Exception as e:
        print(f"[{algorithm}] Error with params {params}: {e}")
        return params, 0.0, False

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

# Dynamic imports for custom clustering algorithms
def dynamic_import(module_name, class_name):
    """Dynamically import clustering algorithm classes with caching"""
    cache_key = f"{module_name}.{class_name}"
    
    if cache_key in _import_cache:
        return _import_cache[cache_key]
    
    try:
        module = importlib.import_module(module_name)
        class_obj = getattr(module, class_name)
        _import_cache[cache_key] = class_obj
        return class_obj
    except ImportError as e:
        print(f"Warning: Could not import {module_name}.{class_name}: {e}")
        _import_cache[cache_key] = None
        return None

# Progress tracking functions for Jaccard Elbow Method
def generate_data_hash(X):
    """Generate a unique hash for the dataset to distinguish different datasets."""
    try:
        # Create a hash based on dataset characteristics (more stable)
        # Use only shape and dtype which are most stable across runs
        data_info = f"{X.shape}_{X.dtype}"
        return hashlib.md5(data_info.encode()).hexdigest()[:8]
    except Exception:
        # Fallback to a simple hash if there's any issue
        return hashlib.md5(str(X.shape).encode()).hexdigest()[:8]

def generate_stable_data_hash(file_type, file_number, X_shape=None):
    """Generate a stable hash based on file information instead of data content."""
    try:
        # Use file information that doesn't change between runs
        # Convert file_number to string to avoid concatenation error
        if X_shape is not None:
            file_info = f"{file_type}_{str(file_number)}_{X_shape}"
        else:
            file_info = f"{file_type}_{str(file_number)}"
        return hashlib.md5(file_info.encode()).hexdigest()[:8]
    except Exception:
        # Fallback to a simple hash if there's any issue
        return hashlib.md5(f"{file_type}_{str(file_number)}".encode()).hexdigest()[:8]

def get_existing_hash_for_file_type(file_type, file_number):
    """Get existing hash for specific file_type and file_number to reuse progress files."""
    # Map common file types to their existing hashes
    hash_mapping = {
        "Kitsune_1": "82b321ce",  # Use existing hash
        "DARPA98_1": "d35ca016",  # Use existing hash
        # Add more mappings as needed
    }
    
    # Convert file_number to string to avoid concatenation error
    try:
        key = f"{file_type}_{str(file_number)}"
        if key in hash_mapping:
            return hash_mapping[key]
        else:
            # Fallback to new hash generation
            return generate_stable_data_hash(file_type, file_number)
    except Exception as e:
        print(f"[DEBUG] Error in get_existing_hash_for_file_type: {e}")
        print(f"[DEBUG] file_type: {file_type}, file_number: {file_number}, type: {type(file_number)}")
        # Fallback to new hash generation
        return generate_stable_data_hash(file_type, file_number)

def get_jaccard_elbow_progress_file_path(algorithm, data_hash):
    """Get the progress file path for Jaccard Elbow method."""
    import os
    # Use relative path to go up one level to the parent directory
    progress_dir = os.path.join("..", "Dataset_ex", "progress_tracking")
    os.makedirs(progress_dir, exist_ok=True)
    
    progress_file = os.path.join(progress_dir, f"jaccard_elbow_{data_hash}_{algorithm}_progress.csv")
    print(f"[DEBUG] Current working directory: {os.getcwd()}")
    print(f"[DEBUG] Progress directory: {os.path.abspath(progress_dir)}")
    print(f"[DEBUG] Progress file: {os.path.abspath(progress_file)}")
    return progress_file

def get_grid_search_progress_file_path(algorithm, data_hash):
    """Get the progress file path for Grid Search method."""
    progress_dir = os.path.join("..", "Dataset_ex", "progress_tracking")
    os.makedirs(progress_dir, exist_ok=True)
    return os.path.join(progress_dir, f"jaccard_elbow_{data_hash}_{algorithm}_Grid_progress.csv")

def load_jaccard_elbow_progress(algorithm, data_hash):
    """Load completed parameter values from progress file with optimized I/O."""
    progress_file = get_jaccard_elbow_progress_file_path(algorithm, data_hash)
    completed_values = set()
    existing_scores = {}
    
    if os.path.exists(progress_file):
        try:
            # Read entire file at once for better performance
            with open(progress_file, 'r', newline='', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Skip header and process all lines efficiently
            for line in lines[1:]:  # Skip header
                row = line.strip().split(',')
                if len(row) >= 3: # Check for at least param, surrogate_score, jaccard_score
                    try:
                        param_value = float(row[0])
                        completed_values.add(param_value)
                        # Load scores from the row
                        surrogate_score_val = float(row[1])
                        jaccard_score_val = float(row[2])
                        existing_scores[param_value] = {'surrogate_score': surrogate_score_val, 'jaccard_score': jaccard_score_val}
                    except ValueError:
                        continue  # Skip invalid rows
            
            print(f"[Jaccard Elbow Progress] Loaded {len(completed_values)} completed parameter values from {progress_file}")
        except Exception as e:
            print(f"[Jaccard Elbow Progress] Error loading progress file {progress_file}: {e}")
    
    return completed_values, existing_scores

def save_jaccard_elbow_progress(algorithm, param_value, jaccard_score, data_hash):
    """Save a completed parameter value to progress file."""
    progress_file = get_jaccard_elbow_progress_file_path(algorithm, data_hash)
    
    try:
        # Check if file exists to determine if we need to write header
        file_exists = os.path.exists(progress_file)
        
        with open(progress_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['param_value', 'jaccard_score', 'timestamp'])
            writer.writerow([param_value, jaccard_score, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            
    except Exception as e:
        print(f"[Jaccard Elbow Progress] Error saving progress to {progress_file}: {e}")
        import traceback
        traceback.print_exc()

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
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(_evaluate_single_elbow_k, tasks),
                total=len(tasks),
                desc=f"{algorithm} Elbow"
            ))
        
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

def dbscan_eps_elbow_method(X, true_labels, min_samples=5, file_type=None, file_number=None, global_known_normal_samples_pca=None, num_processes_for_algo=None, known_normal_idx=None):
    """
    Elbow method for DBSCAN eps parameter, now tuned with Surrogate-Score.
    """
    data_hash = generate_stable_data_hash(file_type, file_number, X.shape)
    
    eps_values = np.linspace(0.1, 2.0, 20) # Test 20 eps values from 0.1 to 2.0
    scores_final = []
    
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
                clusters = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
                
                ch_score = _evaluate_unsupervised_score(X, clusters, known_normal_idx)
                
                # For logging, get the Jaccard score
                _, jaccard_for_log, final_results_df = clustering_nomal_identify(
                    data_features_for_clustering=X, clusters_assigned=clusters,
                    original_labels_aligned=true_labels, global_known_normal_samples_pca=global_known_normal_samples_pca,
                    threshold_value=0.3, num_processes_for_algo=1, data_for_clustering=X,
                    known_normal_idx=known_normal_idx
                )
                ratio_distribution = final_results_df['normal_ratio'].value_counts(normalize=True).to_dict() if final_results_df is not None and not final_results_df.empty else {}
                
                save_jaccard_elbow_progress_parallel("DBSCAN", data_hash, eps, ch_score, jaccard_for_log, ratio_distribution)
                existing_scores[eps] = {'surrogate_score': ch_score, 'jaccard_score': jaccard_for_log}

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
    best_ch_score = scores_final[best_score_idx]
    
    print(f"[DBSCAN] Optimal eps={optimal_eps:.2f} found with CH-Score={best_ch_score:.2f}")
    
    return optimal_eps, best_ch_score

def max_clusters_elbow_method(X, true_labels, algorithm, max_clusters_values=None, file_type=None, file_number=None, global_known_normal_samples_pca=None, num_processes_for_algo=None):
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

def mean_shift_quantile_elbow_method(X, true_labels, n_samples=500, file_type=None, file_number=None, global_known_normal_samples_pca=None, num_processes_for_algo=None, known_normal_idx=None):
    """
    Elbow method for Mean Shift quantile parameter, now tuned with Surrogate-Score.
    """
    data_hash = generate_stable_data_hash(file_type, file_number, X.shape)
    
    quantile_values = np.linspace(0.05, 0.5, 10) # Test 10 quantile values
    scores_final = []

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
                    clusters = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit_predict(X)
                else: # Handle case where bandwidth is zero
                    clusters = np.zeros(X.shape[0], dtype=int)

                ch_score = _evaluate_unsupervised_score(X, clusters, known_normal_idx)

                # For logging, get the Jaccard score
                _, jaccard_for_log, final_results_df = clustering_nomal_identify(
                    data_features_for_clustering=X, clusters_assigned=clusters,
                    original_labels_aligned=true_labels, global_known_normal_samples_pca=global_known_normal_samples_pca,
                    threshold_value=0.3, num_processes_for_algo=1, data_for_clustering=X,
                    known_normal_idx=known_normal_idx
                )
                ratio_distribution = final_results_df['normal_ratio'].value_counts(normalize=True).to_dict() if final_results_df is not None and not final_results_df.empty else {}
                
                save_jaccard_elbow_progress_parallel("MShift", data_hash, quantile, ch_score, jaccard_for_log, ratio_distribution)
                existing_scores[quantile] = {'surrogate_score': ch_score, 'jaccard_score': jaccard_for_log}

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
    best_ch_score = scores_final[best_score_idx]
    
    print(f"[MShift] Optimal quantile={optimal_quantile:.2f} found with CH-Score={best_ch_score:.2f}")
    
    # Return the optimal quantile (as 'k') and its CH score
    return optimal_quantile, best_ch_score

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

def grid_search_with_unsupervised_score_custom(X, algorithm, param_combinations, file_type, file_number, num_processes_for_algo=None, true_labels=None, global_known_normal_samples_pca=None, known_normal_idx=None):
    """
    Performs grid search for a given algorithm and a list of parameter combinations,
    optimizing for an unsupervised score (Calinski-Harabasz).
    """
    from sklearn.model_selection import ParameterGrid
    import itertools
    
    # --- Progress Loading ---
    data_hash = generate_stable_data_hash(file_type, file_number, X.shape)
    progress_file = get_grid_search_progress_file_path(algorithm, data_hash)
    completed_params = set()
    existing_scores = {}

    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    param_str = row.get('param_str')
                    score = float(row.get('score', 0.0))
                    if param_str:
                        completed_params.add(param_str)
                        existing_scores[param_str] = score
            logger.info(f"[{algorithm}] Grid Search: Loaded {len(completed_params)} completed combinations from cache.")
        except Exception as e:
            logger.warning(f"[{algorithm}] Grid Search: Could not load progress file {progress_file}: {e}")

    # Filter out completed combinations
    param_combinations_to_run = [
        p for p in param_combinations if str(sorted(p.items())) not in completed_params
    ]

    if not param_combinations_to_run:
        logger.info(f"[{algorithm}] Grid Search: All parameter combinations are already completed.")
        # Return the best score from the loaded data
        if not existing_scores:
            return None, 0.0
        best_param_str = max(existing_scores, key=existing_scores.get)
        # Reconstruct the params dict from the string
        best_params = dict(eval(best_param_str))
        return best_params, existing_scores[best_param_str]

    # Set default value for num_processes_for_algo if None
    if num_processes_for_algo is None:
        num_processes_for_algo = multiprocessing.cpu_count()
    
    # Determine number of processes for parallel processing
    num_processes = min(num_processes_for_algo, len(param_combinations_to_run), multiprocessing.cpu_count())
    logger.info(f"[{algorithm}] Grid Search: Using {num_processes} processes for {len(param_combinations_to_run)} new parameter combinations")
    
    # Prepare arguments for parallel processing
    args_list = []
    for params in param_combinations_to_run:
        args_list.append((
            params, algorithm, X, completed_params, existing_scores, progress_file, data_hash,
            true_labels, global_known_normal_samples_pca, known_normal_idx
        ))
    
    best_score = -1.0
    best_params = None
    # Initialize with existing best score if any
    if existing_scores:
        best_param_str = max(existing_scores, key=existing_scores.get)
        best_params = dict(eval(best_param_str))
        best_score = existing_scores[best_param_str]

    try:
        # Use parallel processing if more than 1 process is available
        if num_processes > 1:
            print(f"[{algorithm}] Grid Search: Starting parallel processing with {num_processes} processes...")
            
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = list(tqdm(
                    pool.imap(_evaluate_single_param_combination, args_list),
                    total=len(args_list),
                    desc=f"{algorithm} Grid Search"
                ))
            
            # Process results
            for params, score, from_cache in results:
                if not from_cache:
                    # Update the main dicts for future cache hits
                    param_str = str(sorted(params.items()))
                    completed_params.add(param_str)
                    existing_scores[param_str] = score
                
                if score > best_score:
                    best_score = score
                    best_params = params
            
            logger.info(f"[{algorithm}] Grid Search: UPDATED best params: {best_params}, CH Score: {best_score}")
            return best_params, best_score
        
        else:
            # Fallback to sequential processing for single process
            logger.info(f"[{algorithm}] Grid Search: Using sequential processing...")
            for args in args_list:
                params, score, from_cache = _evaluate_single_param_combination(args)
                if not from_cache:
                    param_str = str(sorted(params.items()))
                    completed_params.add(param_str)
                    existing_scores[param_str] = score
                if score > best_score:
                    best_score = score
                    best_params = params
                logger.info(f"[{algorithm}] Grid Search: Completed {params}, CH Score: {score}")
            
            logger.info(f"[{algorithm}] Grid Search: UPDATED best params: {best_params}, CH Score: {best_score}")
            return best_params, best_score
    
    except KeyboardInterrupt:
        print(f"\n[{algorithm}] Grid Search was stopped by the user.")
        print(f"[{algorithm}] Returning the best results found so far.")
        # Return current best results
        return best_params, best_score
    
    except Exception as e:
        print(f"Error in grid_search_with_unsupervised_score_custom for {algorithm}: {e}")
        return None, 0.0


def grid_search_with_unsupervised_score(X, algorithm, param_grid, file_type, file_number, num_processes_for_algo=None, true_labels=None, global_known_normal_samples_pca=None, known_normal_idx=None):
    """
    Performs grid search for a given algorithm and parameter grid,
    optimizing for an unsupervised score (Calinski-Harabasz).
    """
    param_combinations = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    return grid_search_with_unsupervised_score_custom(X, algorithm, param_combinations, file_type, file_number, num_processes_for_algo, true_labels=true_labels, global_known_normal_samples_pca=global_known_normal_samples_pca, known_normal_idx=known_normal_idx)

def run_single_clustering(algorithm, X, params, aligned_original_labels=None, global_known_normal_samples_pca=None):
    """
    Helper function to run a single clustering algorithm with given parameters.
    Returns the raw cluster labels.
    """
    clusters = None  # Initialize clusters to None

    # This dictionary defines the parameters for each algorithm's wrapper function.
    # It ensures that CNI-related parameters are passed correctly if the algorithm needs them.
    base_func_args = {
        'data': None,
        'X': X,
        'aligned_original_labels': aligned_original_labels,
        'global_known_normal_samples_pca': global_known_normal_samples_pca,
        'threshold_value': 0.3,  # A default threshold for CNI
        'num_processes_for_algo': 1,  # Avoid nested parallelism
        **params
    }

    try:
        # Group 1: Standard sklearn-compatible models (do not need CNI info directly)
        if algorithm in ['Kmeans', 'GMM', 'SGMM', 'DBSCAN', 'MShift', 'Kmedoids']:
            model = None
            if algorithm == 'Kmeans':
                model = KMeans(**params)
            elif algorithm == 'Kmedoids':
                effective_params = params.copy()
                if 'method' not in effective_params: effective_params['method'] = 'pam'
                if effective_params['method'] == 'pam':
                    try:
                        model = KMedoids(**effective_params)
                    except MemoryError:
                        logger.warning(f"[Kmedoids] MemoryError with 'pam'. Falling back to 'alternate'.")
                        effective_params['method'] = 'alternate'
                        model = KMedoids(**effective_params)
                else:
                    model = KMedoids(**params)
            elif algorithm == 'GMM':
                model = GaussianMixture(**params)
            elif algorithm == 'SGMM':
                model = BayesianGaussianMixture(**params)
            elif algorithm == 'DBSCAN':
                model = DBSCAN(**params)
            elif algorithm == 'MShift':
                model = MeanShift(**params)
            
            if model:
                clusters = model.fit_predict(X)

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
                # Create a safe, mutable copy of the arguments for this specific call.
                func_args = base_func_args.copy()

                # Standardize the cluster count parameter to 'k' for all custom wrappers.
                # Grid Search might produce 'n_clusters', so we handle that case here.
                if 'n_clusters' in func_args:
                    # If 'n_clusters' exists, rename it to 'k' for consistency.
                    func_args['k'] = func_args.pop('n_clusters')
                
                # SPECIAL HANDLING FOR NeuralGas to match its specific wrapper signature
                if algorithm == 'NeuralGas':
                    # First, ensure cluster count parameter is named 'max_nodes'
                    if 'k' in func_args:
                        func_args['max_nodes'] = func_args.pop('k')
                    
                    # The wrapper expects specific arguments, not a generic dict.
                    # We need to extract them from func_args.
                    neural_gas_args = {
                        'data': func_args.get('data'),
                        'X': func_args.get('X'),
                        'aligned_original_labels': func_args.get('aligned_original_labels'),
                        'global_known_normal_samples_pca': func_args.get('global_known_normal_samples_pca'),
                        'threshold_value': func_args.get('threshold_value'),
                        'num_processes_for_algo': func_args.get('num_processes_for_algo'),
                        'n_start_nodes': func_args.get('n_start_nodes', 2),
                        'max_nodes': func_args.get('max_nodes'),
                        'step': func_args.get('step', 0.2),
                        'max_edge_age': func_args.get('max_edge_age', 50)
                    }
                    # Filter out None values to avoid passing them if they are not set
                    neural_gas_args = {k: v for k, v in neural_gas_args.items() if v is not None}
                    result = clustering_func(**neural_gas_args)
                else:
                    result = clustering_func(**func_args)
                    
                clusters = result.get('raw_cluster_labels')
            else:
                logger.error(f"[run_single_clustering] Failed to import {module_name}")

    except Exception as e:
        logger.error(f"[run_single_clustering] Error running {algorithm} with params {params}: {e}")
        # Ensure clusters is None on failure
        clusters = None

    return clusters

def save_grid_search_progress(progress_file, param_str, score):
    """Safely save Grid Search progress with file locking (Linux)"""
    for attempt in range(3): # Increased retries for grid search
        try:
            # Check if file exists to determine if we need to write header
            file_exists = os.path.exists(progress_file)
            
            with open(progress_file, 'a') as f:
                # Apply exclusive lock (Linux)
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    writer = csv.writer(f)
                    # Write header if file doesn't exist or is empty
                    if not file_exists or os.path.getsize(progress_file) == 0:
                        writer.writerow(['param_str', 'score', 'timestamp'])
                    
                    # Write progress data
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    writer.writerow([f'"{param_str}"', score, timestamp])
                    f.flush()  # Force write to disk
                finally:
                    # Release the lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return True
            
        except Exception as e:
            if attempt < 2: # Retry up to 2 times
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                print(f"[Grid Search] Retry {attempt + 1}/3 for saving progress: {e}")
            else:
                print(f"[Grid Search] Failed to save progress after 3 attempts: {e}")
                return False
    return False

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
        
        # Define reasonable upper bound for clusters.
        params = {'max_clusters': 250, 'random_state': 42}
        
        # Run clustering once.
        clusters = run_single_clustering(algorithm, X, params)
        
        if clusters is not None:
            # Evaluate with CH score (for consistency in logs, though not used for tuning here)
            ch_score = _evaluate_unsupervised_score(X, clusters, known_normal_idx)
            logger.info(f"[{algorithm}] Found {len(np.unique(clusters))} clusters with CH Score: {ch_score:.2f}")
            
            # Evaluate final Jaccard score.
            _, jaccard_score, _ = clustering_nomal_identify(
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
            if algorithm == 'Kmeans' or algorithm == 'FCM':
                elbow_best_params = {'n_clusters': optimal_k, 'random_state': 42}
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
                                                        global_known_normal_samples_pca=global_known_normal_samples_pca)

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
                    global_known_normal_samples_pca=global_known_normal_samples_pca
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
    """Test all algorithms with Jaccard-based Elbow method"""
    
    if algorithms is None:
        algorithms = ['Kmeans', 'Kmedoids', 'DBSCAN', 'Xmeans', 'Gmeans', 'MShift', 
                      'NeuralGas', 'FCM', 'CK', 'GMM', 'SGMM']
    
    print("=" * 60)
    print("JACCARD-BASED ELBOW METHOD FOR ALGORITHM SELECTION")
    print("=" * 60)
    
    results = {}
    total_start_time = time.time()
    
    try:
        for algorithm in algorithms:
            best_params, best_jaccard = apply_jaccard_elbow_method(algorithm, X, true_labels, file_type, file_number, global_known_normal_samples_pca, num_processes_for_algo, known_normal_idx=known_normal_idx)
            results[algorithm] = {
                'best_params': best_params,
                'best_jaccard': best_jaccard
            }
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
        jaccard_val = float(result.get('best_jaccard', 0.0) or 0.0)
        print(f"{i:2d}. {algorithm:12s}: Jaccard={jaccard_val:.4f}, "
              f"Best Params={result['best_params']}")
    
    print(f"\nTotal time: {total_elapsed_time:.2f}s")
    
    # Performance gap analysis
    if len(sorted_results) >= 2:
        best_algorithm = sorted_results[0][0]
        best_jaccard = sorted_results[0][1]['best_jaccard']
        second_best_jaccard = sorted_results[1][1]['best_jaccard']
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
