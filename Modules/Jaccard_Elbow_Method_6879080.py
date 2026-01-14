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
from sklearn.mixture import GaussianMixture

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
from sklearn.metrics import jaccard_score, calinski_harabasz_score
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


# Setup logger
logger = logging.getLogger(__name__)


def save_jaccard_elbow_progress_parallel(algorithm, data_hash, k, jaccard_score, ratio_distribution, max_retries=3):
    """
    Safely append progress to a CSV file from multiple processes, 
    including ratio distribution, timestamp, and retry logic.
    """
    progress_file = get_jaccard_elbow_progress_file_path(algorithm, data_hash)
    
    # Define header and row data dynamically
    header = ['param_value', 'jaccard_score']
    dist_keys = sorted(ratio_distribution.keys())
    header.extend(dist_keys)
    header.append('timestamp') # Add timestamp to header

    row_data = [k, jaccard_score]
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

def _evaluate_unsupervised_score(X, clusters):
    """
    Evaluates clustering using an unsupervised metric (Calinski-Harabasz).
    Returns the score. A higher score is better.
    Returns 0 if clustering is invalid (e.g., single cluster).
    """
    try:
        # CH score requires at least 2 labels.
        unique_labels = np.unique(clusters)
        if len(unique_labels) > 1:
            score = calinski_harabasz_score(X, clusters)
            return score
        else:
            # If only one cluster is found, it cannot be evaluated.
            return 0.0
    except Exception as e:
        print(f"Error calculating Calinski-Harabasz score: {e}")
        return 0.0

def _evaluate_single_elbow_k(args_tuple):
    """Helper function to evaluate a single k value in parallel for the Elbow method."""
    # MODIFIED: Correctly unpack the 'data_hash' argument from the tuple
    k, algorithm, X, original_labels, global_known_normal_samples_pca, num_processes_for_algo, data_hash, data_for_clustering = args_tuple

    # Set thread-related environment variables to 1 to avoid nested parallelism issues
    # This is crucial when running sklearn models in a multiprocessing pool
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

    score = 0.0 # Changed from jaccard to a generic score
    ratio_distribution = {}
    try:
        clusters = None
        num_clusters = k
        if algorithm == 'Kmeans':
            # Use joblib backend for KMeans for better performance with thread-based parallelism
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = model.fit_predict(X)
        elif algorithm == 'Kmedoids':
            model = KMedoids(n_clusters=k, random_state=42)
            clusters = model.fit_predict(X)
        elif algorithm == 'GMM':
            model = GaussianMixture(n_components=k, random_state=42, reg_covar=1e-5)
            clusters = model.fit_predict(X)
        elif algorithm == 'SGMM':
            model = GaussianMixture(n_components=k, random_state=42, covariance_type='spherical', reg_covar=1e-5)
            clusters = model.fit_predict(X)
        elif algorithm == 'FCM':
            FCM = dynamic_import("Clustering_Method.clustering_FCM", "clustering_FCM")
            if FCM:
                result = FCM(None, X, k, original_labels, global_known_normal_samples_pca=global_known_normal_samples_pca, 
                            threshold_value=0.3, num_processes_for_algo=1)
                clusters = result['raw_cluster_labels']
        elif algorithm == 'CK':
            CK = dynamic_import("Clustering_Method.clustering_CK", "clustering_CK")
            if CK:
                result = CK(None, X, k, original_labels, global_known_normal_samples_pca=global_known_normal_samples_pca, 
                            threshold_value=0.3, num_processes_for_algo=1)
                clusters = result['raw_cluster_labels']
        
        if clusters is not None:
            # --- CRITICAL CHANGE: Use unsupervised score for tuning ---
            score = _evaluate_unsupervised_score(X, clusters)
            
            # We still run CNI to get the ratio distribution for analysis, but NOT for tuning.
            _, _, final_results_df = clustering_nomal_identify(
                data_features_for_clustering=X,
                clusters_assigned=clusters,
                original_labels_aligned=original_labels,
                global_known_normal_samples_pca=global_known_normal_samples_pca,
                threshold_value=0.3,
                num_processes_for_algo=1,
                data_for_clustering=X 
            )
            if final_results_df is not None:
                ratio_distribution = final_results_df['normal_ratio'].value_counts(normalize=True).to_dict()

    except Exception as e:
        print(f"Error with {algorithm} k={k}: {e}")
        # Return 0 score for this k value if an error occurs
        score = 0.0
    
    # Save progress using the new generic score
    save_jaccard_elbow_progress_parallel(algorithm, data_hash, k, score, ratio_distribution)
    return k, score


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
    params, algorithm, X_local, completed_params, existing_scores, progress_file, data_hash = args

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
        # Run clustering with the given parameters
        clusters = run_single_clustering(algorithm, X_local, params)

        # --- CRITICAL CHANGE: Use unsupervised score for tuning ---
        if clusters is not None:
            score = _evaluate_unsupervised_score(X_local, clusters)
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
    
    if os.path.exists(progress_file):
        try:
            # Read entire file at once for better performance
            with open(progress_file, 'r', newline='', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Skip header and process all lines efficiently
            for line in lines[1:]:  # Skip header
                row = line.strip().split(',')
                if len(row) >= 1:
                    try:
                        param_value = float(row[0])
                        completed_values.add(param_value)
                    except ValueError:
                        continue  # Skip invalid rows
            
            print(f"[Jaccard Elbow Progress] Loaded {len(completed_values)} completed parameter values from {progress_file}")
        except Exception as e:
            print(f"[Jaccard Elbow Progress] Error loading progress file {progress_file}: {e}")
    
    return completed_values

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

def cluster_count_elbow_method(X, true_labels, algorithm, max_k=250, file_type=None, file_number=None, global_known_normal_samples_pca=None, num_processes_for_algo=None, data_for_clustering=None):
    """Jaccard-based Elbow method for cluster count algorithms (Parallel Version)"""
    # Generate data hash for progress tracking
    data_hash = generate_stable_data_hash(file_type, file_number, X.shape)
    
    k_values = list(range(2, max_k + 1, 5))
    scores_final = []
    
    # Load existing progress (this function returns a SET of completed k_values)
    completed_k_values = load_jaccard_elbow_progress(algorithm, data_hash)

    # Now, load the actual scores for the completed values into a DICT
    existing_scores = {}
    progress_file = get_jaccard_elbow_progress_file_path(algorithm, data_hash)
    if completed_k_values:
        print(f"[{algorithm}] Found existing progress: {len(completed_k_values)} k-values already completed")
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader, None)  # Skip header safely
                    for row in reader:
                        if len(row) >= 2:
                            try:
                                param_value = float(row[0])
                                jaccard_value = float(row[1])
                                if param_value in completed_k_values:
                                    existing_scores[param_value] = jaccard_value
                            except (ValueError, IndexError):
                                continue # Skip malformed rows
            except Exception as e:
                print(f"[{algorithm}] Error loading existing scores: {e}")

    k_values_to_test = [k for k in k_values if k not in completed_k_values]
    
    if not k_values_to_test:
        print(f"[{algorithm}] All k-values already processed.")
        scores_final = [existing_scores.get(k, 0.0) for k in k_values]
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
            (k, algorithm, X, true_labels, global_known_normal_samples_pca, num_processes_for_algo, data_hash, data_for_clustering)
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
        new_scores = {k: score for k, score in results}
        existing_scores.update(new_scores)
        
        # Reconstruct the full list of scores in order
        scores_final = [existing_scores.get(k, 0.0) for k in k_values]

    if not scores_final:
        print(f"[{algorithm}] No scores were calculated.")
        return 2, 0.0

    # Find elbow point
    optimal_k = find_elbow_point(k_values, scores_final)
    best_score = max(scores_final) if scores_final else 0.0
    
    print(f"[{algorithm}] Optimal k={optimal_k}, Best CH-Score={best_score:.4f}")
    
    return optimal_k, best_score

def dbscan_eps_elbow_method(X, true_labels, min_samples=10, file_type=None, file_number=None, global_known_normal_samples_pca=None, num_processes_for_algo=None):
    """Jaccard-based Elbow method for DBSCAN eps parameter"""
    # Generate data hash for progress tracking
    if file_type and file_number:
        data_hash = get_existing_hash_for_file_type(file_type, file_number)
    else:
        data_hash = generate_data_hash(X)
    
    eps_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    jaccard_scores = []
    
    # Load existing progress
    completed_eps_values = load_jaccard_elbow_progress("DBSCAN", data_hash)
    if completed_eps_values:
        print(f"[DBSCAN] Found existing progress: {len(completed_eps_values)} eps values already completed")
    
    print(f"[DBSCAN] Testing {len(eps_values)} eps values with min_samples={min_samples}...")
    
    try:
        for eps in tqdm(eps_values, desc="DBSCAN Elbow"):
            # Skip if already completed
            if eps in completed_eps_values:
                continue
            try:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                pred_labels = dbscan.fit_predict(X)
                
                # Check if clustering was successful
                if len(np.unique(pred_labels)) > 1:  # More than 1 cluster
                    jaccard = jaccard_score(true_labels, pred_labels, average='weighted')
                else:
                    jaccard = 0.0  # Single cluster or failed clustering
                    
                jaccard_scores.append(jaccard)
                
                # Save progress after each eps value
                save_jaccard_elbow_progress("DBSCAN", eps, jaccard, data_hash)
                
            except Exception as e:
                print(f"Error with DBSCAN eps={eps}: {e}")
                jaccard_scores.append(0.0)
                # Save error case as well
                save_jaccard_elbow_progress("DBSCAN", eps, 0.0, data_hash)
    
    except KeyboardInterrupt:
        print(f"\n[DBSCAN] The job was stopped by the user.")
        print(f"[DBSCAN] Interim saved file: Dataset_ex/progress_tracking/jaccard_elbow_{data_hash}_DBSCAN_progress.csv")
        print(f"[DBSCAN] When restarting, it will resume from the point where it was interrupted.")
        # Return current best results
        if jaccard_scores:
            optimal_eps = find_elbow_point(eps_values, jaccard_scores)
            best_jaccard = max(jaccard_scores)
            return optimal_eps, best_jaccard
        else:
            return 0.5, 0.0
    
    # Find elbow point
    optimal_eps = find_elbow_point(eps_values, jaccard_scores)
    best_jaccard = max(jaccard_scores) if jaccard_scores else 0.0
    
    print(f"[DBSCAN] Optimal eps={optimal_eps}, Best Jaccard={best_jaccard:.4f}")
    
    return optimal_eps, best_jaccard

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

def mean_shift_quantile_elbow_method(X, true_labels, n_samples=500, file_type=None, file_number=None, global_known_normal_samples_pca=None, num_processes_for_algo=None):
    """Jaccard-based Elbow method for Mean Shift quantile parameter"""
    # Generate data hash for progress tracking
    if file_type and file_number:
        data_hash = get_existing_hash_for_file_type(file_type, file_number)
    else:
        data_hash = generate_data_hash(X)
    
    quantile_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    jaccard_scores = []
    
    # Load existing progress
    completed_quantiles = load_jaccard_elbow_progress("Mean Shift", data_hash)
    if completed_quantiles:
        print(f"[Mean Shift] Found existing progress: {len(completed_quantiles)} quantile values already completed")
    
    print(f"[Mean Shift] Testing {len(quantile_values)} quantile values with n_samples={n_samples}...")
    
    try:
        for quantile in tqdm(quantile_values, desc="Mean Shift Elbow"):
            # Skip if already completed
            if quantile in completed_quantiles:
                continue
            try:
                # Use Mean Shift with CNI for Jaccard Elbow Method
                from Clustering_Method.clustering_MShift import clustering_MShift
                try:
                    # Step 1: Perform Mean Shift clustering
                    result = clustering_MShift(None, X, true_labels, global_known_normal_samples_pca=global_known_normal_samples_pca, 
                                             threshold_value=0.3, num_processes_for_algo=1 # Prevent nested parallelism
                                             )
                    clusters = result['raw_cluster_labels']
                    
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
                    save_jaccard_elbow_progress("Mean Shift", quantile, jaccard, data_hash)
                    continue
                except Exception as e:
                    print(f"Error with Mean Shift + CNI: {e}, falling back to sklearn only")
                    # Fallback to sklearn KMeans if Mean Shift fails
                    from sklearn.cluster import KMeans
                    model = KMeans(n_clusters=10, random_state=42, n_init=10)
                    pred_labels = model.fit_predict(X)
                
                # Check if clustering was successful
                if len(np.unique(pred_labels)) > 1:  # More than 1 cluster
                    jaccard = jaccard_score(true_labels, pred_labels, average='weighted')
                else:
                    jaccard = 0.0  # Single cluster or failed clustering
                    
                jaccard_scores.append(jaccard)
                
                # Save progress after each quantile value
                save_jaccard_elbow_progress("Mean Shift", quantile, jaccard, data_hash)
                continue
            
            except Exception as e:
                print(f"Error with Mean Shift quantile={quantile}: {e}")
                jaccard_scores.append(0.0)
                # Save error case as well
                save_jaccard_elbow_progress("Mean Shift", quantile, 0.0, data_hash)
    
    except KeyboardInterrupt:
        print(f"\n[Mean Shift] The job was stopped by the user.")
        print(f"[Mean Shift] Interim saved file: Dataset_ex/progress_tracking/jaccard_elbow_{data_hash}_Mean Shift_progress.csv")
        print(f"[Mean Shift] When restarting, it will resume from the point where it was interrupted.")
        # Return current best results
        if jaccard_scores:
            optimal_quantile = find_elbow_point(quantile_values, jaccard_scores)
            best_jaccard = max(jaccard_scores)
            return optimal_quantile, best_jaccard
        else:
            return 0.2, 0.0
    
    # Find elbow point
    optimal_quantile = find_elbow_point(quantile_values, jaccard_scores)
    best_jaccard = max(jaccard_scores) if jaccard_scores else 0.0
    
    print(f"[Mean Shift] Optimal quantile={optimal_quantile}, Best Jaccard={best_jaccard:.4f}")
    
    return optimal_quantile, best_jaccard

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

def grid_search_with_unsupervised_score_custom(X, algorithm, param_combinations, file_type, file_number, num_processes_for_algo=None):
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
            params, algorithm, X, completed_params, existing_scores, progress_file, data_hash
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


def grid_search_with_unsupervised_score(X, algorithm, param_grid, file_type, file_number, num_processes_for_algo=None):
    """
    Performs grid search for a given algorithm and parameter grid,
    optimizing for an unsupervised score (Calinski-Harabasz).
    """
    param_combinations = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    return grid_search_with_unsupervised_score_custom(X, algorithm, param_combinations, file_type, file_number, num_processes_for_algo)

def run_single_clustering(algorithm, X, params):
    """
    Helper function to run a single clustering algorithm with given parameters.
    Returns the final labels and the number of clusters.
    """
    if algorithm == 'Kmeans':
        model = KMeans(**params)
        clusters = model.fit_predict(X)
    
    elif algorithm == 'Kmedoids':
        model = KMedoids(**params)
        clusters = model.fit_predict(X)
    
    elif algorithm == 'GMM':
        model = GaussianMixture(**params)
        clusters = model.fit_predict(X)
    
    elif algorithm == 'SGMM':
        model = GaussianMixture(**params)
        clusters = model.fit_predict(X)
    
    elif algorithm == 'FCM':
        FCM = dynamic_import("Clustering_Method.clustering_FCM", "clustering_FCM")
        if FCM:
            model = FCM(**params)
            clusters = model.fit_predict(X)
        else:
            return None
    
    elif algorithm == 'CK':
        CK = dynamic_import("Clustering_Method.clustering_CK", "clustering_CK")
        if CK:
            model = CK(**params)
            clusters = model.fit_predict(X)
        else:
            return None
    
    elif algorithm == 'Xmeans':
        XMeans = dynamic_import("Clustering_Method.clustering_Xmeans", "XMeansWrapper")
        if XMeans:
            model = XMeans(**params)
            clusters = model.fit_predict(X)
        else:
            return None
    
    elif algorithm == 'Gmeans':
        GMeans = dynamic_import("Clustering_Method.clustering_Gmeans", "GMeans")
        if GMeans:
            model = GMeans(**params)
            clusters = model.fit_predict(X)
        else:
            return None
    
    elif algorithm == 'DBSCAN':
        model = DBSCAN(**params)
        clusters = model.fit_predict(X)
    
    elif algorithm == 'MShift':
        model = MeanShift(**params)
        clusters = model.fit_predict(X)
    
    elif algorithm == 'NeuralGas':
        NeuralGas = dynamic_import("Clustering_Method.clustering_NeuralGas", "NeuralGasWithParams")
        if NeuralGas:
            model = NeuralGas(**params)
            clusters = model.fit_predict(X)
        else:
            return None
    
    else:
        return None
    
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

def comprehensive_algorithm_optimization(algorithm, X, true_labels, file_type, file_number, global_known_normal_samples_pca, num_processes_for_algo=None):
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
            ch_score = _evaluate_unsupervised_score(X, clusters)
            logger.info(f"[{algorithm}] Found {len(np.unique(clusters))} clusters with CH Score: {ch_score:.2f}")
            
            # Evaluate final Jaccard score.
            _, jaccard_score, _ = clustering_nomal_identify(
                data_features_for_clustering=X,
                clusters_assigned=clusters,
                original_labels_aligned=true_labels,
                global_known_normal_samples_pca=global_known_normal_samples_pca,
                threshold_value=0.3, # Use default threshold
                num_processes_for_algo=1,
                data_for_clustering=X
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
        logger.info(f"[{algorithm}] Finding optimal hyperparameter using CH-Score-based Elbow method...")
        
        optimal_k, elbow_ch_score = cluster_count_elbow_method(
            X=X,
            true_labels=true_labels,
            algorithm=algorithm,
            max_k=250,  # Explicitly set max_k
            file_type=file_type,
            file_number=file_number,
            global_known_normal_samples_pca=global_known_normal_samples_pca,
            num_processes_for_algo=num_processes_for_algo,
            data_for_clustering=X
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
            
            if elbow_best_params:
                elbow_clusters = run_single_clustering(algorithm, X, elbow_best_params)
                if elbow_clusters is not None:
                    _, elbow_jaccard_score, _ = clustering_nomal_identify(
                        data_features_for_clustering=X, clusters_assigned=elbow_clusters,
                        original_labels_aligned=true_labels, global_known_normal_samples_pca=global_known_normal_samples_pca,
                        threshold_value=0.3, num_processes_for_algo=1, data_for_clustering=X
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
                best_params_from_grid, best_ch_score_from_grid = grid_search_with_unsupervised_score(X, algorithm, param_grid, file_type, file_number, num_processes_for_algo)
            
            elif algorithm == 'Kmedoids':
                param_grid = {
                    'n_clusters': [optimal_k],
                    'random_state': [42],
                    'method': ['pam', 'alternate']
                }
                best_params_from_grid, best_ch_score_from_grid = grid_search_with_unsupervised_score(X, algorithm, param_grid, file_type, file_number, num_processes_for_algo)

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
                best_params_from_grid, best_ch_score_from_grid = grid_search_with_unsupervised_score_custom(X, algorithm, all_combinations, file_type, file_number, num_processes_for_algo)

            elif algorithm == 'SGMM':
                param_grid = {
                    'n_components': [optimal_k],
                    'random_state': [42],
                    'covariance_type': ['spherical'], # SGMM is GMM with spherical covariance
                    'reg_covar': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
                }
                best_params_from_grid, best_ch_score_from_grid = grid_search_with_unsupervised_score(X, algorithm, param_grid, file_type, file_number, num_processes_for_algo)
                
            elif algorithm == 'DBSCAN':
                # DBSCAN: eps and min_samples need Grid Search
                # We use the 'optimal_k' from elbow as the starting 'eps' value
                param_grid = {
                    'eps': [max(0.1, optimal_k - 0.1), optimal_k, optimal_k + 0.1],
                    'min_samples': [2, 4, 6, 8, 10]
                }
                best_params_from_grid, best_ch_score_from_grid = grid_search_with_unsupervised_score(X, algorithm, param_grid, file_type, file_number, num_processes_for_algo)

            elif algorithm == 'MShift':
                # Mean Shift: bandwidth needs Grid Search
                # We use 'optimal_k' from elbow as the starting 'bandwidth' value
                param_grid = {'bandwidth': [optimal_k * 0.1, optimal_k * 0.5, optimal_k, optimal_k * 1.5]}
                best_params_from_grid, best_ch_score_from_grid = grid_search_with_unsupervised_score(X, algorithm, param_grid, file_type, file_number, num_processes_for_algo)

            elif algorithm in ['FCM', 'CK']:
                # For these algorithms, Elbow result is sufficient.
                if algorithm == 'FCM':
                    best_params_from_grid = {'n_clusters': optimal_k, 'random_state': 42}
                else: # CK
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
                best_params_from_grid, best_ch_score_from_grid = grid_search_with_unsupervised_score_custom(X, algorithm, param_combinations, file_type, file_number, num_processes_for_algo)

            # --- Step 2b: Evaluate Jaccard for Grid Search Result ---
            if best_params_from_grid:
                logger.info(f"[{algorithm}] Evaluating Jaccard score for Grid Search's best params...")
                grid_search_clusters = run_single_clustering(algorithm, X, best_params_from_grid)
                if grid_search_clusters is not None:
                    _, grid_search_jaccard_score, _ = clustering_nomal_identify(
                        data_features_for_clustering=X, clusters_assigned=grid_search_clusters,
                        original_labels_aligned=true_labels, global_known_normal_samples_pca=global_known_normal_samples_pca,
                        threshold_value=0.3, num_processes_for_algo=1, data_for_clustering=X
                    )
                    logger.info(f"[{algorithm}] Grid Search Result Jaccard: {grid_search_jaccard_score:.4f} with params: {best_params_from_grid}")
                else:
                    logger.warning(f"[{algorithm}] Clustering failed for Grid Search params.")
            else:
                logger.info(f"[{algorithm}] Grid Search did not find any valid parameters.")
        
        # --- Step 3: Final Selection ---
        if best_ch_score_from_grid > elbow_ch_score:
            logger.info(f"[{algorithm}] Selecting Grid Search params based on CH Score ({best_ch_score_from_grid:.2f} > {elbow_ch_score:.2f})")
            best_params = best_params_from_grid
            best_jaccard = grid_search_jaccard_score
        else:
            logger.info(f"[{algorithm}] Selecting Elbow params based on CH Score ({elbow_ch_score:.2f} >= {best_ch_score_from_grid:.2f})")
            best_params = elbow_best_params
            best_jaccard = elbow_jaccard_score

    end_time = time.time()
    logger.info(f"[{algorithm}] Comprehensive optimization completed in {end_time - start_time:.2f}s")
    logger.info(f"[{algorithm}] Final best_jaccard: {best_jaccard}, Final best_params: {best_params}")

    return {'best_params': best_params, 'best_jaccard': best_jaccard}

def apply_jaccard_elbow_method(algorithm, X, true_labels, file_type=None, file_number=None, global_known_normal_samples_pca=None, num_processes_for_algo=None):
    """Apply comprehensive optimization (Jaccard Elbow + Grid Search) for each algorithm"""
    return comprehensive_algorithm_optimization(algorithm, X, true_labels, file_type, file_number, global_known_normal_samples_pca, num_processes_for_algo)

def test_all_algorithms_with_jaccard_elbow(X, true_labels, algorithms=None, file_type=None, file_number=None, global_known_normal_samples_pca=None, num_processes_for_algo=None):
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
            best_params, best_jaccard = apply_jaccard_elbow_method(algorithm, X, true_labels, file_type, file_number, global_known_normal_samples_pca, num_processes_for_algo)
            results[algorithm] = {
                'best_params': best_params,
                'best_jaccard': best_jaccard
            }
    except KeyboardInterrupt:
        print(f"\nThe entire algorithm test was stopped by the user.")
        print(f"Returning results from {len(results)} algorithms completed so far.")
        # Continue with current results
    
    total_elapsed_time = time.time() - total_start_time
    
    # Sort results by Jaccard score
    sorted_results = sorted(results.items(), key=lambda x: x[1]['best_jaccard'], reverse=True)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS (Sorted by Jaccard Score)")
    print("=" * 60)
    
    for i, (algorithm, result) in enumerate(sorted_results, 1):
        print(f"{i:2d}. {algorithm:12s}: Jaccard={result['best_jaccard']:.4f}, "
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
