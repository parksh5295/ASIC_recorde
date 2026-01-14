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
import importlib # Added for dynamic imports
import itertools # For max_score logic
from tqdm import tqdm # For max_score logic
from kneed import KneeLocator # For max_score logic
import hashlib # For max_score logic

# module imports
from Tuning_hyperparameter.Surrogate_score import compute_surrogate_score
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify
from Tuning_hyperparameter.jaccard_run_single_clustering import run_single_clustering
from Dataset_Choose_Rule.save_jaccard_elbow import save_jaccard_elbow_progress_parallel, get_grid_search_progress_file_path, save_grid_search_progress

# Setup logger
logger = logging.getLogger(__name__)


def _evaluate_unsupervised_score(X, clusters, known_normal_idx, file_type):
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
            score = compute_surrogate_score(X, clusters, known_normal_idx, file_type=file_type)
            return score
        else:
            # If only noise clusters are found, or it's otherwise invalid.
            return 0.0
    except Exception as e:
        print(f"Error calculating surrogate score: {e}")
        return 0.0

def _evaluate_single_elbow_k(args_tuple):
    """Helper function to evaluate a single k value in parallel for the Elbow method."""
    k, algorithm, X, original_labels, global_known_normal_samples_pca, num_processes_for_algo, data_hash, data_for_clustering, known_normal_idx = args_tuple

    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

    score = 0.0 # Changed from jaccard to a generic score
    jaccard_for_log = 0.0
    ratio_distribution = {}

    try:
        # Step 1: Construct parameters with the correct names for each algorithm
        params = {}
        # The parameter 'k' here represents the value being tested in the elbow method.
        # We must map it to the correct argument name for the target function.
        if algorithm in ['Kmeans', 'Kmedoids']:
            params['n_clusters'] = k
            params['random_state'] = 42
        elif algorithm in ['GMM', 'SGMM']:
            params['n_components'] = k
            params['random_state'] = 42
        elif algorithm in ['FCM', 'CK']:
            params['max_clusters'] = k
        elif algorithm == 'CLARA':
            params['k'] = k
        elif algorithm in ['Xmeans', 'Gmeans']:
            params['max_clusters'] = k
        elif algorithm == 'NeuralGas':
            params['max_nodes'] = k
        
        print(f"DEBUG:[_evaluate_single_elbow_k] Worker for {algorithm} k={k}, sending params: {params}")

        # Step 2: Centralized call to the clustering runner function
        clusters = run_single_clustering(
            algorithm, X, params,
            aligned_original_labels=original_labels,
            global_known_normal_samples_pca=global_known_normal_samples_pca
        )

        # Step 3: Evaluate the results if clustering was successful
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
        else:
            # Handle cases where clustering returns None (e.g., an error occurred inside)
            score = 0.0
            jaccard_for_log = 0.0

    except Exception as e:
        print(f"Error with {algorithm} k={k}: {e}")
        # Return 0 score for this k value if an error occurs
        score = 0.0
        jaccard_for_log = 0.0
    
    # Save progress using the new generic score
    save_jaccard_elbow_progress_parallel(algorithm, data_hash, k, score, jaccard_for_log, ratio_distribution)
    return k, score, jaccard_for_log


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