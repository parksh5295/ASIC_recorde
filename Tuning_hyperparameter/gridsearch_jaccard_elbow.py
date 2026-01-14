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

# module imports
from utils.generate_data_hash import generate_stable_data_hash
from Dataset_Choose_Rule.save_jaccard_elbow import get_grid_search_progress_file_path
from Evaluation.evaluate_jaccard_elbow import _evaluate_single_param_combination


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
