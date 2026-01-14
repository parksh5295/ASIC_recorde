# Input and output is parameter_dict
# The GridSearchCV model is manually implemented: Evaluate based on silhouette score
'''
Output: Dictionaries by Clustering algorithm
xmeans_result = best_results['Xmeans']  ->  {'best_params': {'max_clusters': 50}, 'all_params': {parameter_dict}, 'silhouette_score': 0.78, 'davies_bouldin_score': 0.42}
best_xmeans_params = best_results['Xmeans']['best_params']  ->  {'max_clusters': 50}
'''

import numpy as np
import importlib
import multiprocessing # Added for parallel processing
import os # For os.cpu_count()
import csv
import sys
from datetime import datetime
import hashlib
from sklearn.model_selection import GridSearchCV
from itertools import product
from sklearn.metrics import make_scorer, silhouette_score, davies_bouldin_score, f1_score, accuracy_score, calinski_harabasz_score
from sklearn.cluster import KMeans, DBSCAN
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify
from utils.class_row import nomal_class_data
import joblib # Added for parallel_backend


# === Progress Tracking Functions ===

def generate_data_hash(X):
    """Generate a unique hash for the dataset to distinguish different datasets."""
    try:
        # Create a hash based on dataset characteristics
        data_info = f"{X.shape}_{X.dtype}_{np.mean(X):.6f}_{np.std(X):.6f}_{X.flat[0]:.6f}"
        return hashlib.md5(data_info.encode()).hexdigest()[:8]
    except Exception:
        # Fallback to a simple hash if there's any issue
        return hashlib.md5(str(X.shape).encode()).hexdigest()[:8]

def get_grid_search_progress_file_path(clustering_algorithm, data_hash):
    """Get the progress file path for Grid Search."""
    os.makedirs("../Dataset_ex/progress_tracking", exist_ok=True)
    return f"../Dataset_ex/progress_tracking/grid_search_{data_hash}_{clustering_algorithm}_progress.csv"

def load_grid_search_progress(clustering_algorithm, data_hash):
    """Load completed parameter combinations from progress file."""
    progress_file = get_grid_search_progress_file_path(clustering_algorithm, data_hash)
    completed_combinations = set()
    
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    if len(row) >= 1:
                        # Convert parameter combination back to tuple for comparison
                        param_str = row[0]
                        completed_combinations.add(param_str)
            print(f"[Grid Search Progress] Loaded {len(completed_combinations)} completed parameter combinations from {progress_file}")
        except Exception as e:
            print(f"[Grid Search Progress] Error loading progress file {progress_file}: {e}")
    
    return completed_combinations

def save_grid_search_progress(clustering_algorithm, param_combination, score, db_score, f1, acc, data_hash):
    """Save a completed parameter combination to progress file."""
    progress_file = get_grid_search_progress_file_path(clustering_algorithm, data_hash)
    
    try:
        # Convert parameter combination to string for storage
        param_str = str(sorted(param_combination.items())) if isinstance(param_combination, dict) else str(param_combination)
        
        # Check if file exists to determine if we need to write header
        file_exists = os.path.exists(progress_file)
        
        with open(progress_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['param_combination', 'score', 'db_score', 'f1', 'acc', 'timestamp'])
            writer.writerow([param_str, score, db_score, f1, acc, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            
    except Exception as e:
        print(f"[Grid Search Progress] Error saving progress to {progress_file}: {e}")

def load_completed_grid_search_results(clustering_algorithm, data_hash):
    """Load all completed results from progress file."""
    progress_file = get_grid_search_progress_file_path(clustering_algorithm, data_hash)
    completed_results = []
    
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    if len(row) >= 5:
                        param_str, score, db_score, f1, acc = row[0], float(row[1]), float(row[2]), float(row[3]), float(row[4])
                        # Convert param_str back to dict
                        try:
                            # Simple parsing for dict format
                            param_dict = eval(param_str) if param_str.startswith('[') else {}
                            completed_results.append((param_dict, score, db_score, f1, acc))
                        except:
                            completed_results.append(({}, score, db_score, f1, acc))
            print(f"[Grid Search Progress] Loaded {len(completed_results)} completed results from {progress_file}")
        except Exception as e:
            print(f"[Grid Search Progress] Error loading completed results from {progress_file}: {e}")
    
    return completed_results


# === Top-level Model Creator Functions (for pickling) ===

def _creator_xmeans(params_from_grid, default_parameter_dict, num_processes_for_algo_local):
    XMeansWrapper = dynamic_import("Clustering_Method.clustering_Xmeans", "XMeansWrapper")
    model_constructor_params = {
        'random_state': default_parameter_dict.get('random_state', 42),
        'n_init': default_parameter_dict.get('n_init', 10),
        **params_from_grid,
        'num_processes_for_algo': num_processes_for_algo_local
    }
    return XMeansWrapper(**model_constructor_params)

def _creator_gmeans(params_from_grid, default_parameter_dict, num_processes_for_algo_local):
    GMeans = dynamic_import("Clustering_Method.clustering_Gmeans", "GMeans")
    model_constructor_params = {
        'random_state': default_parameter_dict.get('random_state', 42),
        'n_init': default_parameter_dict.get('n_init', 10),
        **params_from_grid,
        'num_processes_for_algo': num_processes_for_algo_local
    }
    return GMeans(**model_constructor_params)

def _creator_dbscan(params_from_grid, default_parameter_dict, num_processes_for_algo_local):
    n_jobs_for_dbscan = -1 if num_processes_for_algo_local == 0 else num_processes_for_algo_local
    if num_processes_for_algo_local is None: n_jobs_for_dbscan = 1
    model_constructor_params = {
        **params_from_grid,
        'n_jobs': n_jobs_for_dbscan
    }
    return DBSCAN(**model_constructor_params)

def _creator_mshift(params_from_grid, default_parameter_dict, num_processes_for_algo_local):
    MeanShiftWithDynamicBandwidth = dynamic_import("Clustering_Method.clustering_MShift", "MeanShiftWithDynamicBandwidth")
    model_constructor_params = {
        **params_from_grid,
        'num_processes_for_algo': num_processes_for_algo_local
    }
    return MeanShiftWithDynamicBandwidth(**model_constructor_params)

def _creator_neuralgas(params_from_grid, default_parameter_dict, num_processes_for_algo_local):
    NeuralGasWithParams = dynamic_import("Clustering_Method.clustering_NeuralGas", "NeuralGasWithParams")
    # For NeuralGas, params_from_grid is already the full dictionary including num_processes_for_algo
    return NeuralGasWithParams(**params_from_grid)

def _creator_cannwknn(params_from_grid, default_parameter_dict, num_processes_for_algo_local):
    CANNwKNN = dynamic_import("Clustering_Method.clustering_CANNwKNN", "CANNwKNN")
    model_constructor_params = {
        **params_from_grid,
        'num_processes_for_algo': num_processes_for_algo_local
    }
    return CANNwKNN(**model_constructor_params)


# Moved helper for Kmeans parallel processing to top level
def _evaluate_kmeans_n_init_worker(task_args):
    # Unpack num_processes_for_algo_local from task_args
    X_k, n_clusters_k, parameter_dict_k, n_init_k, num_processes_for_algo_local = task_args
    
    # Determine n_jobs for joblib.parallel_backend context
    '''
    n_jobs_for_kmeans = -1 if num_processes_for_algo_local == 0 else num_processes_for_algo_local
    if num_processes_for_algo_local is None: # Default if None is passed through
        n_jobs_for_kmeans = 1
    '''
    n_jobs_for_context = 1 # Default to 1 (sequential)
    if num_processes_for_algo_local == 0: # 0 means use all available cores
        n_jobs_for_context = -1
    elif num_processes_for_algo_local is not None and num_processes_for_algo_local > 0:
        n_jobs_for_context = num_processes_for_algo_local
        
    kmeans = KMeans(
        n_clusters=n_clusters_k,
        random_state=parameter_dict_k.get('random_state', 42), # Use get for safety
        n_init=n_init_k
        # n_jobs argument is not taken by KMeans constructor directly
    )
    
    labels_k = None
    # Use joblib.parallel_backend to control parallelism for KMeans n_init runs
    with joblib.parallel_backend('loky', n_jobs=n_jobs_for_context):
        labels_k = kmeans.fit_predict(X_k)
        
    current_score_k = -1.0 # Initialize as float
    if labels_k is not None and len(set(labels_k)) > 1:
        try:
            current_score_k = calinski_harabasz_score(X_k, labels_k)
        except ValueError: 
            current_score_k = -1.0 
    return n_init_k, current_score_k


# Helper function for Grid_search_all parallel execution
def _evaluate_param_set_all(param_set_args):
    """Evaluates a single parameter set for Grid_search_all."""
    # Unpack arguments, including num_processes_for_algo
    param_set, clustering_algorithm, param_keys_local, X_local, data_local, default_parameter_dict, num_processes_for_algo_local = param_set_args   # No create_model_func_base

    if clustering_algorithm == 'NeuralGas':
        params_from_grid = param_set # param_set is already the full dict for NeuralGas
    else:
        params_from_grid = dict(zip(param_keys_local, param_set))
    
    model = None
    try:
        # model = create_model_func_base(params_from_grid)
        # === MODIFIED: Use the new top-level creators instead of a pickled local function ===
        if clustering_algorithm in ['Xmeans', 'xmeans']:
            model = _creator_xmeans(params_from_grid, default_parameter_dict, num_processes_for_algo_local)
        elif clustering_algorithm in ['Gmeans', 'gmeans']:
            model = _creator_gmeans(params_from_grid, default_parameter_dict, num_processes_for_algo_local)
        elif clustering_algorithm == 'DBSCAN':
            model = _creator_dbscan(params_from_grid, default_parameter_dict, num_processes_for_algo_local)
        elif clustering_algorithm == 'MShift':
            model = _creator_mshift(params_from_grid, default_parameter_dict, num_processes_for_algo_local)
        elif clustering_algorithm == 'NeuralGas':
            model = _creator_neuralgas(params_from_grid, default_parameter_dict, num_processes_for_algo_local)
        elif clustering_algorithm in ['CANNwKNN', 'CANN']:
            model = _creator_cannwknn(params_from_grid, default_parameter_dict, num_processes_for_algo_local)
        else:
            raise ValueError(f"Unknown clustering algorithm '{clustering_algorithm}' in worker.")

    except TypeError as te:
        print(f"[ERROR _evaluate_param_set_all] TypeError creating model for {clustering_algorithm} with grid params {params_from_grid} and num_processes={num_processes_for_algo_local}: {te}")
        return params_from_grid, -1.0, float('inf'), -1.0, -1.0 # Return default error scores
    except Exception as e:
        print(f"[ERROR _evaluate_param_set_all] Exception creating model for {clustering_algorithm} with grid params {params_from_grid}: {e}")
        return params_from_grid, -1.0, float('inf'), -1.0, -1.0 # Return default error scores

    # Initialize scores - THIS BLOCK IS NOW RESTORED AND MOVED AFTER SUCCESSFUL MODEL CREATION
    labels = None 
    score = -1.0 # Silhouette for most, F1 for CANN
    db_score = float('inf') 
    f1 = -1.0
    acc = -1.0

    # Model fitting and scoring - THIS TRY-EXCEPT BLOCK IS NOW RESTORED
    try:
        # print(f"[DEBUG] Worker evaluating: {params} for {clustering_algorithm}") # Optional: worker debug
        if clustering_algorithm in ['CANNwKNN', 'CANN']:
            if data_local is None:
                print(f"[ERROR _evaluate_param_set_all] Full 'data' (data_local) is None for {clustering_algorithm} but required for its evaluation logic.")
                # Scores remain at their default error values, already initialized
            elif 'label' not in data_local.columns:
                 print(f"[ERROR _evaluate_param_set_all] 'label' column missing in data_local for {clustering_algorithm} evaluation.")
            elif len(data_local['label']) != X_local.shape[0]:
                 print(f"[ERROR _evaluate_param_set_all] Length of 'label' column in data_local does not match X_local for {clustering_algorithm}.")                 
            else:
                # Assuming CANNwKNN.fit_predict takes X_reduced (X_local) and full data (data_local)
                # And its internal evaluation or the subsequent evaluate_clustering_with_known_benign handles labels.
                cluster_labels_fit = model.fit_predict(X_local, data_local) 
                num_unique_labels = len(set(cluster_labels_fit))

                if num_unique_labels > 0: # Need at least one cluster 
                    # Pass aligned original labels (data_local['label']) for evaluation
                    f1_calc, acc_calc = evaluate_clustering_with_known_benign(
                        data_local, X_local, cluster_labels_fit, num_unique_labels, data_local['label']
                    )
                    score = f1_calc # For CANN, primary score is F1
                    f1 = f1_calc
                    acc = acc_calc
                # else: scores remain at default error values

        else: # Other algorithms (KMeans, DBSCAN, XMeans, GMeans, MShift, NeuralGas)
            labels_fit = model.fit_predict(X_local)
            labels = labels_fit # Store for potential debugging
            if len(set(labels_fit)) < 2: # Calinski-Harabasz and DB need at least 2 clusters
                # scores remain at -1.0 and inf (default error values)
                pass 
            else:
                # Calculate Calinski-Harabasz and Davies-Bouldin scores
                # evaluate_clustering now handles internal ValueErrors for individual metrics
                current_calinski, current_db = evaluate_clustering(X_local, labels_fit)
                score = current_calinski
                db_score = current_db

    except Exception as e:
        print(f"[ERROR _evaluate_param_set_all] Error during model fit/evaluation for {clustering_algorithm} with params {params_from_grid}: {e}")
        # score, db_score, etc., will retain their initialized default error values set before this try block

    # print(f"[DEBUG] Worker finished: {params}, score: {score}, db: {db_score}, f1: {f1}, acc: {acc}") # Optional
    return params_from_grid, score, db_score, f1, acc


# Dynamic import functions (using importlib)
def dynamic_import(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def Grid_search_Kmeans(X, n_clusters, parameter_dict=None, num_processes_for_algo=None): # Added num_processes_for_algo
    # Maintain complete parameter_dict for compatibility
    if parameter_dict is None:
        parameter_dict = {
            'random_state': 42,
            'n_init': 30,
            'max_clusters': 1000,
            'tol': 1e-4,
            'eps': 0.5,
            'count_samples': 5,
            'quantile': 0.2,
            'n_samples': 500,
            'n_start_nodes': 2,
            'max_nodes': 50,
            'step': 0.2,
            'max_edge_age': 50,
            'epochs': 300,
            'batch_size': 256,
            'n_neighbors': 5,
            'reg_covar': 1e-6
        }

    n_init_values = list(range(2, 102, 3))
    best_score = -1.0 # Initialize as float
    best_params = None

    # Parallelize this loop
    tasks_kmeans = []
    for n_init_val_loop in n_init_values:
        tasks_kmeans.append((X, n_clusters, parameter_dict, n_init_val_loop, num_processes_for_algo))

    if tasks_kmeans:
        pool_processes_kmeans = 1
        if num_processes_for_algo == 0:
            pool_processes_kmeans = os.cpu_count()
            if pool_processes_kmeans is None: pool_processes_kmeans = 1
        elif num_processes_for_algo is not None and num_processes_for_algo > 0:
            pool_processes_kmeans = num_processes_for_algo
        else: # Default or invalid (e.g. None or negative)
            cpu_cores_kmeans = os.cpu_count()
            if cpu_cores_kmeans:
                pool_processes_kmeans = max(1, cpu_cores_kmeans // 2)
            else: # Fallback if os.cpu_count() returns None
                pool_processes_kmeans = max(1, multiprocessing.cpu_count() // 2 if multiprocessing.cpu_count() else 1)
        
        pool_processes_kmeans = min(pool_processes_kmeans, len(tasks_kmeans)) 
        if pool_processes_kmeans == 0 and len(tasks_kmeans) > 0: pool_processes_kmeans = 1
        
        print(f"[Grid_search_Kmeans] Using {pool_processes_kmeans} processes for {len(tasks_kmeans)} n_init values (num_processes_for_algo={num_processes_for_algo}).")
        try:
            with multiprocessing.Pool(processes=pool_processes_kmeans) as pool:
                # Use the top-level worker function
                results_kmeans = pool.map(_evaluate_kmeans_n_init_worker, tasks_kmeans)
            
            for n_init_res, score_res in results_kmeans:
                if score_res > best_score:
                    best_score = score_res
                    best_params = {'n_init': n_init_res}
        except Exception as e:
            print(f"Error during parallel Kmeans n_init search: {e}. Attempting sequential fallback.")
            best_score_seq = -1.0
            best_params_seq = None
            print("Performing sequential Kmeans n_init search as fallback...")
            for task_args_seq in tasks_kmeans: # Iterate through the prepared tasks
                # Call the worker function directly for sequential execution
                # num_processes_for_algo is already part of task_args_seq
                n_init_seq, score_seq = _evaluate_kmeans_n_init_worker(task_args_seq)
                if score_seq > best_score_seq:
                    best_score_seq = score_seq
                    best_params_seq = {'n_init': n_init_seq}
            
            # Use sequential results if they are better or if parallel results were incomplete
            if best_params_seq is not None:
                if best_params is None or best_score_seq > best_score: # If parallel failed or sequential is better
                    best_score = best_score_seq
                    best_params = best_params_seq
                    print("Sequential Kmeans fallback provided results.")
                elif best_params is not None: # Parallel had some result, but ensure it's used if it was better
                    print("Parallel Kmeans results will be used (were better or equal to sequential fallback).")
            else:
                print("Warning: Kmeans sequential fallback also failed or produced no valid results.")
            
            if best_params is None:
                 print("Warning: Kmeans n_init search (parallel and sequential) failed to find best_params. Defaulting n_init.")
                 best_params = {'n_init': parameter_dict.get('n_init', 10)} # Default n_init if all fails

    # Ensure best_params is not None before updating (in case all init values failed)
    if best_params is None:
        best_params = {'n_init': parameter_dict.get('n_init', 10)} # Ensure a default n_init
        print(f"[Grid_search_Kmeans] No best n_init found, defaulting to {best_params['n_init']}")
        best_score = -1.0 # Reflect that this is not from a successful grid search point

    best_param_full = parameter_dict.copy()
    best_param_full.update(best_params)
    best_param_full['calinski_harabasz_score_from_n_init_tuning'] = best_score 
    return best_param_full


def evaluate_clustering(X, labels):
    """Functions to evaluate clustering performance (Calinski-Harabasz Score & Davies-Bouldin Score)"""
    if len(set(labels)) < 2:
        return -1.0, float('inf') 
    
    calinski_score = -1.0
    db_score_val = float('inf')
    try:
        calinski_score = calinski_harabasz_score(X, labels)
    except ValueError:
        # print(f"[WARN evaluate_clustering] Calinski-Harabasz score calculation failed for labels: {np.unique(labels, return_counts=True)}")
        pass # calinski_score remains -1.0
    try:
        db_score_val = davies_bouldin_score(X, labels)
    except ValueError:
        # print(f"[WARN evaluate_clustering] Davies-Bouldin score calculation failed for labels: {np.unique(labels, return_counts=True)}")
        pass # db_score_val remains inf
    return calinski_score, db_score_val


def evaluate_clustering_with_known_benign(data_orig_df, X_reduced_features, model_assigned_labels, num_model_clusters, original_labels_aligned_with_X):
    """Evaluate clustering using F1 and Accuracy after CNI-like processing."""
    # Ensure original_labels_aligned_with_X is a 1D array
    if original_labels_aligned_with_X.ndim > 1:
         original_labels_aligned_with_X = original_labels_aligned_with_X.squeeze()

    # Call CNI to get 0/1 labels (0 for normal-like, 1 for anomalous-like)
    # CNI requires features used for clustering (X_reduced_features), original true labels (original_labels_aligned_with_X),
    # model assigned cluster IDs (model_assigned_labels), and the number of clusters from the model.
    # global_known_normal_samples_pca for CNI will be handled by its internal fallback if not provided, 
    # using original_labels_aligned_with_X to find known normals in the current data slice.
    inferred_cni_labels = clustering_nomal_identify(
        data_features_for_clustering=X_reduced_features, 
        original_labels_aligned=original_labels_aligned_with_X, 
        clusters_assigned=model_assigned_labels, 
        num_total_clusters=num_model_clusters
        # num_processes_for_algo could be passed to CNI if CNI is modified to use it for its pool
    )

    # Ground truth is the original_labels_aligned_with_X (0 for benign/normal, 1 for attack/anomalous)
    ground_truth = original_labels_aligned_with_X

    f1 = -1.0
    acc = -1.0
    try:
        # Ensure both arrays are 1D and have compatible types for metric functions
        if ground_truth.ndim == 1 and inferred_cni_labels.ndim == 1 and len(ground_truth) == len(inferred_cni_labels):
            f1 = f1_score(ground_truth, inferred_cni_labels, zero_division=0)
            acc = accuracy_score(ground_truth, inferred_cni_labels)
        else:
            print(f"[WARN evaluate_clustering_with_known_benign] Ground truth or CNI labels are not 1D or have mismatched lengths.")
            print(f"  GT shape: {ground_truth.shape}, CNI labels shape: {inferred_cni_labels.shape}")
    except ValueError as e_metrics:
        print(f"[WARN evaluate_clustering_with_known_benign] ValueError during F1/Acc calculation: {e_metrics}")
        # f1, acc remain -1.0
        
    return f1, acc


def Grid_search_all(X, clustering_algorithm, parameter_dict=None, data=None, num_processes_for_algo=None): 
    # Generate data hash for progress tracking
    data_hash = generate_data_hash(X)
    
    if parameter_dict is None:
        parameter_dict = {
            'random_state': 42,
            'n_init': 30,
            'max_clusters': 1000,
            'tol': 1e-4,
            'eps': 0.5,
            'count_samples': 5,
            'quantile': 0.2,
            'n_samples': 500,
            'n_start_nodes': 2,
            'max_nodes': 50,
            'step': 0.2,
            'max_edge_age': 50,
            'epochs': 300,
            'batch_size': 256,
            'n_neighbors': 5,
            'reg_covar': 1e-6
        }
    else: # Ensure essential keys exist if parameter_dict is provided
        # --- ROBUST: Also check if 'tol' key exists or is None, and provide a default ---
        if parameter_dict.get('tol') is None:
            parameter_dict['tol'] = 1e-4 # Provide default if missing or None
            
        defaults_for_safety = {
            'random_state': 42, 'n_init': 10, 'max_clusters': 100, 'tol': 1e-4,
            'eps': 0.5, 'count_samples': 5, 'quantile': 0.2, 'n_samples': 200,
            'n_start_nodes': 2, 'max_nodes': 30, 'step': 0.2, 'max_edge_age': 30,
            'epochs': 100, 'batch_size': 128, 'n_neighbors': 5, 'reg_covar': 1e-6
        }
        for key, val in defaults_for_safety.items():
            parameter_dict.setdefault(key, val)

    # Load previous progress
    completed_combinations = load_grid_search_progress(clustering_algorithm, data_hash)
    
    best_results_for_algo = {} 
    print(f"\\n{clustering_algorithm} Performing clustering grid search...")

    param_grid = {}
    #create_model = None 
    param_combinations = [] 
    param_keys = [] # Initialize param_keys here

    # START: Algorithm-specific param_grid and create_model setup
    if clustering_algorithm in ['Xmeans', 'xmeans']:
        XMeansWrapper = dynamic_import("Clustering_Method.clustering_Xmeans", "XMeansWrapper")
        param_grid = {'max_clusters': list(range(30, 301, 30))} # Adjusted range
        '''
        def create_model_local(params_local_grid): 
            model_constructor_params = { 
                'random_state': parameter_dict.get('random_state', 42), 
                'n_init': parameter_dict.get('n_init',10), # XMeansWrapper uses n_init for internal KMeans
                **params_local_grid, 
                'num_processes_for_algo': num_processes_for_algo 
            }
            return XMeansWrapper(**model_constructor_params)
        create_model = create_model_local
        '''

    elif clustering_algorithm in ['Gmeans', 'gmeans']:
        GMeans = dynamic_import("Clustering_Method.clustering_Gmeans", "GMeans")
        log_range = np.logspace(-6, -1, num=10) # Adjusted range
        # lin_range = np.linspace(min(log_range), max(log_range), num=3) # Simplified
        # combined_range = sorted(list(set(np.concatenate((log_range, lin_range)))))
        '''
        param_grid = {'max_clusters': list(range(30, 301, 30)), 'tol': log_range} # Adjusted range
        def create_model_local(params_local_grid):
            model_constructor_params = { 
                'random_state': parameter_dict.get('random_state', 42), 
                'n_init': parameter_dict.get('n_init',10), # GMeans uses n_init for internal KMeans
                **params_local_grid, 
                'num_processes_for_algo': num_processes_for_algo 
            }
            return GMeans(**model_constructor_params)
        create_model = create_model_local
        '''
        param_grid = {'max_clusters': list(range(30, 301, 30)), 'tol': log_range} # Adjusted range

    elif clustering_algorithm == 'DBSCAN':
        param_grid = {'eps': np.arange(0.1, 2.1, 0.05), 'min_samples': list(range(2, 20, 2))}
        '''
        def create_model_local(params_local_grid):
            n_jobs_for_dbscan = -1 if num_processes_for_algo == 0 else num_processes_for_algo
            if num_processes_for_algo is None: n_jobs_for_dbscan = 1
            model_constructor_params = {
                **params_local_grid,
                'n_jobs': n_jobs_for_dbscan
            }
            return DBSCAN(**model_constructor_params)
        create_model = create_model_local 
        '''

    elif clustering_algorithm == 'MShift':
        MeanShiftWithDynamicBandwidth = dynamic_import("Clustering_Method.clustering_MShift", "MeanShiftWithDynamicBandwidth")
        param_grid = {'quantile': np.arange(0.05, 0.51, 0.05), 'n_samples': list(range(50, 501, 30))}
        '''
        def create_model_local(params_local_grid):
            model_constructor_params = {
                **params_local_grid, 
                'num_processes_for_algo': num_processes_for_algo
            }
            return MeanShiftWithDynamicBandwidth(**model_constructor_params)
        create_model = create_model_local
        '''

    elif clustering_algorithm == 'NeuralGas':
        NeuralGasWithParams = dynamic_import("Clustering_Method.clustering_NeuralGas", "NeuralGasWithParams")
        param_combinations = []
        n = X.shape[0]
        estimated_nodes = int(np.sqrt(n / 2)) 
        if estimated_nodes < 5: estimated_nodes = 5 

        max_nodes_candidates = sorted(list(set([
            max(5, int(estimated_nodes * 0.5)),
            max(5, estimated_nodes),
            max(5, int(estimated_nodes * 1.5)),
            max(5, int(estimated_nodes * 2.0)), # Added another larger candidate
            20, 50 # Include some fixed common values
        ])))
        # Ensure max_nodes are not excessively large for very small n, or cap them.
        # Max nodes should ideally not exceed n.
        max_nodes_candidates = [mn for mn in max_nodes_candidates if mn <= n and mn >=2]
        if not max_nodes_candidates: max_nodes_candidates = [min(max(n,2), 20)] # Fallback if all filters out

        # Define candidates for other parameters
        n_start_nodes_candidates = [2, 3]
        step_candidates = [0.1, 0.2, 0.3]
        # max_edge_age will be relative to max_nodes

        for mn_val in max_nodes_candidates:
            for ns_val in n_start_nodes_candidates:
                if ns_val >= mn_val : continue # n_start_nodes should be less than max_nodes
                for step_val in step_candidates:
                    # Generate max_edge_age candidates based on current mn_val
                    # Ensure max_edge_age is reasonable (e.g., not too small, not excessively large)
                    mea_candidates = sorted(list(set([
                        max(10, int(mn_val * 0.5)), 
                        max(10, mn_val), 
                        max(10, int(mn_val * 1.5))
                    ])))
                    for mea_val in mea_candidates:
                        current_params = {
                            'n_start_nodes': ns_val,
                            'max_nodes': mn_val,
                            'step': step_val,
                            'max_edge_age': mea_val,
                            'num_processes_for_algo': num_processes_for_algo # Add num_processes_for_algo
                        }
                        param_combinations.append(current_params)
        
        if not param_combinations: # Fallback if no combinations were generated
            print("[Warning Grid_search_all NeuralGas] No dynamic params generated, using a default set.")
            raw_param_sets = [
                {'n_start_nodes': 2, 'max_nodes': min(max(n,2), 50), 'step': 0.2, 'max_edge_age': 50},
                {'n_start_nodes': 2, 'max_nodes': min(max(n,2), 20), 'step': 0.1, 'max_edge_age': 30}
            ]
            param_combinations = [{**ps, 'num_processes_for_algo': num_processes_for_algo} for ps in raw_param_sets]
        
        '''
        def create_model_local(params_local_grid_dict): 
            return NeuralGasWithParams(**params_local_grid_dict) 
        create_model = create_model_local
        '''
        param_keys = [] # Already initialized, but good to be clear it's not used like others here

    elif clustering_algorithm in ['CANNwKNN', 'CANN']:
        CANNwKNN = dynamic_import("Clustering_Method.clustering_CANNwKNN", "CANNwKNN")
        param_grid = {
            'epochs': [100, 200, 300],
            'batch_size': [128, 256, 512],
            'n_neighbors': [3, 5, 7]
        }
        '''
        def create_model_local(params_local_grid):
            model_constructor_params = {
                **params_local_grid, 
                'num_processes_for_algo': num_processes_for_algo # Pass if CANNwKNN handles it
            }
            return CANNwKNN(**model_constructor_params)
        create_model = create_model_local
        '''
    
    # The redundant 'elif clustering_algorithm == KMeans' block that defined create_model_local for KMeans HAS BEEN REMOVED.
    # KMeans is handled by the special section below.

    else:
        # This 'else' is now for algorithms NOT KMeans and NOT explicitly configured above.
        # This case should ideally not be hit if Data_Labeling calls Grid_search_all only for algos defined here or KMeans.
        print(f"Clustering algorithm '{clustering_algorithm}' is not KMeans and not explicitly configured for param_grid in Grid_search_all. Using default parameters.")
        return {
            'best_params': {}, # No grid params tuned
            'all_params': parameter_dict.copy(), # Return the initial full dict
            'calinski_harabasz_score_from_grid': -1.0,
            'davies_bouldin_score_from_grid': float('inf'),
            'f1_score_from_grid': -1.0,
            'accuracy_score_from_grid': -1.0
        }

    # Generate param_combinations from param_grid if not NeuralGas (which creates it directly)
    if clustering_algorithm != 'NeuralGas':
        if not param_grid: 
            print(f"[Warning Grid_search_all] param_grid is empty for {clustering_algorithm} (and it's not NeuralGas or KMeans). No combinations to test.")
            param_combinations = [] 
        else:
            param_keys = list(param_grid.keys())
            param_values = list(param_grid.values())
            param_combinations = list(product(*param_values))

    # Filter out completed parameter combinations
    if param_combinations and completed_combinations:
        original_count = len(param_combinations)
        param_combinations = [combo for combo in param_combinations 
                            if str(sorted(dict(zip(param_keys, combo)).items())) not in completed_combinations]
        filtered_count = len(param_combinations)
        print(f"[Grid Search Progress] Filtered {original_count - filtered_count} completed combinations. {filtered_count} combinations remaining.")

    if not param_combinations and clustering_algorithm != 'Kmeans':
        # Check if all combinations were already completed
        if completed_combinations:
            print(f"All parameter combinations for {clustering_algorithm} were already completed. Loading results from progress file.")
            completed_results = load_completed_grid_search_results(clustering_algorithm, data_hash)
            if completed_results:
                # Find best result from completed ones
                best_params_algo = None
                best_score_algo = -1.0
                best_db_score_algo = float('inf')
                best_f1_algo = -1.0
                best_acc_algo = -1.0
                
                for params_res, score_res, db_score_res, f1_res, acc_res in completed_results:
                    current_primary_score = f1_res if clustering_algorithm in ['CANNwKNN', 'CANN'] else score_res
                    
                    if current_primary_score > best_score_algo: 
                        best_score_algo = current_primary_score
                        best_params_algo = params_res
                        best_db_score_algo = db_score_res
                        best_f1_algo = f1_res
                        best_acc_algo = acc_res
                    elif abs(current_primary_score - best_score_algo) < 1e-9 and db_score_res < best_db_score_algo:
                        best_params_algo = params_res 
                        best_db_score_algo = db_score_res
                        best_f1_algo = f1_res 
                        best_acc_algo = acc_res
                
                if best_params_algo is not None:
                    final_algo_parameter_dict = parameter_dict.copy()
                    final_algo_parameter_dict.update(best_params_algo)
                    return {
                        'best_params': best_params_algo,
                        'all_params': final_algo_parameter_dict,
                        'calinski_harabasz_score_from_grid': best_score_algo,
                        'davies_bouldin_score_from_grid': best_db_score_algo,
                        'f1_score_from_grid': best_f1_algo,
                        'accuracy_score_from_grid': best_acc_algo
                    }
        
        print(f"No parameter combinations to evaluate for {clustering_algorithm} after processing param_grid. Returning defaults.")
        return {
            'best_params': {}, # No grid params tuned
            'all_params': parameter_dict.copy(),
            'calinski_harabasz_score_from_grid': -1.0,
            'davies_bouldin_score_from_grid': float('inf'),
            'f1_score_from_grid': -1.0,
            'accuracy_score_from_grid': -1.0
        }

    # Special handling for KMeans: call Grid_search_Kmeans for n_init tuning.
    if clustering_algorithm == 'Kmeans':
        # n_clusters should be determined before this, e.g., by Elbow method, and be in parameter_dict.
        n_clusters_for_kmeans = parameter_dict.get('n_clusters', X.shape[0] // 100 if X.shape[0] > 200 else max(2, X.shape[0] // 2 if X.shape[0] > 3 else 2)) 
        if n_clusters_for_kmeans < 2: n_clusters_for_kmeans = 2
        
        # n_jobs_final_kmeans is not strictly needed here as Grid_search_Kmeans handles n_jobs for its KMeans runs.
        # The result from Grid_search_Kmeans will contain the best n_init and its corresponding Calinski-Harabasz score.
        print(f"Running Grid_search_Kmeans for n_init tuning with n_clusters={n_clusters_for_kmeans}...")
        best_param_full_kmeans = Grid_search_Kmeans(X, n_clusters_for_kmeans, parameter_dict.copy(), num_processes_for_algo=num_processes_for_algo)
        
        final_kmeans_score = best_param_full_kmeans.get('calinski_harabasz_score_from_n_init_tuning', -1.0)
        # Davies-Bouldin is not calculated by Grid_search_Kmeans currently.
        # If needed, a separate run of KMeans with best n_init would be required here.
        final_db_score = float('inf') 

        best_results_for_algo = {
            'best_params': {'n_init': best_param_full_kmeans.get('n_init')}, # Only n_init is tuned here
            'all_params': best_param_full_kmeans, 
            'calinski_harabasz_score_from_grid': final_kmeans_score,
            'davies_bouldin_score_from_grid': final_db_score, 
            'f1_score_from_grid': -1.0, 
            'accuracy_score_from_grid': -1.0
        }
        return best_results_for_algo

    # --- Common parallel execution for algorithms OTHER THAN KMeans ---
    best_score_algo = -1.0
    best_params_algo = None 
    best_db_score_algo = float('inf')
    best_f1_algo = -1.0
    best_acc_algo = -1.0

    tasks_all = []
    for param_set_item in param_combinations:
        tasks_all.append((param_set_item, clustering_algorithm, param_keys, X, data, parameter_dict.copy(), num_processes_for_algo))

    if tasks_all:
        pool_processes_all = 1 # Default to 1 if logic below fails or num_processes_for_algo is None/invalid
        if num_processes_for_algo == 0:
            pool_processes_all = os.cpu_count()
            if pool_processes_all is None: pool_processes_all = 1
        elif num_processes_for_algo is not None and num_processes_for_algo > 0:
            pool_processes_all = num_processes_for_algo
        else: 
            cpu_cores_all = os.cpu_count()
            if cpu_cores_all:
                pool_processes_all = max(1, cpu_cores_all // 2)
            else:
                pool_processes_all = max(1, multiprocessing.cpu_count() // 2 if multiprocessing.cpu_count() else 1)

        pool_processes_all = min(pool_processes_all, len(tasks_all)) 
        if pool_processes_all <= 0 and len(tasks_all) > 0: pool_processes_all = 1 # Ensure at least 1 if tasks exist
        
        print(f"[Grid_search_all] Algorithm: {clustering_algorithm}. Using {pool_processes_all} processes for {len(tasks_all)} param combinations (num_processes_for_algo={num_processes_for_algo}).")
        
        try:
            with multiprocessing.Pool(processes=pool_processes_all) as pool:
                results_all = pool.map(_evaluate_param_set_all, tasks_all)
            
            try:
                for params_res, score_res, db_score_res, f1_res, acc_res in results_all:
                    # Save progress for each completed parameter combination
                    save_grid_search_progress(clustering_algorithm, params_res, score_res, db_score_res, f1_res, acc_res, data_hash)
                    
                    current_primary_score = f1_res if clustering_algorithm in ['CANNwKNN', 'CANN'] else score_res
                    
                    if current_primary_score > best_score_algo: 
                        best_score_algo = current_primary_score
                        best_params_algo = params_res
                        best_db_score_algo = db_score_res
                        best_f1_algo = f1_res
                        best_acc_algo = acc_res
                    elif abs(current_primary_score - best_score_algo) < 1e-9 and db_score_res < best_db_score_algo: # Tie-breaking with DB score (use tolerance for float comparison)
                        best_params_algo = params_res 
                        best_db_score_algo = db_score_res
                        best_f1_algo = f1_res 
                        best_acc_algo = acc_res
                        
            except KeyboardInterrupt:
                print(f"\n[Grid Search for {clustering_algorithm}] The job was stopped by the user.")
                print(f"[Grid Search for {clustering_algorithm}] Interim saved file: Dataset_ex/progress_tracking/grid_search_{data_hash}_{clustering_algorithm}_progress.csv")
                print(f"[Grid Search for {clustering_algorithm}] When restarting, it will resume from the point where it was interrupted.")
                # Terminate pool and return current results
                pool.terminate()
                pool.join()
                # Continue with current results
        
        except Exception as e_pool_all:
            print(f"[ERROR Grid_search_all] Error during parallel processing for {clustering_algorithm}: {e_pool_all}. Returning default params.")
            # best_params_algo will be None, leading to fallback below.
            pass # Let it fall through to the None check for best_params_algo

    # Construct the final parameter dictionary for the algorithm
    final_algo_parameter_dict = parameter_dict.copy()
    if best_params_algo is not None:
        final_algo_parameter_dict.update(best_params_algo)
    else: 
        print(f"[Warning Grid_search_all] For {clustering_algorithm}, no best parameters found from grid. Using initial/default parameters.")
        # --- FIX: Ensure 'tol' has a default value for Gmeans on fallback ---
        if clustering_algorithm in ['Gmeans', 'gmeans', 'Xmeans', 'xmeans'] and final_algo_parameter_dict.get('tol') is None:
            final_algo_parameter_dict['tol'] = 1e-4 # Default tolerance
        # best_params_algo remains None, scores remain at their default error values.

    best_results_for_algo = {
        'best_params': best_params_algo if best_params_algo is not None else {}, 
        'all_params': final_algo_parameter_dict, 
        'calinski_harabasz_score_from_grid': best_score_algo if clustering_algorithm not in ['CANNwKNN', 'CANN'] else -1.0,
        'davies_bouldin_score_from_grid': best_db_score_algo,
        'f1_score_from_grid': best_f1_algo if clustering_algorithm in ['CANNwKNN', 'CANN'] else -1.0,
        'accuracy_score_from_grid': best_acc_algo if clustering_algorithm in ['CANNwKNN', 'CANN'] else -1.0
    }

    # Clean up progress file after successful completion
    if best_params_algo is not None:
        try:
            progress_file = get_grid_search_progress_file_path(clustering_algorithm, data_hash)
            if os.path.exists(progress_file):
                os.remove(progress_file)
                print(f"[Grid Search Progress] Cleaned up progress file: {progress_file}")
        except Exception as e:
            print(f"[Grid Search Progress] Warning: Could not clean up progress file: {e}")

    return best_results_for_algo

def _evaluate_single_neural_gas_combination(args):
    """Evaluate a single Neural Gas parameter combination"""
    params, X_local, completed_params, existing_scores, progress_file = args
    
    # Check if already completed
    param_str = str(sorted(params.items()))
    if param_str in completed_params:
        return params, existing_scores.get(param_str, 0.0), True
    
    try:
        # Import Neural Gas class and silhouette_score
        from Clustering_Method.clustering_NeuralGas import NeuralGasWithParams
        from sklearn.metrics import silhouette_score
        
        # Create model with current parameters
        model = NeuralGasWithParams(**params)
        pred_labels = model.fit_predict(X_local)
        
        # Check for single cluster or failed clustering
        if len(np.unique(pred_labels)) > 1:
            # Use silhouette score for clustering quality
            try:
                score = silhouette_score(X_local, pred_labels)
            except:
                score = 0.0
        else:
            score = 0.0  # Single cluster or failed clustering
        
        # Save progress
        import os
        import csv
        from datetime import datetime
        
        os.makedirs(os.path.dirname(progress_file), exist_ok=True)
        with open(progress_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not os.path.exists(progress_file) or os.path.getsize(progress_file) == 0:
                writer.writerow(['param_str', 'silhouette_score', 'timestamp'])
            writer.writerow([param_str, score, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        
        return params, score, False
        
    except Exception as e:
        print(f"[Neural Gas] Error with params {params}: {e}")
        return params, 0.0, False

def grid_search_neural_gas_custom(X, param_combinations, num_processes_for_algo=None):
    """Custom Grid Search for Neural Gas with pre-generated parameter combinations"""
    import multiprocessing
    from tqdm import tqdm
    import os
    import csv
    from datetime import datetime
    from sklearn.metrics import silhouette_score
    
    # Set default value for num_processes_for_algo if None
    if num_processes_for_algo is None:
        num_processes_for_algo = multiprocessing.cpu_count()
    
    best_score = 0.0
    best_params = None
    
    print(f"[Neural Gas] Custom Grid Search: Testing {len(param_combinations)} parameter combinations...")
    
    # Generate data hash for progress tracking
    data_hash = f"{hash(str(X.shape) + str(X.dtype))}"
    
    # Load existing progress
    progress_file = f"Dataset_ex/progress_tracking/neural_gas_custom_{data_hash}_progress.csv"
    completed_params = set()
    existing_scores = {}
    
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 2:
                        param_str = row[0]
                        score_value = float(row[1])
                        completed_params.add(param_str)
                        existing_scores[param_str] = score_value
            print(f"[Neural Gas] Custom Grid Search: Found {len(completed_params)} completed parameter combinations")
        except Exception as e:
            print(f"[Neural Gas] Custom Grid Search: Error loading existing progress: {e}")
    else:
        print(f"[Neural Gas] Custom Grid Search: No existing progress file found")
    
    # Determine number of processes for parallel processing
    num_processes = min(num_processes_for_algo, len(param_combinations), multiprocessing.cpu_count())
    print(f"[Neural Gas] Custom Grid Search: Using {num_processes} processes for {len(param_combinations)} parameter combinations")
    
    # Prepare arguments for parallel processing
    args_list = [(params, X, completed_params, existing_scores, progress_file) for params in param_combinations]
    
    try:
        # Use parallel processing if more than 1 process is available
        if num_processes > 1:
            print(f"[Neural Gas] Custom Grid Search: Starting parallel processing with {num_processes} processes...")
            
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = list(tqdm(
                    pool.imap(_evaluate_single_neural_gas_combination, args_list),
                    total=len(args_list),
                    desc="Neural Gas Custom Grid Search"
                ))
            
            # Process results
            for params, score, is_existing in results:
                if score > best_score:
                    best_score = score
                    best_params = params
                    print(f"[Neural Gas] Custom Grid Search: UPDATED best params: {params}, Score: {score:.4f}")
                elif not is_existing:
                    print(f"[Neural Gas] Custom Grid Search: Completed {params}, Score: {score:.4f}")
        
        else:
            # Fallback to sequential processing for single process
            print(f"[Neural Gas] Custom Grid Search: Using sequential processing...")
            
            for args in tqdm(args_list, desc="Neural Gas Custom Grid Search"):
                params, score, is_existing = _evaluate_single_neural_gas_combination(args)
                if score > best_score:
                    best_score = score
                    best_params = params
                    print(f"[Neural Gas] Custom Grid Search: UPDATED best params: {params}, Score: {score:.4f}")
                elif not is_existing:
                    print(f"[Neural Gas] Custom Grid Search: Completed {params}, Score: {score:.4f}")
        
        print(f"[Neural Gas] Custom Grid Search completed. Best params: {best_params}, Best Score: {best_score:.4f}")
        
    except Exception as e:
        print(f"[Neural Gas] Custom Grid Search error: {e}")
        return None
    
    return best_params