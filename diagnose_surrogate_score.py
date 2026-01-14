import argparse
import numpy as np
import pandas as pd
import time
import os
import sys
import logging
import copy
import importlib
from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from tqdm import tqdm # Added for progress bar

# --- Add project root to sys.path for module imports ---
# This ensures that we can import from 'Clustering_Method', 'Tuning_hyperparameter', etc.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now, import the necessary modules from the project
from best_clustering_selector_parallel import apply_labeling_logic, time_scalar_transfer, pca_func
from Heterogeneous_Method.Feature_Encoding import Heterogeneous_Feature_named_featrues
from Modules.Heterogeneous_module import choose_heterogeneous_method
from Dataset_Choose_Rule.choose_amount_dataset import file_path_line_nonnumber, file_cut
from Clustering_Method.clustering_score import evaluate_clustering_wos
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify
from Dataset_Choose_Rule.Intermediate_reset import reset_intermediate_files

# --- Surrogate Score Imports ---
# We will compare the 'f1+elkan' version
from Tuning_hyperparameter.Surrogate_score_f1_elkan import compute_surrogate_score as surrogate_score_f1_elkan


# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

cpu_count = os.cpu_count()
num_processes = cpu_count


def preprocess_data_for_diagnosis(file_type, file_number):
    """
    Mirrors the complete preprocessing pipeline from best_clustering_selector_parallel.py
    to ensure an apples-to-apples comparison.
    """
    logger.info(f"--- Preprocessing data for {file_type} #{file_number} ---")
    
    # 1. Load data
    file_path, _ = file_path_line_nonnumber(file_type, file_number)
    data = file_cut(file_type, file_path, 'all')
    data.columns = data.columns.str.strip()
    
    # 2. Apply labeling
    data = apply_labeling_logic(data, file_type)
    
    # --- This mirrors PIPELINE A ('scaling_label_encoding') ---
    # 3. Time scalar transfer
    data_for_processing = time_scalar_transfer(data.copy(), file_type)

    # 4. Feature selection
    feature_dict = Heterogeneous_Feature_named_featrues(file_type)
    features_for_clustering = feature_dict.get('categorical_features', []) + \
                              feature_dict.get('time_features', []) + \
                              feature_dict.get('packet_length_features', []) + \
                              feature_dict.get('count_features', []) + \
                              feature_dict.get('binary_features', [])
    existing_features = [f for f in features_for_clustering if f in data_for_processing.columns]
    data_for_processing = data_for_processing[existing_features + ['label']].copy()

    # 5. Data cleaning (NaN, inf)
    numerical_features = feature_dict.get('time_features', []) + \
                         feature_dict.get('packet_length_features', []) + \
                         feature_dict.get('count_features', [])
    columns_to_clean = [col for col in numerical_features if col in data_for_processing.columns]
    
    if columns_to_clean:
        initial_rows = len(data_for_processing)
        for col in columns_to_clean:
            if not pd.api.types.is_numeric_dtype(data_for_processing[col]):
                data_for_processing[col] = pd.to_numeric(data_for_processing[col], errors='coerce')
        for col in columns_to_clean:
            if not np.isfinite(data_for_processing[col].values).all():
                if np.isinf(data_for_processing[col].values).any():
                    finite_max = data_for_processing.loc[np.isfinite(data_for_processing[col]), col].max()
                    replacement_val = 0 if pd.isna(finite_max) else finite_max
                    data_for_processing[col].replace([np.inf, -np.inf], replacement_val, inplace=True)
        data_for_processing.dropna(subset=columns_to_clean, inplace=True)
        logger.info(f"Dropped {initial_rows - len(data_for_processing)} rows during cleaning.")
        
    # 6. Final scaling/encoding
    original_labels = data_for_processing['label'].to_numpy()
    X = data_for_processing.drop(columns=['label'])
    X_processed, _, _, _ = choose_heterogeneous_method(X, file_type, 'scaling_label_encoding', 'N')
    
    # 7. PCA
    pca_want = 'N' if file_type in ['DARPA98', 'DARPA', 'CICModbus23', 'CICModbus'] else 'Y'
    if pca_want == 'Y':
        data_for_clustering = pca_func(X_processed)
    else:
        data_for_clustering = X_processed.to_numpy() if hasattr(X_processed, 'to_numpy') else X_processed
        
    # 8. Create known_normal_idx (using the same logic)
    main_all_normal_indices = np.where(original_labels == 0)[0]
    sample_size = int(len(main_all_normal_indices) * 0.90)
    np.random.seed(42)
    random_indices = np.random.choice(len(main_all_normal_indices), size=sample_size, replace=False)
    known_normal_idx = main_all_normal_indices[random_indices]

    logger.info(f"Preprocessing complete. Data shape: {data_for_clustering.shape}, Labels: {original_labels.shape}, Known Normals: {known_normal_idx.shape}")
    
    return data_for_clustering, original_labels, known_normal_idx


def jaccard_oracle_score(X, labels, known_normal_idx, **kwargs):
    """
    This is the "Oracle" function. It cheats by using the true labels to calculate
    true Jaccard score using the aligned_original_labels passed in kwargs.
    """
    true_labels = kwargs.get('aligned_original_labels')
    if true_labels is None:
        logger.error("Oracle requires 'aligned_original_labels' in kwargs!")
        return 0.0
    if labels is None:
        return 0.0

    # A true oracle uses the final evaluation function directly.
    _, jaccard, _ = clustering_nomal_identify(
        data_features_for_clustering=X,
        clusters_assigned=labels,
        original_labels_aligned=true_labels,
        global_known_normal_samples_pca=None, # Not used in this diagnostic context
        threshold_value=0.3, # Using a consistent threshold
        num_processes_for_algo=num_processes,
        data_for_clustering=X,
        known_normal_idx=known_normal_idx
    )
    
    return jaccard if jaccard is not None else 0.0


def tune_with_surrogate(algorithm, X, true_labels, known_normal_idx, surrogate_function, file_type, k_max):
    """
    Helper function to tune a single clustering algorithm using a surrogate score.
    - `k_max`: The maximum number of clusters to test for k-based algorithms.
    """
    best_score = -1
    best_params = None
    best_jaccard = 0.0

    # Handle auto-tuning algorithms separately as they don't have a grid to search
    if algorithm in ['Xmeans', 'Gmeans']:
        logger.info(f"Running {algorithm} as a self-tuning algorithm (single run).")
        params = {'max_clusters': 250, 'random_state': 42}
        clusters = run_single_clustering(
            algorithm=algorithm, X=X, params=params, aligned_original_labels=true_labels
        )
        if clusters is not None:
            best_params = params
            best_score = surrogate_function(
                X=X, labels=clusters, known_normal_idx=known_normal_idx,
                file_type=file_type, aligned_original_labels=true_labels
            )
            _, best_jaccard, _ = clustering_nomal_identify(
                data_features_for_clustering=X, clusters_assigned=clusters,
                original_labels_aligned=true_labels, global_known_normal_samples_pca=None,
                threshold_value=0.3, num_processes_for_algo=num_processes,
                data_for_clustering=X, known_normal_idx=known_normal_idx
            )
        return best_params, best_jaccard, best_score

    # --- Build the full parameter grid for the search ---
    param_grid = []
    # Algorithms that tune based on number of clusters (k)
    if algorithm in ['Kmeans', 'CLARA', 'GMM', 'SGMM', 'FCM', 'CK', 'NeuralGas']:
        k_step = 5  # To keep the search manageable, test every 5 k's
        k_values = range(2, k_max + 1, k_step)
        if not list(k_values) and k_max >= 2:
            k_values = [k_max] # Ensure at least one k is tested if k_max is small
        elif not list(k_values):
            k_values = [2]
            
        logger.info(f"[{algorithm}] Searching k values: {list(k_values)}")
        
        for k in k_values:
            # Get other parameter variations for each k
            param_grid.extend(get_param_grid(algorithm, k, X))
            
    # Algorithms that have a different primary hyperparameter
    elif algorithm in ['DBSCAN', 'MShift']:
        param_grid = get_param_grid(algorithm, None, X)

    if not param_grid:
        logger.warning(f"No parameter grid for {algorithm}, skipping.")
        return {}, 0.0, -1 # Return 3 values to avoid unpack error

    logger.info(f"Tuning {algorithm} with {len(param_grid)} parameter combinations using '{surrogate_function.__name__}'...")

    for params in tqdm(param_grid, desc=f"Tuning {algorithm}"):
        # 1. Run clustering
        clusters = run_single_clustering(
            algorithm=algorithm,
            X=X,
            params=params,
            aligned_original_labels=true_labels,
            global_known_normal_samples_pca=None # Not used in this diagnostic context
        )

        if clusters is None:
            continue

        # 2. Calculate the score using the provided surrogate function
        current_score = surrogate_function(
            X=X,
            labels=clusters,
            known_normal_idx=known_normal_idx,
            file_type=file_type, # Pass file_type to the surrogate function
            aligned_original_labels=true_labels # Pass true_labels for the Oracle
        )

        # 3. Track the best score and parameters
        if current_score > best_score:
            best_score = current_score
            best_params = params
            
            # 4. Calculate the true Jaccard score for the best params so far
            _, current_jaccard, _ = clustering_nomal_identify(
                data_features_for_clustering=X,
                clusters_assigned=clusters,
                original_labels_aligned=true_labels,
                global_known_normal_samples_pca=None,
                threshold_value=0.3,
                num_processes_for_algo=num_processes,
                data_for_clustering=X,
                known_normal_idx=known_normal_idx
            )
            best_jaccard = current_jaccard

    return best_params, best_jaccard, best_score


def run_diagnostic(algorithm, X, true_labels, known_normal_idx, file_type, k_max):
    """
    Runs both Oracle-guided and Surrogate-guided tuning for a single algorithm
    and returns a comparison of the results.
    """
    logger.info(f"\n===== Diagnosing Algorithm: {algorithm} =====")
    
    # --- Oracle-Guided Tuning ---
    oracle_best_params, oracle_best_jaccard, _ = tune_with_surrogate(
        algorithm=algorithm, X=X, true_labels=true_labels,
        known_normal_idx=known_normal_idx, surrogate_function=jaccard_oracle_score,
        file_type=file_type, k_max=k_max
    )

    # --- Surrogate-Guided Tuning ---
    surrogate_best_params, surrogate_best_jaccard, best_surrogate_score_found = tune_with_surrogate(
        algorithm=algorithm, X=X, true_labels=true_labels,
        known_normal_idx=known_normal_idx, surrogate_function=surrogate_score_f1_elkan,
        file_type=file_type, k_max=k_max
    )
    #surrogate_best_score = surrogate_results['best_jaccard']
    #surrogate_best_params = surrogate_results['best_params']

    # --- Analysis ---
    # What surrogate score would the Oracle's choice have received?
    surrogate_score_of_oracle_choice = 0.0
    if oracle_best_params:
        # Run clustering with the oracle's best parameters
        oracle_choice_clusters = run_single_clustering(
            algorithm, X, oracle_best_params,
            aligned_original_labels=true_labels
        )
        if oracle_choice_clusters is not None:
            # Calculate the surrogate score for that clustering
            surrogate_score_of_oracle_choice = surrogate_score_f1_elkan(
                X=X,
                labels=oracle_choice_clusters,
                known_normal_idx=known_normal_idx,
                file_type=file_type
            )
    else:
        logger.warning(f"Oracle tuning for {algorithm} did not find any valid parameters.")

    print(f"\nReport for {algorithm} on {file_type}:")
    print("-" * 50)
    print(f"Surrogate-Guided Best Jaccard: {surrogate_best_jaccard:.4f} with params: {surrogate_best_params}")
    print(f"Oracle-Guided Best Jaccard:    {oracle_best_jaccard:.4f} with params: {oracle_best_params}")
    print("-" * 50)

    return {
        'algorithm': algorithm,
        'oracle_best_jaccard': oracle_best_jaccard,
        'oracle_best_params': oracle_best_params,
        'surrogate_best_jaccard': surrogate_best_jaccard,
        'surrogate_best_params': surrogate_best_params,
        'jaccard_loss': oracle_best_jaccard - surrogate_best_jaccard,
        'surrogate_score_of_oracle_choice': surrogate_score_of_oracle_choice,
        'best_surrogate_score_found': best_surrogate_score_found,
        'params_are_different': oracle_best_params != surrogate_best_params
    }


# =================================================================================
# == Functions Ported from Jaccard_Elbow_Method.py for standalone execution      ==
# =================================================================================

def get_param_grid(algorithm, optimal_k, X):
    """
    Generates a parameter grid for a given algorithm.
    - For k-based algos, `optimal_k` is the target k.
    - For other algos, `optimal_k` can be None for a wide search.
    """
    param_grid = []

    if algorithm == 'Kmeans':
        param_grid = [{'n_clusters': k, 'random_state': 42, 'n_init': n_init}
                      for k in [optimal_k]
                      for n_init in [10, 20, 50, 80]]
    
    elif algorithm == 'Kmedoids' or algorithm == 'CLARA': # CLARA is run instead of Kmedoids
        k_values = sorted(list(set([k for k in [optimal_k - 2, optimal_k, optimal_k + 2] if k > 1])))
        sample_multipliers = [1.0, 1.5]
        iteration_values = [3, 5]
        for k_val in k_values:
            base_samples = 40 + 2 * k_val
            for multiplier in sample_multipliers:
                num_samples = int(base_samples * multiplier)
                if X.shape[0] > 0 and num_samples > X.shape[0]:
                    num_samples = X.shape[0]
                for num_iter in iteration_values:
                    param_grid.append({
                        'k': k_val, 
                        'number_samples': num_samples, 
                        'number_iterations': num_iter
                    })

    elif algorithm == 'GMM':
        cov_types = ['full', 'tied', 'diag', 'spherical']
        reg_covars = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        param_grid = [{'n_components': optimal_k, 'random_state': 42, 'covariance_type': ct, 'reg_covar': rc}
                      for ct in cov_types
                      for rc in reg_covars]

    elif algorithm == 'SGMM':
        reg_covars = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        param_grid = [{'n_components': optimal_k, 'random_state': 42, 'covariance_type': 'spherical', 'reg_covar': rc}
                      for rc in reg_covars]

    elif algorithm == 'DBSCAN':
        # If optimal_k is None, we are in a wide search. Define a static grid.
        if optimal_k is None:
             eps_values = np.linspace(0.1, 2.0, 10)
        else: # This is the fine-tuning path around a specific k from the old logic
             eps_values = [max(0.1, optimal_k - 0.1), optimal_k, optimal_k + 0.1]
        min_samples = [5, 10] # Reduced for speed
        param_grid = [{'eps': eps, 'min_samples': ms}
                      for eps in eps_values
                      for ms in min_samples]

    elif algorithm == 'MShift':
        # This algorithm's bandwidth is data-dependent.
        # We estimate a default bandwidth and search around it.
        try:
            # Use a smaller sample for faster estimation
            X_sample = X[np.random.choice(X.shape[0], min(1000, X.shape[0]), replace=False)]
            default_bandwidth = estimate_bandwidth(X_sample, quantile=0.2)
            if default_bandwidth is None or default_bandwidth <= 0:
                default_bandwidth = 1.0 # Fallback
        except Exception:
            default_bandwidth = 1.0 # Fallback
        
        param_grid = [{'bandwidth': b} for b in [default_bandwidth * 0.5, default_bandwidth, default_bandwidth * 1.5, default_bandwidth * 2.0]]

    elif algorithm in ['FCM', 'CK', 'Xmeans', 'Gmeans', 'NeuralGas', 'CLARA']:
        param_grid = [{'n_clusters': optimal_k, 'random_state': 42}]

    elif algorithm == 'NeuralGas':
        param_grid = [
            {'n_start_nodes': 2, 'max_nodes': optimal_k, 'step': 0.1, 'max_edge_age': 50},
            {'n_start_nodes': 2, 'max_nodes': optimal_k, 'step': 0.2, 'max_edge_age': 50},
            {'n_start_nodes': 2, 'max_nodes': optimal_k, 'step': 0.2, 'max_edge_age': 80},
            {'n_start_nodes': 5, 'max_nodes': optimal_k, 'step': 0.2, 'max_edge_age': 50},
        ]
        
    return param_grid


# Cache for dynamic imports to avoid repeated imports
_import_cache = {}

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

def run_single_clustering(algorithm, X, params, aligned_original_labels=None, global_known_normal_samples_pca=None):
    """
    Helper function to run a single clustering algorithm with given parameters.
    Returns the raw cluster labels.
    (Ported from Jaccard_Elbow_Method.py)
    """
    clusters = None
    base_func_args = {
        'data': None, 'X': X, 'aligned_original_labels': aligned_original_labels,
        'global_known_normal_samples_pca': global_known_normal_samples_pca,
        'threshold_value': 0.3, 'num_processes_for_algo': num_processes,
    }
    if algorithm != 'NeuralGas':
        base_func_args.update(params)

    try:
        if algorithm in ['Kmeans', 'GMM', 'SGMM', 'DBSCAN', 'MShift']:
            model = None
            if algorithm == 'Kmeans':
                model = KMeans(**params)
            elif algorithm in ['GMM', 'SGMM']:
                reg_covar = params.get('reg_covar', 1e-5)
                model_class = GaussianMixture if algorithm == 'GMM' else BayesianGaussianMixture
                for _ in range(7):
                    try:
                        temp_params = params.copy()
                        temp_params['reg_covar'] = reg_covar
                        if algorithm == 'SGMM': temp_params['covariance_type'] = 'spherical'
                        current_model = model_class(**temp_params)
                        clusters = current_model.fit_predict(X)
                        break
                    except ValueError:
                        reg_covar *= 10
                else:
                   clusters = None
                model = None
            elif algorithm == 'DBSCAN':
                model = DBSCAN(**params)
            elif algorithm == 'MShift':
                model = MeanShift(**params)
            
            if model:
                clusters = model.fit_predict(X)

        elif algorithm in ['FCM', 'CK', 'Xmeans', 'Gmeans', 'NeuralGas', 'CLARA']:
            algo_map = {
                'FCM': 'clustering_FCM', 'CK': 'clustering_CK', 'Xmeans': 'clustering_Xmeans',
                'Gmeans': 'clustering_Gmeans', 'NeuralGas': 'clustering_NeuralGas', 'CLARA': 'clustering_CLARA'
            }
            module_name = algo_map[algorithm]
            clustering_func = dynamic_import(f"Clustering_Method.{module_name}", module_name)
            
            if clustering_func:
                func_args = base_func_args.copy()
                func_args.update(params)
                
                if algorithm in ['Gmeans', 'Xmeans', 'FCM', 'CK', 'CLARA']:
                    func_args.pop('random_state', None)

                if 'n_clusters' in func_args:
                    cluster_param = func_args.pop('n_clusters')
                    if algorithm in ['FCM', 'CK', 'Xmeans', 'Gmeans']:
                        func_args['max_clusters'] = cluster_param
                    elif algorithm == 'CLARA':
                        func_args['k'] = cluster_param
                
                if algorithm == 'NeuralGas':
                    neural_gas_args = {
                        'data': func_args.get('data'), 'X': func_args.get('X'),
                        'aligned_original_labels': func_args.get('aligned_original_labels'),
                        'global_known_normal_samples_pca': func_args.get('global_known_normal_samples_pca'),
                        'n_start_nodes': func_args.get('n_start_nodes', 2),
                        'max_nodes': func_args.get('max_nodes'),
                        'step': func_args.get('step', 0.2),
                        'max_edge_age': func_args.get('max_edge_age', 50)
                    }
                    neural_gas_args = {k: v for k, v in neural_gas_args.items() if v is not None or k == 'data'}
                    result = clustering_func(**neural_gas_args)
                else:
                    result = clustering_func(**func_args)
                clusters = result.get('raw_cluster_labels')
    except Exception as e:
        print(f"[run_single_clustering] Error running {algorithm} with params {params}: {e}")
        clusters = None

    return clusters

def main():
    parser = argparse.ArgumentParser(description="Diagnose surrogate score against an oracle.")
    parser.add_argument('--file_type', type=str, default="MiraiBotnet", help='Data file type')
    parser.add_argument('--file_number', type=int, default=1, help='File number')
    parser.add_argument('--algorithms', type=str, default="Kmeans,CLARA,Xmeans,Gmeans,GMM,SGMM,FCM,CK,DBSCAN,MShift,NeuralGas", help='Comma-separated list of algorithms to test')
    parser.add_argument('--k_max', type=int, default=350, help='Maximum number of clusters (k) to test for k-based algorithms.')
    parser.add_argument('--reset', action='store_true', help='Reset all intermediate progress files and start fresh')
    args = parser.parse_args()

    if args.reset:
        reset_intermediate_files(args.file_type, args.file_number)
        print("All intermediate progress files have been deleted. Starting fresh...")
    
    # 1. Preprocess data exactly as in the main script
    X, y_true, known_normal_idx = preprocess_data_for_diagnosis(args.file_type, args.file_number)
    
    # 2. Run diagnostics for each algorithm
    algorithms_to_test = [algo.strip() for algo in args.algorithms.split(',')]
    all_diagnostics = []
    
    for algorithm in algorithms_to_test:
        # NOTE: comprehensive_algorithm_optimization needs a small modification to accept a
        # custom surrogate function. For this PoC, we assume it's modified.
        # If not, we'd need to monkey-patch or temporarily modify the file.
        # Let's proceed assuming it can be passed as an argument.
        try:
            result = run_diagnostic(algorithm, X, y_true, known_normal_idx, args.file_type, k_max=args.k_max)
            all_diagnostics.append(result)
        except Exception as e:
            logger.error(f"Failed to diagnose algorithm {algorithm}: {e}")
            import traceback
            traceback.print_exc()

    # 3. Print final report
    logger.info("\n\n" + "="*80)
    logger.info("              Surrogate Score Diagnostic Report")
    logger.info("="*80)
    
    df = pd.DataFrame(all_diagnostics)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 60)

    print(df.to_string())

    logger.info("\n" + "-"*80)
    logger.info("Report Columns Explained:")
    logger.info("- oracle_best_jaccard: The best possible Jaccard score found for this algorithm.")
    logger.info("- surrogate_best_jaccard: The Jaccard score achieved by the params chosen by the surrogate.")
    logger.info("- jaccard_loss: (oracle - surrogate). How much performance is lost due to surrogate inaccuracy.")
    logger.info("- surrogate_score_of_oracle_choice: What the surrogate score *would have been* for the BEST params.")
    logger.info("- best_surrogate_score_found: The highest surrogate score that was actually found.")
    logger.info("  -> If (best_surrogate_score_found > surrogate_score_of_oracle_choice), the surrogate is misleading.")
    logger.info("-" * 80)


if __name__ == '__main__':
    main()
