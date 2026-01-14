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
from utils.generate_data_hash import generate_stable_data_hash
from Clustering_Method.clustering_nomal_identify import create_global_reference_normal_samples
from Dataset_Choose_Rule.Intermediate_reset import reset_intermediate_files

# --- Import Max Score utilities ---
from utils.max_score_utils import comprehensive_optimization_max_score
from Modules.Jaccard_Elbow_Method import test_all_algorithms_with_jaccard_elbow

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

cpu_count = os.cpu_count()
num_processes = cpu_count


def preprocess_data_for_diagnosis(file_type, file_number, heterogeneous_method='scaling_label_encoding'):
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
        
    # 8. Create known_normal_idx (using the same logic as best_clustering_selector_parallel.py)
    data_hash = generate_stable_data_hash(file_type, file_number, X_shape=data_for_clustering.shape)
    global_known_normal_samples_pca, known_normal_indices = create_global_reference_normal_samples(file_type, file_number, heterogeneous_method)
    
    # Ensure data consistency between source data and known normal samples
    try:
        all_normal_samples_main = data_for_clustering[original_labels == 0]
        num_all_normal = all_normal_samples_main.shape[0]
        if num_all_normal > 1:
            sample_size = int(num_all_normal * 0.90)
            if sample_size == 0 and num_all_normal > 0: sample_size = 1
            
            np.random.seed(42)
            random_indices = np.random.choice(num_all_normal, size=sample_size, replace=False)
            consistent_known_normal_samples_pca = all_normal_samples_main[random_indices]
            main_all_normal_indices = np.where(original_labels == 0)[0]
            consistent_known_normal_indices = main_all_normal_indices[random_indices]
            logger.info(f"Created consistent known_normal_samples set. Shape: {consistent_known_normal_samples_pca.shape}")
        else:
            logger.warning("Not enough normal samples in main data. Using global reference as fallback.")
            consistent_known_normal_samples_pca = global_known_normal_samples_pca
            consistent_known_normal_indices = known_normal_indices
    except Exception as e:
        logger.error(f"Error during data consistency fix: {e}. Using original global reference.")
        consistent_known_normal_samples_pca = global_known_normal_samples_pca
        consistent_known_normal_indices = known_normal_indices

    logger.info(f"Preprocessing complete. Data shape: {data_for_clustering.shape}, Labels: {original_labels.shape}, Known Normals: {consistent_known_normal_indices.shape}")
    
    return data_for_clustering, original_labels, consistent_known_normal_samples_pca, consistent_known_normal_indices


def run_virtual_labeling_with_surrogate(algorithm, X, original_labels, known_normal_samples_pca, known_normal_idx, file_type, file_number, use_max_score=False):
    """
    Runs virtual labeling (without knowing true labels) using surrogate score optimization.
    This mimics the behavior of best_clustering_selector_parallel.py.
    """
    logger.info(f"\n===== Running Virtual Labeling for {algorithm} =====")
    
    # Convert algorithm names to match the Jaccard Elbow method format
    jaccard_algorithms_map = {
        'kmeans': 'Kmeans', 'CLARA': 'CLARA', 'GMM': 'GMM', 'SGMM': 'SGMM',
        'Gmeans': 'Gmeans', 'Xmeans': 'Xmeans', 'DBSCAN': 'DBSCAN', 'MShift': 'MShift',
        'FCM': 'FCM', 'CK': 'CK', 'NeuralGas': 'NeuralGas'
    }
    jaccard_algorithm = jaccard_algorithms_map.get(algorithm, algorithm)
    
    try:
        if use_max_score:
            logger.info("Using Max Score optimization...")
            result = comprehensive_optimization_max_score(
                jaccard_algorithm, X, original_labels, file_type, file_number,
                known_normal_samples_pca,
                num_processes_for_algo=num_processes,
                known_normal_idx=known_normal_idx
            )
        else:
            logger.info("Using Elbow Method optimization...")
            jaccard_results = test_all_algorithms_with_jaccard_elbow(
                X, original_labels, [jaccard_algorithm], file_type, file_number,
                known_normal_samples_pca,
                num_processes_for_algo=num_processes,
                known_normal_idx=known_normal_idx
            )
            result = jaccard_results.get(jaccard_algorithm, {})
        
        best_params = result.get('best_params', {})
        best_jaccard = result.get('best_jaccard', 0.0)
        
        if not best_params:
            logger.warning(f"No valid parameters found for {algorithm}")
            return None, 0.0, {}
        
        # Run the final clustering with best parameters to get the virtual labels
        from Tuning_hyperparameter.jaccard_run_single_clustering import run_single_clustering
        
        raw_cluster_labels = run_single_clustering(
            jaccard_algorithm, 
            X, 
            best_params,
            aligned_original_labels=original_labels,
            global_known_normal_samples_pca=known_normal_samples_pca
        )
        
        if raw_cluster_labels is None:
            logger.warning(f"Clustering failed for {algorithm}")
            return None, 0.0, {}
        
        # Apply CNI to get final virtual labels
        final_labels, virtual_jaccard, _ = clustering_nomal_identify(
            data_features_for_clustering=X,
            clusters_assigned=raw_cluster_labels,
            original_labels_aligned=original_labels,
            global_known_normal_samples_pca=known_normal_samples_pca,
            threshold_value=0.3,  # Using default threshold
            num_processes_for_algo=num_processes,
            data_for_clustering=X,
            known_normal_idx=known_normal_idx
        )
        
        logger.info(f"Virtual labeling completed for {algorithm}. Virtual Jaccard: {virtual_jaccard:.4f}")
        
        return final_labels, virtual_jaccard, {
            'best_params': best_params,
            'best_jaccard': best_jaccard,
            'raw_cluster_labels': raw_cluster_labels
        }
        
    except Exception as e:
        logger.error(f"Error in virtual labeling for {algorithm}: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0, {}


def run_diagnostic_vanilla(algorithm, X, original_labels, known_normal_samples_pca, known_normal_idx, file_type, file_number, use_max_score=False):
    """
    Runs comparison between virtual labeling (without knowing true labels) and actual labels.
    """
    logger.info(f"\n===== Diagnosing Algorithm: {algorithm} =====")
    
    # --- Virtual Labeling (without knowing true labels) ---
    virtual_labels, virtual_jaccard, virtual_info = run_virtual_labeling_with_surrogate(
        algorithm=algorithm, X=X, original_labels=original_labels,
        known_normal_samples_pca=known_normal_samples_pca, known_normal_idx=known_normal_idx,
        file_type=file_type, file_number=file_number, use_max_score=use_max_score
    )
    
    if virtual_labels is None:
        logger.error(f"Virtual labeling failed for {algorithm}")
        return {
            'algorithm': algorithm,
            'virtual_jaccard': 0.0,
            'actual_jaccard': 0.0,
            'jaccard_gap': 0.0,
            'virtual_params': {},
            'virtual_success': False
        }
    
    # --- Calculate actual Jaccard score (true labels vs virtual labels) ---
    try:
        # Filter out noise and invalid labels
        valid_predicted_indices = (virtual_labels != -1)
        valid_original_indices = np.isfinite(original_labels)
        valid_indices = valid_predicted_indices & valid_original_indices
        
        if len(valid_indices) == 0:
            logger.warning(f"No valid labels found for {algorithm}")
            actual_jaccard = 0.0
        else:
            original_labels_filtered = original_labels[valid_indices]
            virtual_labels_filtered = virtual_labels[valid_indices]
            
            # Calculate actual Jaccard score
            from sklearn.metrics import jaccard_score
            actual_jaccard = jaccard_score(original_labels_filtered, virtual_labels_filtered, average='binary', zero_division=0)
    
    except Exception as e:
        logger.error(f"Error calculating actual Jaccard for {algorithm}: {e}")
        actual_jaccard = 0.0
    
    # --- Analysis ---
    jaccard_gap = actual_jaccard - virtual_jaccard
    
    print(f"\nReport for {algorithm} on {file_type}:")
    print("-" * 50)
    print(f"Virtual Labeling Jaccard: {virtual_jaccard:.4f} (estimated by CNI)")
    print(f"Actual Jaccard:           {actual_jaccard:.4f} (true labels vs virtual labels)")
    print(f"Jaccard Gap:              {jaccard_gap:.4f} (actual - virtual)")
    print(f"Best Parameters:          {virtual_info.get('best_params', {})}")
    print("-" * 50)

    return {
        'algorithm': algorithm,
        'virtual_jaccard': virtual_jaccard,
        'actual_jaccard': actual_jaccard,
        'jaccard_gap': jaccard_gap,
        'virtual_params': virtual_info.get('best_params', {}),
        'virtual_success': True,
        'optimization_method': 'Max Score' if use_max_score else 'Elbow Method'
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose virtual labeling against actual labels.")
    parser.add_argument('--file_type', type=str, default="MiraiBotnet", help='Data file type')
    parser.add_argument('--file_number', type=int, default=1, help='File number')
    parser.add_argument('--algorithms', type=str, default="kmeans,CLARA,Xmeans,Gmeans,GMM,SGMM,FCM,CK,DBSCAN,MShift,NeuralGas", help='Comma-separated list of algorithms to test')
    parser.add_argument('--use_max_score', action='store_true', help='Use max score optimization instead of elbow method')
    parser.add_argument('--heterogeneous', type=str, default="scaling_label_encoding", help='Heterogeneous method')
    parser.add_argument('--reset', action='store_true', help='Reset all intermediate progress files and start fresh')
    args = parser.parse_args()

    if args.reset:
        reset_intermediate_files(args.file_type, args.file_number)
        print("All intermediate progress files have been deleted. Starting fresh...")
    
    # 1. Preprocess data exactly as in the main script
    X, y_true, known_normal_samples_pca, known_normal_idx = preprocess_data_for_diagnosis(
        args.file_type, args.file_number, args.heterogeneous
    )
    
    # 2. Run diagnostics for each algorithm
    algorithms_to_test = [algo.strip() for algo in args.algorithms.split(',')]
    all_diagnostics = []
    
    for algorithm in algorithms_to_test:
        try:
            result = run_diagnostic_vanilla(
                algorithm, X, y_true, known_normal_samples_pca, known_normal_idx,
                args.file_type, args.file_number, args.use_max_score
            )
            all_diagnostics.append(result)
        except Exception as e:
            logger.error(f"Failed to diagnose algorithm {algorithm}: {e}")
            import traceback
            traceback.print_exc()

    # 3. Print final report
    logger.info("\n\n" + "="*80)
    logger.info("              Virtual Labeling Diagnostic Report")
    logger.info("="*80)
    
    df = pd.DataFrame(all_diagnostics)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 60)

    print(df.to_string())

    logger.info("\n" + "-"*80)
    logger.info("Report Columns Explained:")
    logger.info("- virtual_jaccard: Jaccard score estimated by CNI during virtual labeling")
    logger.info("- actual_jaccard: True Jaccard score when comparing virtual labels vs actual labels")
    logger.info("- jaccard_gap: (actual - virtual). Positive means CNI underestimated, negative means overestimated")
    logger.info("- virtual_params: Best parameters found during virtual labeling optimization")
    logger.info("- virtual_success: Whether virtual labeling completed successfully")
    logger.info("- optimization_method: Whether Max Score or Elbow Method was used")
    logger.info("-" * 80)


if __name__ == '__main__':
    main()
