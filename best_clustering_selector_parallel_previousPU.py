#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Best Clustering Algorithm Selector - Parallel Version
Run all clustering algorithms and select the algorithm with the highest Jaccard coefficient to save the results
Used with run_paral.sh to process multiple datasets simultaneously
Based on Data_Labeling.py with all core features included
"""

# Essential imports
import argparse
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


# Platform-specific import for file locking
import sys
IS_LINUX = sys.platform == "linux"
if IS_LINUX:
    try:
        import fcntl
    except ImportError:
        IS_LINUX = False
        print("[WARN] fcntl module not found, file locking in parallel writes will be disabled. This is expected on non-Linux systems.")

# Set NUMEXPR_MAX_THREADS to prevent threading issues
os.environ['NUMEXPR_MAX_THREADS'] = '128'


''' # To compensate for differences in AI-GPU server environments
# Force single-threaded BLAS for reproducibility across different servers
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['BLIS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

# Disable CUDA for NumPy/Scikit-learn if available
os.environ['CUDA_VISIBLE_DEVICES'] = ''
'''

# Core functionality imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import jaccard_score

from Dataset_Choose_Rule.choose_amount_dataset import file_path_line_nonnumber, file_cut
from definition.Anomal_Judgment import anomal_judgment_nonlabel, anomal_judgment_label
from utils.time_transfer import time_scalar_transfer
from Modules.Heterogeneous_module import choose_heterogeneous_method
from Heterogeneous_Method.separate_group_mapping import map_intervals_to_groups
from Modules.PCA import pca_func
from utils.cluster_adjust_mapping import cluster_mapping
from utils.apply_labeling import apply_labeling_logic
from Clustering_Method.clustering_score import evaluate_clustering, evaluate_clustering_wos
from Clustering_Method.clustering_nomal_identify import create_global_reference_normal_samples, clustering_nomal_identify
from Clustering_Method.run_clustering_cni import run_clustering_with_cni
from Dataset_Choose_Rule.Intermediate_reset import reset_intermediate_files
from Tuning_hyperparameter.jaccard_run_single_clustering import run_single_clustering
from Modules.Jaccard_Elbow_Method import test_all_algorithms_with_jaccard_elbow
from utils.generate_data_hash import generate_stable_data_hash
from utils.minmaxscaler import apply_minmax_scaling_and_save_scalers
from utils.max_score_utils import comprehensive_optimization_max_score
from utils.best_clustering_sampling import apply_memory_optimization_sampling, run_chunked_virtual_labeling
from Dataset_Choose_Rule.save_jaccard_elbow import get_grid_search_progress_file_path, save_grid_search_progress
from utils.downcast import maybe_downcast_float32


# Setup logger
logger = logging.getLogger(__name__)

def dynamic_import(module_name, function_name):
    """Dynamically import a function from a module."""
    try:
        module = importlib.import_module(module_name)
        return getattr(module, function_name)
    except ImportError as e:
        logger.error(f"Failed to import {function_name} from {module_name}: {e}")
        return None
    except AttributeError:
        logger.error(f"Function {function_name} not found in module {module_name}.")
        return None

def convert_jaccard_algo_to_original(jaccard_algo_name):
    """Converts Jaccard Elbow method's algorithm name to the original script's name."""
    mapping = {
        'Kmeans': 'kmeans',
        #'Kmedoids': 'kmedoids', # No longer in use due to memory issues -> Clara
        'MShift': 'MShift'
        # Other algorithms often use the same name, so they can be returned as is.
    }
    return mapping.get(jaccard_algo_name, jaccard_algo_name)


def get_save_imports():
    """Lazy import for save functionality"""
    from utils.minmaxscaler import apply_minmax_scaling_and_save_scalers
    from Dataset_Choose_Rule.save_csv import csv_compare_clustering_ex, csv_compare_matrix_clustering_ex
    from Dataset_Choose_Rule.time_save import time_save_csv_VL_ex
    return apply_minmax_scaling_and_save_scalers, csv_compare_clustering_ex, csv_compare_matrix_clustering_ex, time_save_csv_VL_ex

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of supported clustering algorithms (excluding CANNwKNN)
CLUSTERING_ALGORITHMS = [
    'kmeans', 'CLARA', 'GMM', 'SGMM', 'Gmeans', 'Xmeans', 
    'DBSCAN', 'MShift', 'FCM', 'CK', 'NeuralGas'
]

def get_system_resources():
    """Get system resource information."""
    cpu_count = os.cpu_count()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    logger.info(f"System Resources:")
    logger.info(f"  CPU Cores: {cpu_count}")
    logger.info(f"  Total RAM: {memory.total / (1024**3):.2f} GB")
    logger.info(f"  Available RAM: {memory.available / (1024**3):.2f} GB")
    logger.info(f"  Total Disk: {disk.total / (1024**3):.2f} GB")
    logger.info(f"  Available Disk: {disk.free / (1024**3):.2f} GB")
    
    return cpu_count, memory, disk

def persist_array_to_memmap(array, cache_dir, filename, logger):
    """
    Persist a NumPy-compatible array to a memory-mapped file so it can be
    re-used without keeping the entire copy in RAM.
    """
    os.makedirs(cache_dir, exist_ok=True)
    array_np = np.ascontiguousarray(np.asarray(array))
    path = os.path.join(cache_dir, filename)
    memmap = np.memmap(path, dtype=array_np.dtype, mode='w+', shape=array_np.shape)
    memmap[:] = array_np[:]
    memmap.flush()
    logger.info(f"[Memory Cache] Persisted '{filename}' ({array_np.nbytes / (1024**2):.2f} MB) to {cache_dir}")
    return path, memmap


def evaluate_thresholds_for_best_algorithm(data, data_for_clustering, original_labels, best_algorithm,
                                         global_known_normal_samples_pca, autotune_enabled,
                                         num_processes_for_clustering_algo, file_type, file_number):
    """Evaluate multiple thresholds (0.3, 0.4, 0.5, 0.6) for the best algorithm"""
    logger.info(f"Evaluating thresholds for best algorithm: {best_algorithm}")
    
    threshold_candidates = [0.3, 0.4, 0.5, 0.6]
    threshold_results = {}
    
    for threshold in threshold_candidates:
        logger.info(f"Testing threshold: {threshold}")
        try:
            cluster_labels, _, _ = run_clustering_with_cni(
                data, data_for_clustering, original_labels, best_algorithm,
                global_known_normal_samples_pca, autotune_enabled,
                num_processes_for_clustering_algo, [threshold],  # Only test this threshold
                file_type, file_number  # Pass for progress tracking
            )
            
            if cluster_labels is not None and len(cluster_labels) == len(original_labels):
                metrics = evaluate_clustering_wos(original_labels, cluster_labels)
                micro_metrics = metrics.get('average=micro', {})
                jaccard_score = micro_metrics.get('jaccard', -1.0)
                
                threshold_results[threshold] = {
                    'jaccard_score': jaccard_score,
                    'cluster_labels': cluster_labels,
                    'metrics': metrics
                }
                
                logger.info(f"  Threshold {threshold}: Jaccard Score = {jaccard_score:.4f}")
            else:
                logger.warning(f"  Threshold {threshold}: Invalid cluster labels")
                
        except Exception as e:
            logger.error(f"  Threshold {threshold}: Error - {str(e)}")
            continue
    
    return threshold_results

def ensure_dataset_ex_directory():
    """Ensure Dataset_ex directory structure exists (similar to Dataset_Paral)"""
    # Create main Dataset_ex directory
    dataset_ex_dir = "../Dataset_ex"
    if not os.path.exists(dataset_ex_dir):
        os.makedirs(dataset_ex_dir)
        logger.info(f"Created main directory: {dataset_ex_dir}")
    
    # Create subdirectories similar to Dataset_Paral structure
    subdirs = [
        "Data_Label",  # Changed from save_dataset to Data_Label
        "time_log/virtual_labeling",
        "time_log/virtual_labeling_ex",  # For our specific time logs
        "signature",
        "mapping_info",  # For storing mapping information
        "progress_tracking"  # For tracking chunk processing progress
    ]
    
    for subdir in subdirs:
        full_path = os.path.join(dataset_ex_dir, subdir)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            logger.info(f"Created subdirectory: {full_path}")
    
    return dataset_ex_dir

'''
def load_progress(file_type, file_number, algorithm):
    """Load existing progress from CSV file"""
    progress_file = get_progress_file_path(file_type, file_number, algorithm)
    completed_chunks = set()
    
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    if len(row) >= 2:
                        chunk_idx = int(row[0])
                        status = row[1]
                        if status == 'completed':
                            completed_chunks.add(chunk_idx)
        except Exception as e:
            print(f"Warning: Could not load progress file {progress_file}: {e}")
    
    return completed_chunks
'''

'''
def get_next_chunk_to_process(completed_chunks, total_chunks):
    """Get the next chunk index that needs to be processed"""
    for i in range(total_chunks):
        if i not in completed_chunks:
            return i
    return None  # All chunks completed
'''


def preprocess_data(data, file_type):
    """
    Consolidates the initial data preprocessing steps before PCA, including
    time transfer, scaling, and label encoding.
    """
    # Step 1: Time scalar transfer (must be done first)
    logger.info("Applying time scalar transfer...")
    data = time_scalar_transfer(data, file_type)

    # Step 2: Apply scaling and label encoding
    logger.info("Applying scaling and label encoding...")
    try:
        features = data.drop(columns=['label'])
    except KeyError:
        features = data.copy()

    # This call correctly processes the time-converted features
    X_processed, _, _, _ = choose_heterogeneous_method(features, file_type, 'scaling_label_encoding', 'N')
    
    return X_processed

def main():
    parser = argparse.ArgumentParser(description='Best Clustering Algorithm Selector - Parallel Version')
    parser.add_argument('--file_type', type=str, default="MiraiBotnet", help='Data file type')
    parser.add_argument('--file_number', type=int, default=1, help='File number')
    parser.add_argument('--train_test', type=int, default=0, help='Train = 0, test = 1')
    parser.add_argument('--heterogeneous', type=str, default="scaling_label_encoding", help='Heterogeneous method')
    parser.add_argument('--autotune', type=str, default="y", help='Enable autotuning (y/n)')
    parser.add_argument('--max_workers', type=int, default=0, help='Max parallel workers (0 = auto)')
    parser.add_argument('--eval_clustering_silhouette', type=str, default="n", help="Evaluate with silhouette score (y/n)")
    parser.add_argument('--reset', action='store_true', help='Reset all intermediate progress files and start fresh')
    parser.add_argument('--chunk_reset', action='store_true', help='Only delete temporary chunk files and exit.')
    parser.add_argument('--max_score', action='store_true', help='Use max surrogate score instead of elbow method for k selection')
    parser.add_argument('--disable_sampling', action='store_true', help='Disable memory optimization sampling (use full dataset)')
    parser.add_argument('--use_float32', action='store_true', help='Downcast; large floating-point arrays to float32 to save memory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling/consistency fixes (default: 42)')
    parser.add_argument('--known_normal_fraction', type=float, default=0.90, help='Fraction of normal samples to use when building known_normal reference (default: 0.90).')
    parser.add_argument('--cni_normal_batch_size', type=int, default=50000, help='Batch size for deduplicating known-normal samples during virtual labeling (set to 0 to disable)')
    parser.add_argument('--cni_thresholds', type=str, default=None, help='Comma-separated list of CNI thresholds to evaluate (e.g., 0.05,0.1,0.2). Overrides defaults if set.')
    
    args = parser.parse_args()

    cni_normal_batch_size = args.cni_normal_batch_size if args.cni_normal_batch_size and args.cni_normal_batch_size > 0 else None
    
    # --- NEW: Handle chunk_reset argument ---
    if args.chunk_reset:
        from utils.cleanup_temp_files import cleanup_chunk_temp_files
        cleanup_chunk_temp_files()
        sys.exit(0)
    
    if args.reset:
        reset_intermediate_files(args.file_type, args.file_number)
        print("All intermediate progress files have been deleted. Starting fresh...")
    
    total_start_time = time.time()
    timing_info = {}
    
    cpu_count, memory, disk = get_system_resources()
    
    # For small, tricky datasets, align global RNG to the provided seed to influence clustering internals
    if args.file_type in ['MiraiBotnet', 'netML']:
        np.random.seed(args.seed)
        try:
            import random
            random.seed(args.seed)
        except Exception:
            pass
    
    if args.max_workers > 0:
        num_processes = min(args.max_workers, cpu_count)
    else:
        num_processes = cpu_count
    
    logger.info(f"Using {num_processes} processes for parallel tasks.")
    
    algorithm_selection_threshold = [0.4]
    if args.cni_thresholds:
        try:
            final_evaluation_thresholds = [float(x) for x in args.cni_thresholds.split(',') if x.strip() != ""]
            logger.info(f"[CNI] Using custom threshold list: {final_evaluation_thresholds}")
        except ValueError:
            logger.warning(f"[CNI] Failed to parse --cni_thresholds '{args.cni_thresholds}'. Falling back to defaults.")
            final_evaluation_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    else:
        final_evaluation_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    file_type = args.file_type
    file_number = args.file_number
    heterogeneous_method = args.heterogeneous
    eval_clustering_silhouette_flag = args.eval_clustering_silhouette.lower() == 'y'
    autotune_enabled = args.autotune.lower() == 'y'
    
    dataset_ex_dir = ensure_dataset_ex_directory()
    memory_cache_dir = os.path.join(dataset_ex_dir, "memory_cache")
    temp_memmap_files = []
    
    logger.info("Step 1: Loading data...")
    start_time = time.time()
    file_path, file_number = file_path_line_nonnumber(file_type, file_number)
    data = file_cut(file_type, file_path, 'all')
    data.columns = data.columns.str.strip()
    timing_info['1_load_data'] = time.time() - start_time
    logger.info(f"Step 1 completed in {timing_info['1_load_data']:.2f}s")
    
    logger.info("Step 2: Applying labeling logic...")
    start_time = time.time()
    data = apply_labeling_logic(data, file_type)
    timing_info['2_labeling_time'] = time.time() - start_time
    logger.info(f"Step 2 completed in {timing_info['2_labeling_time']:.2f}s")
    
    # --- SELECTIVE PREPROCESSING PIPELINE ---
    if heterogeneous_method == 'scaling_label_encoding':
        # --- PIPELINE A: Scaling and Label Encoding ---
        logger.info("Using 'scaling_label_encoding' pipeline.")
        
        # Step 3: Time Scalar Transfer (MUST BE DONE FIRST)
        logger.info("Step 3: Applying time scalar transfer...")
        start_time = time.time()
        # Apply to the whole `data` dataframe before feature selection
        data_for_processing = time_scalar_transfer(data.copy(), file_type)
        timing_info['3_time_transfer_time'] = time.time() - start_time
        logger.info(f"Step 3 completed in {timing_info['3_time_transfer_time']:.2f}s")
        
        # Step 4: Explicitly select features for clustering
        logger.info("Step 4: Selecting features for clustering...")
        start_time = time.time()
        from Heterogeneous_Method.Feature_Encoding import Heterogeneous_Feature_named_featrues
        
        feature_dict = Heterogeneous_Feature_named_featrues(file_type)
        categorical_features = feature_dict.get('categorical_features', [])
        time_features = feature_dict.get('time_features', [])
        packet_length_features = feature_dict.get('packet_length_features', [])
        count_features = feature_dict.get('count_features', [])
        binary_features = feature_dict.get('binary_features', [])

        features_for_clustering = categorical_features + time_features + packet_length_features + count_features + binary_features
        
        # Now, select from the time-transferred dataframe
        existing_features = [f for f in features_for_clustering if f in data_for_processing.columns]
        missing_features = set(features_for_clustering) - set(existing_features)
        if missing_features:
            logger.warning(f"The following features defined in Feature_Encoding.py were not found: {missing_features}")

        try:
            # Re-assign data_for_processing to be the selected subset
            data_for_processing = data_for_processing[existing_features + ['label']].copy()
        except KeyError as e:
            logger.critical(f"CRITICAL ERROR: A required column is missing after feature selection: {e}. Cannot proceed.")
            sys.exit(1)
            
        timing_info['4_feature_selection_time'] = time.time() - start_time
        logger.info(f"Step 4 completed in {timing_info['4_feature_selection_time']:.2f}s. Selected {len(existing_features)} features.")

        # Step 5: Data Cleaning
        logger.info("Step 5: Cleaning numerical features...")
        start_time = time.time()
        numerical_features = time_features + packet_length_features + count_features
        columns_to_clean = [col for col in numerical_features if col in data_for_processing.columns]
        
        if columns_to_clean:
            initial_rows = len(data_for_processing)
            # 1. Coerce to numeric, turning invalid parsing into NaN
            for col in columns_to_clean:
                if not pd.api.types.is_numeric_dtype(data_for_processing[col]):
                    data_for_processing[col] = pd.to_numeric(data_for_processing[col], errors='coerce')

            # 2. Handle infinity values
            for col in columns_to_clean:
                if not np.isfinite(data_for_processing[col].values).all():
                    if np.isinf(data_for_processing[col].values).any():
                        finite_max = data_for_processing.loc[np.isfinite(data_for_processing[col]), col].max()
                        replacement_val = 0 if pd.isna(finite_max) else finite_max
                        data_for_processing[col].replace([np.inf, -np.inf], replacement_val, inplace=True)
                        logger.warning(f"Replaced infinite values in column '{col}' with {replacement_val}.")
            
            # 3. Drop rows with any remaining NaN values
            logger.info(f"[DEBUG Cleaning] Label distribution before NaN drop: {data_for_processing['label'].value_counts().to_dict()}")
            rows_with_nan = data_for_processing[columns_to_clean].isnull().any(axis=1)
            indices_to_drop = data_for_processing[rows_with_nan].index
            
            if not indices_to_drop.empty:
                num_rows_to_drop = len(indices_to_drop)
                labels_of_dropped_rows = data_for_processing.loc[indices_to_drop, 'label'].value_counts().to_dict()
                logger.warning(f"[DEBUG Cleaning] Label distribution of rows TO BE DROPPED: {labels_of_dropped_rows}")
                
                drop_percentage = (num_rows_to_drop / initial_rows) * 100
                logger.warning(f"Found {num_rows_to_drop} rows ({drop_percentage:.2f}%) with NaN values in numerical columns to remove.")
                data_for_processing.drop(index=indices_to_drop, inplace=True)
                
                if data_for_processing.empty:
                    logger.critical("CRITICAL ERROR: All data rows removed during cleaning. Cannot proceed.")
                    sys.exit(1)
            
            logger.info(f"[DEBUG Cleaning] Label distribution after NaN drop: {data_for_processing['label'].value_counts().to_dict()}")
        
        timing_info['5_data_cleaning_time'] = time.time() - start_time
        logger.info(f"Step 5 completed in {timing_info['5_data_cleaning_time']:.2f}s")
        
        # Step 6: Separate labels and apply final Scaling/Encoding
        logger.info("Step 6: Separating labels and applying final scaling/encoding...")
        start_time = time.time()
        original_labels = data_for_processing['label'].to_numpy()
        X = data_for_processing.drop(columns=['label'])
        X_processed, _, _, _ = choose_heterogeneous_method(X, file_type, 'scaling_label_encoding', 'N')
        timing_info['6_scaling_encoding_time'] = time.time() - start_time
        logger.info(f"Step 6 completed in {timing_info['6_scaling_encoding_time']:.2f}s.")

        # Step 7: Applying PCA
        logger.info("Step 7: Applying PCA...")
        start_time = time.time()
        pca_want = 'N' if file_type in ['DARPA98', 'DARPA', 'CICModbus23', 'CICModbus'] else 'Y'
        
        if pca_want == 'Y':
            data_for_clustering = pca_func(X_processed)
        else:
            data_for_clustering = X_processed.to_numpy() if hasattr(X_processed, 'to_numpy') else X_processed
        
        timing_info['7_pca_time'] = time.time() - start_time
        logger.info(f"Step 7 completed in {timing_info['7_pca_time']:.2f}s")
        data_for_clustering = maybe_downcast_float32(data_for_clustering, args.use_float32, "data_for_clustering")

    else:
        # --- PIPELINE B: Interval Inverse (and other embedding methods) ---
        logger.info(f"Using default embedding pipeline for '{heterogeneous_method}'.")

        # Step 3: Time scalar transfer
        logger.info("Step 3: Applying time scalar transfer...")
        start_time = time.time()
        data_processed = time_scalar_transfer(data, file_type) # Use a different variable name
        timing_info['3_time_transfer_time'] = time.time() - start_time
        logger.info(f"Step 3 completed in {timing_info['3_time_transfer_time']:.2f}s")

        # Step 4: Embedding and Group Mapping
        logger.info("Step 4: Applying embedding and group mapping...")
        start_time = time.time()
        regul = 'N'
        embedded_dataframe, _, category_mapping, data_list = choose_heterogeneous_method(data_processed, file_type, heterogeneous_method, regul)
        group_mapped_df, _ = map_intervals_to_groups(embedded_dataframe, category_mapping, data_list, regul)
        timing_info['4_embedding_time'] = time.time() - start_time
        logger.info(f"Step 4 completed in {timing_info['4_embedding_time']:.2f}s")
        
        # Step 5: MinMax Scaling
        logger.info("Step 5: Applying MinMaxScaler...")
        start_time = time.time()
        # No need for another import, it's at the top of the file
        X_scaled_for_pca = pd.DataFrame(MinMaxScaler().fit_transform(group_mapped_df), columns=group_mapped_df.columns, index=group_mapped_df.index)
        timing_info['5_scaling_time'] = time.time() - start_time
        logger.info(f"Step 5 completed in {timing_info['5_scaling_time']:.2f}s")
        
        # Step 6: PCA
        logger.info("Step 6: Applying PCA...")
        start_time = time.time()
        pca_want = 'N' if file_type in ['DARPA98', 'DARPA', 'CICModbus23', 'CICModbus'] else 'Y'
        
        if pca_want == 'Y':
            data_for_clustering = pca_func(X_scaled_for_pca)
        else:
            data_for_clustering = X_scaled_for_pca.to_numpy() if hasattr(X_scaled_for_pca, 'to_numpy') else X_scaled_for_pca

        timing_info['6_pca_time'] = time.time() - start_time
        logger.info(f"Step 6 completed in {timing_info['6_pca_time']:.2f}s")
        
        # Align original_labels for consistency
        original_labels = data['label'].to_numpy()
        data_for_clustering = maybe_downcast_float32(data_for_clustering, args.use_float32, "data_for_clustering")

    logger.info("Step 8: Creating Global Reference Normal Samples for CNI...")
    start_time = time.time()
    data_hash = generate_stable_data_hash(file_type, file_number, X_shape=data_for_clustering.shape)
    global_known_normal_samples_pca_for_cni, known_normal_indices = create_global_reference_normal_samples(file_type, file_number, heterogeneous_method)
    timing_info['8_global_ref_creation_time'] = time.time() - start_time
    logger.info(f"Step 8 completed in {timing_info['8_global_ref_creation_time']:.2f}s")

    # Step 8.5: Apply memory optimization sampling for algorithm selection
    logger.info("Step 8.5: Applying memory optimization sampling...")
    start_time = time.time()
    
    if args.disable_sampling:
        logger.info("[Memory Optimization] Sampling disabled by user - Using full dataset")
        algorithm_selection_data = data_for_clustering
        algorithm_selection_labels = original_labels
    else:
        algorithm_selection_data, algorithm_selection_labels = apply_memory_optimization_sampling(
            data_for_clustering, original_labels, file_type, seed=args.seed    # Add seed information
        )
    
    timing_info['8.5_memory_optimization_time'] = time.time() - start_time
    logger.info(f"Step 8.5 completed in {timing_info['8.5_memory_optimization_time']:.2f}s")
    algorithm_selection_data = maybe_downcast_float32(algorithm_selection_data, args.use_float32, "algorithm_selection_data")
    gc.collect()

    logger.info("\n" + "="*60)
    logger.info("Step 9: Starting algorithm selection...")
    if args.max_score:
        logger.info("Mode: Max Score optimization")
    else:
        logger.info("Mode: Elbow Method optimization")
    start_time = time.time()
    
    # Convert algorithm names to match the Jaccard Elbow method format
    jaccard_algorithms_map = {
        'kmeans': 'Kmeans', 'CLARA': 'CLARA', 'GMM': 'GMM', 'SGMM': 'SGMM',
        'Gmeans': 'Gmeans', 'Xmeans': 'Xmeans', 'DBSCAN': 'DBSCAN', 'MShift': 'MShift',
        'FCM': 'FCM', 'CK': 'CK', 'NeuralGas': 'NeuralGas'
    }
    jaccard_algorithms = [jaccard_algorithms_map[algo] for algo in CLUSTERING_ALGORITHMS]
    
    try:
        # === DATA CONSISTENCY FIX ===
        logger.info("[FIX] Ensuring data consistency between source data and known normal samples...")
        try:
            all_normal_samples_main = data_for_clustering[original_labels == 0]
            num_all_normal = all_normal_samples_main.shape[0]
            if num_all_normal > 1:
                frac = max(0.0, min(1.0, args.known_normal_fraction))
                sample_size = int(num_all_normal * frac)
                if sample_size == 0 and num_all_normal > 0: sample_size = 1
                
                np.random.seed(args.seed)
                random_indices = np.random.choice(num_all_normal, size=sample_size, replace=False)
                consistent_known_normal_samples_pca = all_normal_samples_main[random_indices]
                # Also get the corresponding indices from the main data for the new surrogate score
                main_all_normal_indices = np.where(original_labels == 0)[0]
                consistent_known_normal_indices = main_all_normal_indices[random_indices]
                logger.info(f"[FIX] Re-created a consistent known_normal_samples set. Shape: {consistent_known_normal_samples_pca.shape}")
                logger.info(f"[FIX] Re-created consistent known_normal_indices. Shape: {consistent_known_normal_indices.shape}")
            else:
                logger.warning("[FIX] Not enough normal samples in main data. Using global reference as fallback.")
                consistent_known_normal_samples_pca = global_known_normal_samples_pca_for_cni
                consistent_known_normal_indices = known_normal_indices # Fallback to indices from global ref
        except Exception as e:
            logger.error(f"[FIX] Error during data consistency fix: {e}. Using original global reference.")
            consistent_known_normal_samples_pca = global_known_normal_samples_pca_for_cni
            consistent_known_normal_indices = known_normal_indices # Fallback to indices from global ref
        
        # === SAMPLING CONSISTENCY FIX ===
        # If sampling was applied, we need to recalculate known_normal_indices for the sampled data
        
        # List of datasets that should be sampled (from sampling_configs in utils/best_clustering_sampling.py)
        datasets_with_sampling = ['CICIDS2017', 'CICIDS', 'CICIoT2023', 'CICIoT', 'IoTID20', 'IoTID', 'Kitsune', 'NSL-KDD', 'NSL_KDD']
        
        # Check if sampling was actually applied for this dataset type
        should_sample = not args.disable_sampling and file_type in datasets_with_sampling and algorithm_selection_data.shape[0] < data_for_clustering.shape[0]
        
        if should_sample:
            # Store the original full dataset indices before potentially overwriting (only when sampling is applied)
            consistent_known_normal_indices_full_dataset = consistent_known_normal_indices.copy()
            logger.info("[SAMPLING FIX] Sampling was applied - Recalculating known_normal_indices for sampled data...")
            logger.info(f"[SAMPLING FIX] File type: {file_type}, Sampled: {algorithm_selection_data.shape[0]}, Original: {data_for_clustering.shape[0]}")
            logger.info(f"[SAMPLING FIX] Preserving full dataset indices. Shape: {consistent_known_normal_indices_full_dataset.shape}")
            try:
                # Find normal samples in the sampled data
                sampled_normal_indices = np.where(algorithm_selection_labels == 0)[0]
                if len(sampled_normal_indices) > 0:
                    frac = max(0.0, min(1.0, args.known_normal_fraction))
                    sample_size_sampled = int(len(sampled_normal_indices) * frac)
                    if sample_size_sampled == 0 and len(sampled_normal_indices) > 0: 
                        sample_size_sampled = 1
                    
                    np.random.seed(args.seed)  # Use same seed for consistency
                    selected_sampled_indices = np.random.choice(len(sampled_normal_indices), 
                                                              size=min(sample_size_sampled, len(sampled_normal_indices)), 
                                                              replace=False)
                    consistent_known_normal_indices = sampled_normal_indices[selected_sampled_indices]
                    
                    logger.info(f"[SAMPLING FIX] Recalculated known_normal_indices for sampled data. Shape: {consistent_known_normal_indices.shape}")
                    logger.info(f"[SAMPLING FIX] Sampled data shape: {algorithm_selection_data.shape}, Original data shape: {data_for_clustering.shape}")
                    logger.info(f"[SAMPLING FIX] Full dataset indices preserved. Shape: {consistent_known_normal_indices_full_dataset.shape}")
                else:
                    logger.warning("[SAMPLING FIX] No normal samples found in sampled data.")
                    consistent_known_normal_indices = np.array([])
            except Exception as e:
                logger.error(f"[SAMPLING FIX] Error recalculating indices for sampled data: {e}")
                consistent_known_normal_indices = np.array([])
        else:
            # No sampling applied case: both indices are the same
            consistent_known_normal_indices_full_dataset = consistent_known_normal_indices
            logger.info(f"[SAMPLING FIX] No sampling applied for {file_type} - using same indices for both full and sampled data")
        # === END SAMPLING FIX ===

        if args.max_score:
            logger.info("Invoking Max Score optimization path...")
            jaccard_results = {}
            for algorithm in jaccard_algorithms:
                logger.info(f"[{algorithm}] Starting Max Score optimization with grid search progress tracking...")
                
                # Note: DBSCAN and MShift now use internal chunking for large datasets
                # No need for pre-sampling here - they will handle memory management internally
                
                # Check for existing grid search progress files
                data_hash = generate_stable_data_hash(file_type, file_number, algorithm_selection_data.shape)
                grid_progress_file = get_grid_search_progress_file_path(algorithm, data_hash)
                if os.path.exists(grid_progress_file):
                    logger.info(f"[{algorithm}] Found existing grid search progress file: {grid_progress_file}")
                else:
                    logger.info(f"[{algorithm}] No existing grid search progress file found. Will create: {grid_progress_file}")
                
                jaccard_results[algorithm] = comprehensive_optimization_max_score(
                    algorithm, algorithm_selection_data, algorithm_selection_labels, file_type, file_number,    # algorithm, data_for_clustering, original_labels, file_type, file_number,
                    consistent_known_normal_samples_pca,
                    num_processes_for_algo=num_processes,
                    known_normal_idx=consistent_known_normal_indices
                )
        else:
            logger.info("Invoking Elbow Method optimization path...")
            
            # Check for existing grid search progress files for each algorithm
            data_hash = generate_stable_data_hash(file_type, file_number, algorithm_selection_data.shape)
            for algorithm in jaccard_algorithms:
                grid_progress_file = get_grid_search_progress_file_path(algorithm, data_hash)
                if os.path.exists(grid_progress_file):
                    logger.info(f"[{algorithm}] Found existing grid search progress file: {grid_progress_file}")
                else:
                    logger.info(f"[{algorithm}] No existing grid search progress file found. Will create: {grid_progress_file}")
            
            jaccard_results = test_all_algorithms_with_jaccard_elbow(
                algorithm_selection_data, algorithm_selection_labels, jaccard_algorithms, file_type, file_number,   # data_for_clustering, original_labels, jaccard_algorithms, file_type, file_number,
                consistent_known_normal_samples_pca,
                num_processes_for_algo=num_processes,
                known_normal_idx=consistent_known_normal_indices
            )

        best_jaccard_algorithm = None
        best_jaccard_score = -1.0
        for algorithm, result in jaccard_results.items():
            current_jaccard_score = result.get('best_jaccard', 0.0)
            if current_jaccard_score > best_jaccard_score:
                best_jaccard_score = current_jaccard_score
                best_jaccard_algorithm = algorithm


        if not best_jaccard_algorithm:
            raise ValueError("Jaccard Elbow method failed to select any algorithm.")

        best_algorithm_name = convert_jaccard_algo_to_original(best_jaccard_algorithm)
        best_params = jaccard_results[best_jaccard_algorithm]['best_params']
        
        logger.info(f"Jaccard Elbow method selected: '{best_jaccard_algorithm}' (as {best_algorithm_name}) with Jaccard score: {best_jaccard_score:.4f}")
        logger.info(f"Best parameters found: {best_params}")

        # === NEW: Determine best_threshold (use sampled data by default; use full data if small dataset) ===
        logger.info(f"\nStep 9.5: Determining best CNI threshold for '{best_jaccard_algorithm}'...")
        use_full_for_threshold = (not args.disable_sampling) and data_for_clustering.shape[0] <= 10_000
        if use_full_for_threshold:
            logger.info(f"[Threshold Eval] Using FULL data for threshold search (small dataset: {data_for_clustering.shape[0]} rows).")
            threshold_eval_data = data_for_clustering
            threshold_eval_labels = original_labels
            kn_idx_for_threshold = consistent_known_normal_indices_full_dataset
        else:
            threshold_eval_data = algorithm_selection_data
            threshold_eval_labels = algorithm_selection_labels
            kn_idx_for_threshold = consistent_known_normal_indices

        # Run clustering on the chosen dataset to get raw labels for threshold evaluation
        sample_raw_labels = run_single_clustering(
            best_jaccard_algorithm,
            threshold_eval_data,
            best_params,
            aligned_original_labels=threshold_eval_labels,
            global_known_normal_samples_pca=consistent_known_normal_samples_pca
        )

        best_threshold_on_sample = None
        best_jaccard_on_sample = -1.0

        if sample_raw_labels is not None:
            logger.info("  Evaluating thresholds on sampled data result...")
            for threshold in final_evaluation_thresholds:
                # Note: consistent_known_normal_indices is already adapted for sampling if it happened
                _, jaccard_score_from_cni, _ = clustering_nomal_identify(
                    data_features_for_clustering=threshold_eval_data,
                    clusters_assigned=sample_raw_labels,
                    original_labels_aligned=threshold_eval_labels,
                    global_known_normal_samples_pca=consistent_known_normal_samples_pca,
                    threshold_value=threshold,
                    num_processes_for_algo=num_processes,
                    data_for_clustering=threshold_eval_data,
                    known_normal_idx=kn_idx_for_threshold,
                    normal_data_batch_size=cni_normal_batch_size
                )
                if jaccard_score_from_cni > best_jaccard_on_sample:
                    best_jaccard_on_sample = jaccard_score_from_cni
                    best_threshold_on_sample = threshold
            logger.info(f"  Best threshold on sampled data is {best_threshold_on_sample} with Jaccard score {best_jaccard_on_sample:.4f}")
        else:
            logger.warning("  Could not generate raw labels for sampled data. Falling back to default threshold 0.4.")
            best_threshold_on_sample = 0.4
        # === END NEW SECTION ===

        # Release sampled data arrays before moving to full-dataset processing
        try:
            del sample_raw_labels
        except NameError:
            pass
        del algorithm_selection_data
        del algorithm_selection_labels
        gc.collect()

        # Persist large full-dataset arrays to memory-mapped files to reduce RAM pressure
        if not isinstance(data_for_clustering, np.memmap):
            data_cache_path, data_for_clustering = persist_array_to_memmap(
                data_for_clustering, memory_cache_dir,
                f"{file_type}_{file_number}_data_for_clustering.mmap", logger
            )
            temp_memmap_files.append(data_cache_path)
        if not isinstance(original_labels, np.memmap):
            labels_cache_path, original_labels = persist_array_to_memmap(
                original_labels, memory_cache_dir,
                f"{file_type}_{file_number}_original_labels.mmap", logger
            )
            temp_memmap_files.append(labels_cache_path)

        logger.info(f"\nStep 10: Applying chunked virtual labeling to the FULL dataset...")
        cni_start_time = time.time()
        
        best_cluster_labels = run_chunked_virtual_labeling(
            data_for_clustering, # FULL data
            original_labels, # FULL labels
            best_jaccard_algorithm,
            best_threshold_on_sample, # The globally fixed threshold
            file_type, file_number,
            consistent_known_normal_samples_pca,
            consistent_known_normal_indices_full_dataset, # Indices for the full dataset
            num_processes,
            normal_data_batch_size=cni_normal_batch_size
        )
        
        # Recalculate the final Jaccard score on the full dataset labels
        if best_cluster_labels is not None and len(best_cluster_labels) == len(original_labels) and np.any(best_cluster_labels != -1):
            best_jaccard_score = jaccard_score(original_labels, best_cluster_labels, average='weighted')
        else:
            best_jaccard_score = -1.0 # Or 0.0, depending on desired behavior for failure
        best_threshold = best_threshold_on_sample # For logging and saving

        timing_info['10_cni_evaluation_time'] = time.time() - cni_start_time
        logger.info(f"Step 10 completed in {timing_info['10_cni_evaluation_time']:.2f}s. Final Jaccard on full data: {best_jaccard_score:.4f}")
        
    except Exception as e:
        logger.error(f"An error occurred during the Jaccard Elbow method or subsequent CNI evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.info("Falling back to original algorithm selection method...")
        
        best_algorithm_name = None
        best_jaccard_score = -1.0
        best_cluster_labels = None
        best_threshold = 0.4
        
        total_algorithms = len(CLUSTERING_ALGORITHMS)
        
        for idx, algorithm in enumerate(CLUSTERING_ALGORITHMS, 1):
            logger.info(f"Testing algorithm {idx}/{total_algorithms}: {algorithm}")
            temp_labels, temp_thresh, _ = run_clustering_with_cni(
                data, data_for_clustering, original_labels, algorithm,
                global_known_normal_samples_pca_for_cni, autotune_enabled,
                num_processes, algorithm_selection_threshold,
                file_type, file_number
            )
            if temp_labels is not None and len(temp_labels) == len(original_labels):
                metrics = evaluate_clustering_wos(original_labels, temp_labels)
                current_jaccard_score = metrics.get('average=micro', {}).get('jaccard', -1.0)
                if current_jaccard_score > best_jaccard_score:
                    best_jaccard_score = current_jaccard_score
                    best_algorithm_name = algorithm
                    best_cluster_labels = temp_labels
                    best_threshold = temp_thresh
    
    timing_info['9_clustering_and_tuning_time'] = time.time() - start_time
    logger.info(f"Step 9 completed in {timing_info['9_clustering_and_tuning_time']:.2f}s")
    
    if best_algorithm_name and best_cluster_labels is not None:
        logger.info(f"Best algorithm: {best_algorithm_name} with Jaccard score: {best_jaccard_score:.4f}")
        
        data.reset_index(drop=True, inplace=True)
        if len(best_cluster_labels) == len(data):
            data['cluster'] = best_cluster_labels
        else:
            logger.warning(f"Length mismatch between final labels ({len(best_cluster_labels)}) and data ({len(data)}). Padding with -1.")
            padded_labels = np.full(len(data), -1, dtype=int)
            padded_labels[:len(best_cluster_labels)] = best_cluster_labels
            data['cluster'] = padded_labels

        data['adjusted_cluster'] = 1 - data['cluster']
        
        logger.info("Algorithm Performance Summary:")
        if 'jaccard_results' in locals() and jaccard_results:
            for alg, result in sorted(jaccard_results.items(), key=lambda x: x[1].get('best_jaccard', 0.0), reverse=True):
                logger.info(f"  {alg}: {result.get('best_jaccard', 0.0):.4f}")
        else:
            logger.info(f"  {best_algorithm_name}: {best_jaccard_score:.4f}")
        
        if file_type in ['CICIDS2017', 'CICIDS']:
            output_file_type = "CICIDS2017"
        elif file_type in ['DARPA', 'DARPA98']:
            output_file_type = "DARPA98"
        elif file_type in ['CICModbus23', 'CICModbus']:
            output_file_type = "CICModbus23"
        elif file_type in ['IoTID20', 'IoTID']:
            output_file_type = "IoTID20"
        elif file_type == 'Kitsune':
            output_file_type = "Kitsune"
        elif file_type in ['CICIoT', 'CICIoT2023']:
            output_file_type = "CICIoT2023"
        else:
            output_file_type = file_type

        output_dir = os.path.join(dataset_ex_dir, "load_dataset", output_file_type)
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, f"best_clustering_{output_file_type}_{file_number}.csv")
        data.to_csv(output_filename, index=False)
        logger.info(f"Results saved to: {output_filename}")
        
        start = time.time()
        
        determined_gmm_type = None 
        if best_algorithm_name.upper().startswith("GMM"):
            parts = best_algorithm_name.split('_')
            if len(parts) == 1 and parts[0].upper() == 'GMM': 
                determined_gmm_type = "normal"
            elif len(parts) == 2 and parts[0].upper() == 'GMM' and parts[1].lower() in ['normal', 'full', 'tied', 'diag']:
                determined_gmm_type = parts[1].lower()
        
        timing_info['11_save_time_start_hook'] = time.time() - start
        
        apply_minmax_scaling_and_save_scalers, csv_compare_clustering_ex, csv_compare_matrix_clustering_ex, time_save_csv_VL_ex = get_save_imports()
        time_save_csv_VL_ex(file_type, file_number, best_algorithm_name, timing_info)
        
        if 'cluster' in data.columns and len(data['cluster']) == len(original_labels):
            _, csv_compare_clustering_ex, _, _ = get_save_imports()
            csv_compare_clustering_ex(
                file_type, best_algorithm_name, file_number, data,
                GMM_type=determined_gmm_type,
                optimal_cni_threshold=best_threshold
            )
            
            y_true = data['label'].to_numpy()
            y_pred_original = data['cluster'].to_numpy()
            y_pred_adjusted = data['adjusted_cluster'].to_numpy()
            
            if eval_clustering_silhouette_flag:
                metrics_original = evaluate_clustering(y_true, y_pred_original, data_for_clustering)
                metrics_adjusted = evaluate_clustering(y_true, y_pred_adjusted, data_for_clustering)
            else:
                metrics_original = evaluate_clustering_wos(y_true, y_pred_original)
                metrics_adjusted = evaluate_clustering_wos(y_true, y_pred_adjusted)
            
            _, _, csv_compare_matrix_clustering_ex, _ = get_save_imports()
            csv_compare_matrix_clustering_ex(
                file_type, file_number, best_algorithm_name,
                metrics_original, metrics_adjusted,
                GMM_type=determined_gmm_type,
                optimal_cni_threshold=best_threshold
            )
        else:
            logger.warning("[WARN Save] 'cluster' column not available or length mismatch. Skipping detailed CSV saving.")
        
        logger.info("Skipping saving performance results for all other non-best algorithms as new CNI logic makes it redundant.")

        timing_info['11_save_time'] = time.time() - start
        logger.info(f"Step 11 finished. Save Time: {timing_info['11_save_time']:.2f}s")
        
        summary_filename = os.path.join(dataset_ex_dir, f"clustering_summary_{file_type}_{file_number}.txt")
        with open(summary_filename, 'w', encoding='utf-8') as f:
            f.write(f"Best Clustering Algorithm Selection Results\n")
            f.write(f"==========================================\n")
            f.write(f"File Type: {file_type}\n")
            f.write(f"File Number: {file_number}\n")
            f.write(f"Best Algorithm: {best_algorithm_name}\n")
            f.write(f"Best Jaccard Score: {best_jaccard_score:.4f}\n")
            f.write(f"Best Threshold: {best_threshold}\n")
            f.write(f"Total Time: {time.time() - total_start_time:.2f}s\n")
            f.write(f"Internal Processes: {num_processes}\n")
            f.write(f"System CPU Cores: {cpu_count}\n")
            f.write(f"Available RAM: {memory.available / (1024**3):.2f} GB\n\n")
            
            f.write("All Algorithm Results (Jaccard Elbow Method):\n")
            f.write("--------------------------------------------\n")
            if 'jaccard_results' in locals() and jaccard_results:
                for alg, result in sorted(jaccard_results.items(), key=lambda x: x[1].get('best_jaccard', 0.0), reverse=True):
                    f.write(f"{alg}: {result.get('best_jaccard', 0.0):.4f}\n")
            else:
                f.write(f"{best_algorithm_name}: {best_jaccard_score:.4f}\n")
        
        logger.info(f"Summary saved to: {summary_filename}")
        
    else:
        logger.error("No valid clustering results found")
    
    try:
        total_end_time = time.time()
        timing_info['0_total_time'] = total_end_time - total_start_time
        
        apply_minmax_scaling_and_save_scalers, csv_compare_clustering_ex, csv_compare_matrix_clustering_ex, time_save_csv_VL_ex = get_save_imports()
        time_save_csv_VL_ex(file_type, file_number, best_algorithm_name if best_algorithm_name else "none", timing_info)
        
        total_time = time.time() - total_start_time
        logger.info(f"Total execution time: {total_time:.2f}s")
        
        final_memory = psutil.virtual_memory()
        logger.info(f"Final memory usage: {final_memory.percent:.1f}%")
    finally:
        try:
            if isinstance(data_for_clustering, np.memmap):
                data_for_clustering._mmap.close()
        except Exception:
            pass
        try:
            if isinstance(original_labels, np.memmap):
                original_labels._mmap.close()
        except Exception:
            pass
        for temp_file in temp_memmap_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.info(f"[Memory Cache] Removed temporary file: {temp_file}")
            except OSError as cleanup_error:
                logger.warning(f"[Memory Cache] Could not remove {temp_file}: {cleanup_error}")
        gc.collect()
    '''
    except Exception as cleanup_error:
        logger.error(f"[Memory Cache] Cleanup failed: {cleanup_error}")
    '''


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nThe job was stopped by the user.")
        print("Check the interim saved files:")
        print("- .csv files in ../Dataset_ex/progress_tracking/ folder")
        print("  - jaccard_elbow_*_progress.csv (Elbow method progress)")
        print("  - jaccard_elbow_*_Grid_progress.csv (Grid search progress)")
        print("- Time log files in ../Dataset_ex/time_log/ folder")
        print("\nWhen restarting, it will resume from the point where it was interrupted.")
        sys.exit(0)
