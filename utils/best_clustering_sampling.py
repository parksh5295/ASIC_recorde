import logging
import gc
import numpy as np
import math
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import glob
import os

from utils.max_score_utils import comprehensive_optimization_max_score
from Tuning_hyperparameter.jaccard_run_single_clustering import run_single_clustering
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify
from utils.generate_data_hash import generate_temp_chunk_hash
from Dataset_Choose_Rule.save_jaccard_elbow import get_grid_search_progress_file_path


# Setup logger
logger = logging.getLogger(__name__)

def apply_memory_optimization_sampling(data_for_clustering, original_labels, file_type, seed=42):    # Add seed information
    """
    Apply dataset-specific sampling for memory optimization.
    Large datasets are sampled to prevent memory issues during algorithm selection.
    """
    # Dataset-specific sampling configurations
    sampling_configs = {
        'CICIDS2017': {'sample_size': 30000, 'description': 'Large network dataset'},
        'CICIDS': {'sample_size': 30000, 'description': 'Large network dataset'},
        'CICIoT2023': {'sample_size': 300000, 'description': 'IoT dataset'},
        'CICIoT': {'sample_size': 300000, 'description': 'IoT dataset'},
        'IoTID20': {'sample_size': 30000, 'description': 'IoT dataset'},
        'IoTID': {'sample_size': 30000, 'description': 'IoT dataset'},
        'Kitsune': {'sample_size': 30000, 'description': 'IoT dataset - no sampling needed'},
        'NSL-KDD': {'sample_size': 30000, 'description': 'Network dataset'},
        'NSL_KDD': {'sample_size': 30000, 'description': 'Network dataset'},
        'netML': {'sample_size': None, 'description': 'Network dataset'},
        # Smaller datasets don't need sampling
        'MiraiBotnet': {'sample_size': None, 'description': 'Small dataset - no sampling needed'},
        'DARPA98': {'sample_size': None, 'description': 'Small dataset - no sampling needed'},
        'DARPA': {'sample_size': None, 'description': 'Small dataset - no sampling needed'},
        'CICModbus23': {'sample_size': None, 'description': 'Small dataset - no sampling needed'},
        'CICModbus': {'sample_size': None, 'description': 'Small dataset - no sampling needed'},
    }
    
    config = sampling_configs.get(file_type, {'sample_size': None, 'description': 'Unknown dataset'})
    sample_size = config['sample_size']
    
    if sample_size is None:
        logger.info(f"[Memory Optimization] {file_type}: {config['description']} - Using full dataset")
        return data_for_clustering, original_labels
    
    total_samples = len(data_for_clustering)
    if total_samples <= sample_size:
        logger.info(f"[Memory Optimization] {file_type}: Only {total_samples} samples - Using full dataset")
        return data_for_clustering, original_labels
    
    logger.info(f"[Memory Optimization] {file_type}: {config['description']}")
    logger.info(f"[Memory Optimization] Sampling {sample_size} from {total_samples} samples ({sample_size/total_samples*100:.1f}%)")
    
    # Stratified sampling to maintain label distribution with explicit seed
    rng = np.random.default_rng(seed)    # Add seed information
    unique_labels, label_counts = np.unique(original_labels, return_counts=True)
    
    sampled_indices = []
    for label in unique_labels:
        label_indices = np.where(original_labels == label)[0]
        label_sample_size = max(1, int(sample_size * (len(label_indices) / total_samples)))
        if len(label_indices) <= label_sample_size:
            sampled_indices.extend(label_indices)
        else:
            selected_indices = rng.choice(label_indices, size=label_sample_size, replace=False)    # np.random.choice -> rng.choice
            sampled_indices.extend(selected_indices)
    
    sampled_indices = np.array(sampled_indices)
    
    sampled_data = data_for_clustering[sampled_indices]
    sampled_labels = original_labels[sampled_indices]
    
    logger.info(f"[Memory Optimization] Final sample: {len(sampled_data)} samples")
    logger.info(f"[Memory Optimization] Label distribution: {dict(zip(*np.unique(sampled_labels, return_counts=True)))}")
    
    return sampled_data, sampled_labels


def run_chunked_virtual_labeling(
    data_for_clustering, original_labels,
    best_jaccard_algorithm, best_global_threshold,
    file_type, file_number,
    global_known_normal_samples_pca,
    consistent_known_normal_indices_full_dataset, # The indices for the full dataset
    num_processes,
    chunk_size=30000,
    normal_data_batch_size=None,
    raw_data_for_save=None,
    chunk_save_path=None
):
    """
    Run the final virtual labeling process in chunks, re-optimizing hyperparameters for each chunk.
    """
    logger.info(f"Starting chunked virtual labeling for {best_jaccard_algorithm} with fixed threshold={best_global_threshold}...")
    n_samples = data_for_clustering.shape[0]
    final_labels_full = np.full(n_samples, -1, dtype=int)

    # If dataset is small, no need to chunk
    if n_samples <= chunk_size:
        logger.info("Dataset is smaller than chunk size, processing as a single chunk.")
        num_chunks = 1
    else:
        num_chunks = math.ceil(n_samples / chunk_size)

    for i in tqdm(range(num_chunks), desc="Chunked Virtual Labeling"):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_samples)
        
        chunk_data = data_for_clustering[start_idx:end_idx]
        chunk_original_labels = original_labels[start_idx:end_idx]

        logger.info(f"  Processing Chunk {i+1}/{num_chunks} (Samples {start_idx}-{end_idx-1})")

        chunk_results = None
        chunk_best_params = None
        chunk_raw_labels = None
        chunk_final_labels = None
        try:
            # 1. Find best hyperparameters FOR THIS CHUNK
            # Find which of the full dataset's normal indices fall within this chunk
            chunk_known_normal_indices_mask = (consistent_known_normal_indices_full_dataset >= start_idx) & (consistent_known_normal_indices_full_dataset < end_idx)
            chunk_known_normal_indices_global = consistent_known_normal_indices_full_dataset[chunk_known_normal_indices_mask]
            # Adjust indices to be local to the chunk
            chunk_known_normal_indices_local = chunk_known_normal_indices_global - start_idx

            logger.info("    Finding best params for chunk...")
            # Note: We create a temporary, unique file_number for each chunk to avoid cache collisions
            # between different chunks for grid search progress.
            chunk_file_number = f"{file_number}_chunk_{i+1}"
            
            chunk_results = comprehensive_optimization_max_score(
                best_jaccard_algorithm,
                chunk_data,
                chunk_original_labels,
                file_type, chunk_file_number, # Use temporary chunk-specific identifier
                global_known_normal_samples_pca,
                num_processes_for_algo=num_processes,
                known_normal_idx=chunk_known_normal_indices_local,
                is_chunk_processing=True  # Flag to indicate this is a chunk
            )
            chunk_best_params = chunk_results.get('best_params')
            
            if not chunk_best_params:
                logger.warning(f"    Could not find best params for chunk {i+1}. Labeling as anomalous (-1).")
                # This part will be filled with -1 by default, so we can just continue
                continue
            logger.info(f"    Best params for chunk: {chunk_best_params}")

            # 2. Run clustering on the chunk with its best params
            chunk_raw_labels = run_single_clustering(
                best_jaccard_algorithm,
                chunk_data,
                chunk_best_params,
                aligned_original_labels=chunk_original_labels,
                global_known_normal_samples_pca=global_known_normal_samples_pca
            )

            if chunk_raw_labels is None:
                logger.warning(f"    Clustering failed for chunk {i+1}. Labeling as anomalous (-1).")
                continue
                
            # 3. Apply CNI with the FIXED global threshold
            chunk_final_labels, _, _ = clustering_nomal_identify(
                data_features_for_clustering=chunk_data,
                clusters_assigned=chunk_raw_labels,
                original_labels_aligned=chunk_original_labels,
                global_known_normal_samples_pca=global_known_normal_samples_pca,
                threshold_value=best_global_threshold, # Using the fixed global threshold
                num_processes_for_algo=num_processes,
                data_for_clustering=chunk_data,
                known_normal_idx=chunk_known_normal_indices_local,
                normal_data_batch_size=normal_data_batch_size
            )
            
            # 4. Store the results for the current chunk
            final_labels_full[start_idx:end_idx] = chunk_final_labels
            # Optional per-chunk save to CSV (checkpointing)
            if chunk_save_path and raw_data_for_save is not None:
                try:
                    chunk_df = raw_data_for_save.iloc[start_idx:end_idx].copy()
                    chunk_df['cluster'] = chunk_final_labels
                    chunk_df['adjusted_cluster'] = 1 - chunk_final_labels
                    write_header = (i == 0)
                    chunk_df.to_csv(chunk_save_path, mode='w' if write_header else 'a', header=write_header, index=False)
                    logger.info(f"    Saved chunk {i+1}/{num_chunks} to {chunk_save_path} (rows {start_idx}-{end_idx-1}).")
                except Exception as e:
                    logger.warning(f"    [Checkpoint] Failed to save chunk {i+1}: {e}")
        finally:
            # Release large per-chunk objects and force GC
            chunk_data = None
            chunk_original_labels = None
            chunk_results = None
            chunk_best_params = None
            chunk_raw_labels = None
            chunk_final_labels = None
            gc.collect()

    # Clean up temporary chunk progress files after all chunks are done
    logger.info("Cleaning up temporary chunk progress files...")
    for i in range(num_chunks):
        chunk_file_number = f"{file_number}_chunk_{i+1}"
        # Use the specific temp hash function to find the files created during the run
        temp_hash = generate_temp_chunk_hash(file_type, chunk_file_number)
        
        # Construct file paths using the same logic as the optimization functions
        # This is safer than a broad glob or calling the main reset function
        progress_file = get_grid_search_progress_file_path(best_jaccard_algorithm, temp_hash)
        
        # Also find related diagnostics files, now with the new 'chunk_diagnostics_' prefix
        diag_pattern = f"../Dataset_ex/progress_tracking/chunk_diagnostics_{temp_hash}_*.csv"

        files_to_delete = [progress_file] + glob.glob(diag_pattern)

        for file_path in files_to_delete:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"  Removed temp file: {os.path.basename(file_path)}")
                except OSError as e:
                    logger.warning(f"  Could not remove temp file {os.path.basename(file_path)}: {e}")
    
    return final_labels_full