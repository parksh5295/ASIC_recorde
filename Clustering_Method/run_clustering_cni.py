import os
import csv
import gc
from datetime import datetime
import math
import logging
import pandas as pd
import numpy as np
import time
import multiprocessing

from utils.apply_labeling import apply_labeling_logic
from Clustering_Method.clustering_score import evaluate_clustering, evaluate_clustering_wos
from Modules.Clustering_Algorithm_Autotune import choose_clustering_algorithm
from Modules.Clustering_Algorithm_Nonautotune import choose_clustering_algorithm_Non_optimization

# get_autotune_imports imports
# from Modules.Clustering_Algorithm_Autotune import choose_clustering_algorithm
# from Modules.Clustering_Algorithm_Nonautotune import choose_clustering_algorithm_Non_optimization


# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def cleanup_memory():
    """Clean up memory by forcing garbage collection"""
    gc.collect()

# Conditional imports - loaded when needed
'''
def get_autotune_imports():
    """Lazy import for autotune functionality"""
    try:
        print("[DEBUG] Attempting to import Modules.Clustering_Algorithm_Autotune...")
        from Modules.Clustering_Algorithm_Autotune import choose_clustering_algorithm
        print("[DEBUG] Successfully imported choose_clustering_algorithm")
        
        print("[DEBUG] Attempting to import Modules.Clustering_Algorithm_Nonautotune...")
        from Modules.Clustering_Algorithm_Nonautotune import choose_clustering_algorithm_Non_optimization
        print("[DEBUG] Successfully imported choose_clustering_algorithm_Non_optimization")
        
        return choose_clustering_algorithm, choose_clustering_algorithm_Non_optimization
    except ImportError as e:
        print(f"[ERROR] Import error in get_autotune_imports: {e}")
        print(f"[ERROR] Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise e
'''

def get_threshold_optimization_progress_file_path(file_type, file_number, algorithm):
    """Get the threshold optimization progress file path"""
    return f"../Dataset_ex/progress_tracking/{file_type}_{file_number}_{algorithm}_threshold_progress.csv"
    

def load_threshold_optimization_progress(file_type, file_number, algorithm):
    """Load existing threshold optimization progress"""
    progress_file = get_threshold_optimization_progress_file_path(file_type, file_number, algorithm)
    
    threshold_jaccard_scores = {}
    threshold_cluster_counts = {}
    completed_chunks = set()
    
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header and len(header) >= 5:
                    for row in reader:
                        if len(row) >= 5:
                            chunk_idx = int(row[0])
                            threshold = float(row[1])
                            jaccard_score = float(row[2])
                            cluster_count = int(row[3])
                            
                            # Initialize threshold dictionaries if needed
                            if threshold not in threshold_jaccard_scores:
                                threshold_jaccard_scores[threshold] = []
                            if threshold not in threshold_cluster_counts:
                                threshold_cluster_counts[threshold] = []
                            
                            # Add scores
                            threshold_jaccard_scores[threshold].append(jaccard_score)
                            threshold_cluster_counts[threshold].append(cluster_count)
                            completed_chunks.add(chunk_idx)
                            
        except Exception as e:
            print(f"Warning: Could not load threshold optimization progress from {progress_file}: {e}")
    
    return threshold_jaccard_scores, threshold_cluster_counts, completed_chunks

def get_threshold_optimization_file_path(file_type, file_number, algorithm):
    """Get the threshold optimization results file path"""
    return f"../Dataset_ex/progress_tracking/{file_type}_{file_number}_{algorithm}_threshold_optimization.csv"

def load_threshold_optimization_results(file_type, file_number, algorithm):
    """Load existing threshold optimization results"""
    optimization_file = get_threshold_optimization_file_path(file_type, file_number, algorithm)
    
    if os.path.exists(optimization_file):
        try:
            with open(optimization_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header and len(header) >= 6:
                    row = next(reader, None)
                    if row and len(row) >= 6:
                        return {
                            'optimal_threshold': float(row[1]),
                            'jaccard_scores': eval(row[2]) if row[2] else {},
                            'cluster_counts': eval(row[3]) if row[3] else {},
                            'selected_chunks': eval(row[4]) if row[4] else [],
                            'timestamp': row[5]
                        }
        except Exception as e:
            print(f"Warning: Could not load threshold optimization results from {optimization_file}: {e}")
    
    return None

def select_random_chunks_for_optimization(total_chunks, num_chunks_to_select):
    """Select random chunks for threshold optimization"""
    if num_chunks_to_select >= total_chunks:
        return list(range(total_chunks))
    
    selected_chunks = sorted(np.random.choice(total_chunks, num_chunks_to_select, replace=False))
    return selected_chunks

def save_threshold_optimization_results(file_type, file_number, algorithm, optimal_threshold, 
                                      jaccard_scores, cluster_counts, selected_chunks):
    """Save threshold optimization results to CSV file"""
    optimization_file = get_threshold_optimization_file_path(file_type, file_number, algorithm)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(optimization_file), exist_ok=True)
    
    try:
        with open(optimization_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['algorithm', 'optimal_threshold', 'jaccard_scores', 'cluster_counts', 'selected_chunks', 'timestamp'])
            
            # Write results
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            writer.writerow([
                algorithm,
                optimal_threshold,
                str(jaccard_scores),
                str(cluster_counts),
                str(selected_chunks),
                timestamp
            ])
            
    except Exception as e:
        print(f"Warning: Could not save threshold optimization results to {optimization_file}: {e}")

def save_threshold_optimization_progress(file_type, file_number, algorithm, chunk_idx, threshold, jaccard_score, cluster_count):
    """Save threshold optimization progress for a single chunk-threshold combination"""
    progress_file = get_threshold_optimization_progress_file_path(file_type, file_number, algorithm)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)
    
    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(progress_file)
    
    try:
        with open(progress_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header if file is new
            if not file_exists:
                writer.writerow(['chunk_idx', 'threshold', 'jaccard_score', 'cluster_count', 'timestamp'])
            
            # Write progress data
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            writer.writerow([chunk_idx, threshold, jaccard_score, cluster_count, timestamp])
            
    except Exception as e:
        print(f"Warning: Could not save threshold optimization progress to {progress_file}: {e}")

def get_progress_file_path(file_type, file_number, algorithm):
    """Get the progress tracking file path for a specific dataset and algorithm"""
    return f"../Dataset_ex/progress_tracking/{file_type}_{file_number}_{algorithm}_progress.csv"

def save_progress(file_type, file_number, algorithm, chunk_idx, status, jaccard_scores=None):
    """Save progress to CSV file"""
    progress_file = get_progress_file_path(file_type, file_number, algorithm)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)
    
    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(progress_file)
    
    try:
        with open(progress_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header if file is new
            if not file_exists:
                writer.writerow(['chunk_idx', 'status', 'timestamp', 'jaccard_scores'])
            
            # Write progress data
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            jaccard_str = str(jaccard_scores) if jaccard_scores else ''
            writer.writerow([chunk_idx, status, timestamp, jaccard_str])
            
    except Exception as e:
        print(f"Warning: Could not save progress to {progress_file}: {e}")


def run_full_dataset_with_optimal_threshold(data, data_for_clustering, original_labels, algorithm,
                                           global_known_normal_samples_pca, autotune_enabled,
                                           num_processes_for_clustering_algo, optimal_cni_threshold):
    """Run full dataset clustering with pre-determined optimal threshold"""
    logger.info(f"Running full dataset clustering with optimal threshold: {optimal_cni_threshold}")
    
    chunk_size = 30000
    num_samples = data_for_clustering.shape[0]
    num_chunks = math.ceil(num_samples / chunk_size)
    logger.info(f"Total samples: {num_samples}, Chunk size: {chunk_size}, Number of chunks: {num_chunks}")
    
    final_predict_results_list = []
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_samples)
        current_chunk_data_np = data_for_clustering[start_idx:end_idx]
        current_chunk_original_labels_np = original_labels[start_idx:end_idx]
        
        logger.info(f"  Processing Chunk {i+1}/{num_chunks} with optimal threshold {optimal_cni_threshold}")
        
        if autotune_enabled:
            chunk_clustering_result, _ = choose_clustering_algorithm(
                data, current_chunk_data_np, current_chunk_original_labels_np, algorithm,
                global_known_normal_samples_pca=global_known_normal_samples_pca, 
                threshold_value=optimal_cni_threshold,
                num_processes_for_algo=num_processes_for_clustering_algo
            )
        else:
            chunk_clustering_result = choose_clustering_algorithm_Non_optimization(
                data, current_chunk_data_np, current_chunk_original_labels_np, algorithm,
                global_known_normal_samples_pca=global_known_normal_samples_pca, 
                threshold_value=optimal_cni_threshold,
                num_processes_for_clustering_algo=num_processes_for_clustering_algo
            )
        
        if 'Cluster_labeling' in chunk_clustering_result and chunk_clustering_result['Cluster_labeling'] is not None:
            labels_for_this_chunk = chunk_clustering_result['Cluster_labeling']
        else:
            logger.error(f"    [ERROR] Chunk {i+1} processing failed. Using empty array for this chunk.")
            labels_for_this_chunk = np.array([])
        
        final_predict_results_list.append(labels_for_this_chunk)
    
    if final_predict_results_list and not all(arr.size == 0 for arr in final_predict_results_list):
        valid_results_to_concat = [arr for arr in final_predict_results_list if arr.size > 0]
        if valid_results_to_concat:
            final_predict_results = np.concatenate(valid_results_to_concat)
            if len(final_predict_results) != num_samples:
                logger.warning(f"[WARNING] Length of concatenated final labels ({len(final_predict_results)}) does not match total samples ({num_samples})")
        else:
            logger.error("[ERROR] All chunks failed to produce labels.")
            final_predict_results = np.array([])
    else:
        logger.error("[ERROR] No valid prediction results to concatenate.")
        final_predict_results = np.array([])
    
    return final_predict_results, optimal_cni_threshold, None


def run_clustering_with_cni(data, data_for_clustering, original_labels, algorithm, 
                           global_known_normal_samples_pca, autotune_enabled, 
                           num_processes_for_clustering_algo, threshold_candidates, 
                           file_type=None, file_number=None, stable_cluster_bounds=None):
    """Run clustering with CNI threshold optimization (based on Data_Labeling.py chunking logic)"""
    
    # Define chunking parameters
    chunk_size = 30000
    num_samples = data_for_clustering.shape[0]
    num_chunks = math.ceil(num_samples / chunk_size)
    logger.info(f"Total samples: {num_samples}, Chunk size: {chunk_size}, Number of chunks: {num_chunks}")

    # Check if threshold optimization results already exist
    optimization_results = None
    if file_type and file_number:
        optimization_results = load_threshold_optimization_results(file_type, file_number, algorithm)
        if optimization_results:
            logger.info(f"Found existing threshold optimization results for {algorithm}: optimal_threshold={optimization_results['optimal_threshold']}")
            logger.info(f"Results from: {optimization_results['timestamp']}")
            # Use existing results and skip to Phase 3
            optimal_cni_threshold = optimization_results['optimal_threshold']
            return run_full_dataset_with_optimal_threshold(
                data, data_for_clustering, original_labels, algorithm,
                global_known_normal_samples_pca, autotune_enabled,
                num_processes_for_clustering_algo, optimal_cni_threshold
            )

    # Early termination parameters (define first)
    min_chunks_for_statistics = 5  # Minimum chunks needed for reliable statistics
    max_chunks_for_early_termination = 6  # Maximum chunks to process before checking for early termination (reduced for faster processing)
    convergence_threshold = 0.01  # Jaccard score convergence threshold

    # Load existing threshold optimization progress if available
    threshold_jaccard_scores_across_chunks = {thresh: [] for thresh in threshold_candidates}
    threshold_cluster_counts_across_chunks = {thresh: [] for thresh in threshold_candidates}
    completed_optimization_chunks = set()
    
    if file_type and file_number:
        # Load existing threshold optimization progress
        existing_jaccard_scores, existing_cluster_counts, completed_optimization_chunks = load_threshold_optimization_progress(file_type, file_number, algorithm)
        
        if existing_jaccard_scores or existing_cluster_counts:
            logger.info(f"Loaded existing threshold optimization progress: {len(completed_optimization_chunks)} chunks completed")
            threshold_jaccard_scores_across_chunks.update(existing_jaccard_scores)
            threshold_cluster_counts_across_chunks.update(existing_cluster_counts)
        else:
            logger.info("No existing threshold optimization progress found")

    # Select random chunks for threshold optimization
    selected_chunks_for_optimization = select_random_chunks_for_optimization(num_chunks, max_chunks_for_early_termination)
    logger.info(f"Selected {len(selected_chunks_for_optimization)} random chunks for threshold optimization: {selected_chunks_for_optimization}")

    # Store temporary labels for reuse
    chunk_threshold_labels_temp_storage = {}

    # Phase 1: Iterate through chunks and thresholds to collect Jaccard scores
    logger.info("Phase 1: Collecting Jaccard scores and temporary labels for each chunk and threshold...")
    
    # Pre-load clustering functions to avoid repeated imports
    '''
    try:
        choose_clustering_algorithm, choose_clustering_algorithm_Non_optimization = get_autotune_imports()
    except Exception as e:
        logger.error(f"Error loading clustering functions: {e}")
        raise e
    '''
    
    processed_chunks = 0
    for i in selected_chunks_for_optimization:
        # Skip chunks that are already completed for threshold optimization
        if i in completed_optimization_chunks:
            logger.info(f"  Skipping Chunk {i+1}/{num_chunks} (threshold optimization already completed)")
            continue
        start_chunk_time = time.time()
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_samples)
        
        current_chunk_data_np = data_for_clustering[start_idx:end_idx]
        current_chunk_original_labels_np = original_labels[start_idx:end_idx]

        logger.info(f"  Processing Chunk {i+1}/{num_chunks} (Samples {start_idx}-{end_idx-1}), Shape: {current_chunk_data_np.shape}")
        
        for current_threshold_in_chunk_loop in threshold_candidates:
            # Optimize logging frequency - only log every 2nd threshold
            if len(threshold_candidates) <= 2 or current_threshold_in_chunk_loop == threshold_candidates[0] or current_threshold_in_chunk_loop == threshold_candidates[-1]:
                logger.info(f"    Chunk {i+1}, Testing CNI threshold: {current_threshold_in_chunk_loop}")
            gmm_type_for_this_run = None 
            
            try:
                if autotune_enabled: 
                    temp_chunk_clustering_result, gmm_type_for_this_run = choose_clustering_algorithm(
                        data, current_chunk_data_np, current_chunk_original_labels_np, algorithm, 
                        global_known_normal_samples_pca=global_known_normal_samples_pca,
                        threshold_value=current_threshold_in_chunk_loop,
                        num_processes_for_algo=num_processes_for_clustering_algo
                    )
                else: 
                    temp_chunk_clustering_result = choose_clustering_algorithm_Non_optimization(
                        data, current_chunk_data_np, current_chunk_original_labels_np, algorithm, 
                        global_known_normal_samples_pca=global_known_normal_samples_pca,
                        threshold_value=current_threshold_in_chunk_loop,
                        num_processes_for_algo=num_processes_for_clustering_algo
                    )
            except NameError as e:
                if "choose_clustering_algorithm" in str(e):
                    logger.error(f"choose_clustering_algorithm not defined. autotune_enabled={autotune_enabled}")
                    logger.error(f"Available functions: choose_clustering_algorithm={choose_clustering_algorithm is not None}, choose_clustering_algorithm_Non_optimization={choose_clustering_algorithm_Non_optimization is not None}")
                    raise e
                else:
                    raise e

            if isinstance(temp_chunk_clustering_result, dict) and 'Cluster_labeling' in temp_chunk_clustering_result and temp_chunk_clustering_result['Cluster_labeling'] is not None:
                y_pred_chunk_current_thresh = temp_chunk_clustering_result['Cluster_labeling']

                if y_pred_chunk_current_thresh.size == current_chunk_original_labels_np.size and y_pred_chunk_current_thresh.size > 0:
                    chunk_threshold_labels_temp_storage[(i, current_threshold_in_chunk_loop)] = y_pred_chunk_current_thresh
                    chunk_metrics = evaluate_clustering_wos(current_chunk_original_labels_np, y_pred_chunk_current_thresh)
                    
                    micro_metrics_dict = chunk_metrics.get('average=micro', {})
                    current_jaccard_micro_chunk = micro_metrics_dict.get('jaccard', -1.0)

                    logger.info(f"      INFO: Thresh {current_threshold_in_chunk_loop} - Calculated Jaccard (micro): {current_jaccard_micro_chunk}")

                    if current_jaccard_micro_chunk != -1.0:
                         threshold_jaccard_scores_across_chunks[current_threshold_in_chunk_loop].append(current_jaccard_micro_chunk)
                         logger.info(f"        DEBUG: Thresh {current_threshold_in_chunk_loop} - Stored Jaccard. Current list for this thresh: {threshold_jaccard_scores_across_chunks[current_threshold_in_chunk_loop]}")
                         
                         # Store cluster count for convergence analysis
                         num_clusters = len(np.unique(y_pred_chunk_current_thresh))
                         threshold_cluster_counts_across_chunks[current_threshold_in_chunk_loop].append(num_clusters)
                         logger.info(f"        DEBUG: Thresh {current_threshold_in_chunk_loop} - Stored cluster count: {num_clusters}")
                         
                         # Save progress immediately after each threshold processing
                         if file_type and file_number:
                             save_threshold_optimization_progress(
                                 file_type, file_number, algorithm, i, 
                                 current_threshold_in_chunk_loop, current_jaccard_micro_chunk, num_clusters
                             )
                    else:
                        logger.info(f"        DEBUG: Thresh {current_threshold_in_chunk_loop} - Jaccard score is -1.0, not storing.")
                else: 
                    logger.warning(f"      WARN: Thresh {current_threshold_in_chunk_loop} - Label size mismatch or empty labels. No Jaccard calculated or stored for this run.")
            else: 
                logger.warning(f"      WARN: Thresh {current_threshold_in_chunk_loop} - 'Cluster_labeling' missing, None, or result not a dict.")
        
        end_chunk_time = time.time()
        logger.info(f"  Chunk {i+1} (threshold sweep) processed in {end_chunk_time - start_chunk_time:.2f}s.")
        
        # Mark this chunk as completed for threshold optimization
        completed_optimization_chunks.add(i)
        
        # Save general chunk progress
        if file_type and file_number:
            chunk_jaccard_scores = {}
            for thresh, scores in threshold_jaccard_scores_across_chunks.items():
                if scores:
                    chunk_jaccard_scores[thresh] = scores[-1]  # Get the last score for this chunk
            save_progress(file_type, file_number, algorithm, i, 'completed', chunk_jaccard_scores)
        
        processed_chunks += 1
        
        # Clean up memory after each chunk
        cleanup_memory()
        
        # Check for early termination after processing enough chunks
        if processed_chunks >= min_chunks_for_statistics and processed_chunks >= max_chunks_for_early_termination:
            logger.info(f"  Checking for early termination after {processed_chunks} chunks...")
            
            # Calculate current optimal threshold based on processed chunks
            current_optimal_threshold = 0.3
            current_best_jaccard = -1.0
            threshold_stability = {}
            
            for thresh_val, jaccard_list in threshold_jaccard_scores_across_chunks.items():
                if len(jaccard_list) >= min_chunks_for_statistics:
                    scores_np = np.array(jaccard_list)
                    q1 = np.percentile(scores_np, 25)
                    q3 = np.percentile(scores_np, 75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    filtered_scores = scores_np[(scores_np >= lower_bound) & (scores_np <= upper_bound)]
                    if filtered_scores.size > 0:
                        robust_avg = np.mean(filtered_scores)
                        robust_std = np.std(filtered_scores)
                        threshold_stability[thresh_val] = {
                            'mean': robust_avg,
                            'std': robust_std,
                            'cv': robust_std / robust_avg if robust_avg > 0 else float('inf')  # Coefficient of Variation
                        }
                        
                        if robust_avg > current_best_jaccard:
                            current_best_jaccard = robust_avg
                            current_optimal_threshold = thresh_val
            
            # Check for convergence: if the optimal threshold is stable
            if len(threshold_stability) >= len(threshold_candidates):
                optimal_cv = threshold_stability[current_optimal_threshold]['cv']
                logger.info(f"  Current optimal threshold: {current_optimal_threshold} (Jaccard: {current_best_jaccard:.4f}, CV: {optimal_cv:.4f})")
                
                # Check cluster count convergence for optimal threshold
                optimal_cluster_counts = threshold_cluster_counts_across_chunks.get(current_optimal_threshold, [])
                if len(optimal_cluster_counts) >= min_chunks_for_statistics:
                    cluster_counts_np = np.array(optimal_cluster_counts)
                    cluster_cv = np.std(cluster_counts_np) / np.mean(cluster_counts_np) if np.mean(cluster_counts_np) > 0 else float('inf')
                    cluster_range = np.max(cluster_counts_np) - np.min(cluster_counts_np)
                    
                    logger.info(f"  Cluster count convergence: mean={np.mean(cluster_counts_np):.1f}, std={np.std(cluster_counts_np):.1f}, CV={cluster_cv:.4f}, range={cluster_range}")
                    
                    # Check if cluster counts are stable
                    cluster_stable = cluster_cv < 0.1 and cluster_range <= 5  # CV < 10% and range <= 5
                    
                    if cluster_stable:
                        logger.info(f"  Cluster counts are stable! Will use constrained range for remaining chunks.")
                        # Store stable cluster range for later use
                        stable_cluster_mean = int(np.mean(cluster_counts_np))
                        stable_cluster_range = max(3, int(np.std(cluster_counts_np) * 2))  # 짹2? range, minimum 3
                        stable_cluster_bounds = (max(2, stable_cluster_mean - stable_cluster_range), 
                                               stable_cluster_mean + stable_cluster_range)
                        logger.info(f"  Stable cluster bounds: {stable_cluster_bounds}")
                    else:
                        stable_cluster_bounds = None
                        logger.info(f"  Cluster counts still unstable. Will use full range for remaining chunks.")
                else:
                    stable_cluster_bounds = None
                    logger.info(f"  Not enough cluster count data yet ({len(optimal_cluster_counts)} < {min_chunks_for_statistics})")
                
                # Check if all thresholds have low coefficient of variation (stable)
                all_stable = all(stats['cv'] < 0.05 for stats in threshold_stability.values())  # CV < 5%
                
                if all_stable:
                    logger.info(f"  All thresholds show stable results (CV < 5%). Early termination.")
                    # Store stable cluster bounds for Phase 3
                    if 'stable_cluster_bounds' in locals() and stable_cluster_bounds is not None:
                        logger.info(f"  Will use cluster bounds {stable_cluster_bounds} for remaining chunks.")
                    break
                else:
                    logger.info(f"  Some thresholds still unstable. Continuing...")
            
            # Fallback: if we have enough data, proceed anyway
            total_scores = sum(len(scores) for scores in threshold_jaccard_scores_across_chunks.values())
            if total_scores >= min_chunks_for_statistics * len(threshold_candidates):
                logger.info(f"  Sufficient data collected ({total_scores} scores). Proceeding to threshold optimization...")
                break
            else:
                logger.info(f"  Not enough data yet ({total_scores} scores). Continuing...")
        
        # Also check if we have processed enough chunks for basic statistics
        elif processed_chunks >= min_chunks_for_statistics:
            total_scores = sum(len(scores) for scores in threshold_jaccard_scores_across_chunks.values())
            if total_scores >= min_chunks_for_statistics * len(threshold_candidates):
                logger.info(f"  Sufficient data collected after {processed_chunks} chunks ({total_scores} scores). Checking for convergence...")
                
                # Advanced convergence check: trend analysis and stability
                if processed_chunks >= 10:  # Need at least 10 chunks for convergence check
                    recent_scores = {}
                    convergence_metrics = {}
                    
                    for thresh_val, jaccard_list in threshold_jaccard_scores_across_chunks.items():
                        if len(jaccard_list) >= 8:  # Need at least 8 recent scores for trend analysis
                            recent_scores[thresh_val] = jaccard_list[-8:]  # Last 8 scores
                            
                            # Calculate trend (slope of linear regression)
                            x = np.arange(len(recent_scores[thresh_val]))
                            y = np.array(recent_scores[thresh_val])
                            slope = np.polyfit(x, y, 1)[0]  # Linear trend
                            
                            # Calculate stability metrics
                            recent_std = np.std(recent_scores[thresh_val])
                            recent_cv = recent_std / np.mean(recent_scores[thresh_val]) if np.mean(recent_scores[thresh_val]) > 0 else float('inf')
                            
                            convergence_metrics[thresh_val] = {
                                'slope': slope,
                                'std': recent_std,
                                'cv': recent_cv,
                                'trend_stable': abs(slope) < 0.001  # Very small trend
                            }
                    
                    if len(convergence_metrics) == len(threshold_candidates):
                        # Check convergence conditions
                        all_trends_stable = all(metrics['trend_stable'] for metrics in convergence_metrics.values())
                        all_cv_low = all(metrics['cv'] < 0.03 for metrics in convergence_metrics.values())  # CV < 3%
                        
                        logger.info(f"  Convergence check: trends_stable={all_trends_stable}, cv_low={all_cv_low}")
                        for thresh_val, metrics in convergence_metrics.items():
                            logger.info(f"    Threshold {thresh_val}: slope={metrics['slope']:.6f}, CV={metrics['cv']:.4f}")
                        
                        if all_trends_stable and all_cv_low:
                            logger.info(f"  Strong convergence detected after {processed_chunks} chunks. Early termination.")
                            break
                        elif all_cv_low:  # At least stable variance
                            logger.info(f"  Moderate convergence detected after {processed_chunks} chunks. Early termination.")
                            break

    # Phase 2: Determine Optimal CNI Threshold (IQR Outlier Removal + Mean)
    logger.info("Phase 2: Determining Optimal CNI Threshold...")
    logger.info(f"  Processed {processed_chunks} chunks out of {num_chunks} total chunks ({processed_chunks/num_chunks*100:.1f}%)")
    optimal_cni_threshold = 0.3  # Default if all else fails or no scores
    best_robust_average_jaccard = -1.0
    stable_cluster_bounds = None  # Initialize stable cluster bounds

    for thresh_val, jaccard_list in threshold_jaccard_scores_across_chunks.items():
        if not jaccard_list:
            logger.info(f"  Threshold {thresh_val}: No Jaccard scores recorded.")
            continue

        scores_np = np.array(jaccard_list)
        q1 = np.percentile(scores_np, 25)
        q3 = np.percentile(scores_np, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Filter out outliers
        filtered_scores = scores_np[(scores_np >= lower_bound) & (scores_np <= upper_bound)]
        
        if filtered_scores.size > 0:
            robust_average_jaccard = np.mean(filtered_scores)
            logger.info(f"  Threshold {thresh_val}: Robust Avg Jaccard (micro) = {robust_average_jaccard:.4f} (from {filtered_scores.size}/{scores_np.size} scores after outlier removal)")
            if robust_average_jaccard > best_robust_average_jaccard:
                best_robust_average_jaccard = robust_average_jaccard
                optimal_cni_threshold = thresh_val
        else:
            # If all scores were outliers, fall back to mean of original scores
            original_mean = np.mean(scores_np)
            logger.info(f"  Threshold {thresh_val}: All scores considered outliers. Original mean Jaccard = {original_mean:.4f} (from {scores_np.size} scores)")
            if original_mean > best_robust_average_jaccard:
                 best_robust_average_jaccard = original_mean
                 optimal_cni_threshold = thresh_val
                 logger.info(f"    (Using original mean as it's currently the best overall: {original_mean:.4f}) ")

    logger.info(f"Optimal CNI Threshold selected: {optimal_cni_threshold} with best robust average Jaccard (micro): {best_robust_average_jaccard:.4f}")
    
    # Save threshold optimization results
    if file_type and file_number:
        save_threshold_optimization_results(
            file_type, file_number, algorithm, optimal_cni_threshold,
            threshold_jaccard_scores_across_chunks, threshold_cluster_counts_across_chunks,
            selected_chunks_for_optimization
        )
        logger.info(f"Saved threshold optimization results for {algorithm}")
        
        # Clean up progress file since optimization is complete
        progress_file = get_threshold_optimization_progress_file_path(file_type, file_number, algorithm)
        if os.path.exists(progress_file):
            try:
                os.remove(progress_file)
                logger.info(f"Cleaned up threshold optimization progress file: {progress_file}")
            except Exception as e:
                logger.warning(f"Could not remove progress file {progress_file}: {e}")

    # Phase 3: Re-process chunks with Optimal CNI Threshold to get final labels
    logger.info(f"Phase 3: Assembling final predictions by re-processing chunks with optimal_cni_threshold = {optimal_cni_threshold}...")
    final_predict_results_list = []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_samples)
        current_chunk_data_np = data_for_clustering[start_idx:end_idx]
        current_chunk_original_labels_np = original_labels[start_idx:end_idx]
        
        final_gmm_type_for_chunk = None
        final_params_for_chunk = {}
        labels_for_this_chunk_optimal = None

        # Check if we already processed this chunk with the optimal threshold
        chunk_key = (i, optimal_cni_threshold)
        if chunk_key in chunk_threshold_labels_temp_storage:
            # Reuse existing labels from Phase 1
            labels_for_this_chunk_optimal = chunk_threshold_labels_temp_storage[chunk_key]
            logger.info(f"  Chunk {i+1}: Reusing labels from Phase 1 (optimal threshold {optimal_cni_threshold})")
        else:
            # Process this chunk for the first time with optimal threshold
            logger.info(f"  Chunk {i+1}: Processing with optimal threshold {optimal_cni_threshold} for the first time")
            
            # Use stable cluster bounds if available
            if stable_cluster_bounds is not None:
                logger.info(f"  Chunk {i+1}: Using stable cluster bounds {stable_cluster_bounds} for faster processing")
                # Modify the clustering algorithm to use constrained cluster range
                # This would require modifying the choose_clustering_algorithm function
                # For now, we'll log the information
                
            if autotune_enabled:
                final_chunk_clustering_result, final_gmm_type_for_chunk = choose_clustering_algorithm(
                    data, current_chunk_data_np, current_chunk_original_labels_np, algorithm,
                    global_known_normal_samples_pca=global_known_normal_samples_pca, threshold_value=optimal_cni_threshold,
                    num_processes_for_algo=num_processes_for_clustering_algo
                )
            else:
                final_chunk_clustering_result = choose_clustering_algorithm_Non_optimization(
                    data, current_chunk_data_np, current_chunk_original_labels_np, algorithm,
                    global_known_normal_samples_pca=global_known_normal_samples_pca, threshold_value=optimal_cni_threshold,
                    num_processes_for_clustering_algo=num_processes_for_clustering_algo
                )
        
            if 'Cluster_labeling' in final_chunk_clustering_result and final_chunk_clustering_result['Cluster_labeling'] is not None:
                labels_for_this_chunk_optimal = final_chunk_clustering_result['Cluster_labeling']
                final_params_for_chunk = final_chunk_clustering_result.get('Best_parameter_dict', {})
            else:
                logger.error(f"    [ERROR] Chunk {i+1} processing with optimal threshold failed to produce labels. Using empty array for this chunk.")
                labels_for_this_chunk_optimal = np.array([]) 

        final_predict_results_list.append(labels_for_this_chunk_optimal)

    if final_predict_results_list and not all(arr.size == 0 for arr in final_predict_results_list):
        valid_results_to_concat = [arr for arr in final_predict_results_list if arr.size > 0]
        if valid_results_to_concat:
            final_predict_results = np.concatenate(valid_results_to_concat)
            if len(final_predict_results) != num_samples:
                 logger.warning(f"[WARNING Phase 3] Length of concatenated final labels ({len(final_predict_results)}) does not match total samples ({num_samples}) after potentially excluding failed chunks.")
        else:
            logger.error("[ERROR Phase 3] All chunks failed to produce labels in Phase 3. final_predict_results will be empty.")
            final_predict_results = np.array([])
    else:
        logger.error("[ERROR Phase 3] No valid prediction results to concatenate after Phase 3.")
        final_predict_results = np.array([])

    return final_predict_results, optimal_cni_threshold, stable_cluster_bounds
