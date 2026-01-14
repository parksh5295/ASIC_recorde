# with Elbow method
# input 'data' is X or X_reduced
# 'clustering' is 'clustering_algorithm'.fit_predict(data)
# output(optimal_k): 'Only' Optimal number of cluster by data

# Some Clustering Algorihtm; Kmeans, Kmedians, GMM, SGMM, FCM, CK requires additional work to tune the number of clusters.

import numpy as np
from Clustering_Method.common_clustering import get_clustering_function
import multiprocessing # Added for parallel processing
import os # For os.cpu_count()
from tqdm import tqdm
import csv
import sys
from datetime import datetime
import hashlib


def generate_data_hash(X):
    """Generate a hash based on data characteristics for unique identification"""
    if X is None or len(X) == 0:
        return "empty_data"
    
    # Use data shape, sample of data, and basic statistics for hashing
    data_info = f"{X.shape}_{X.dtype}_{np.mean(X):.6f}_{np.std(X):.6f}_{X[0,0] if X.size > 0 else 0:.6f}"
    return hashlib.md5(data_info.encode()).hexdigest()[:8]


def get_elbow_progress_file_path(clustering_algorithm, max_clusters, data_hash):
    """Get the elbow progress tracking file path"""
    return f"../Dataset_ex/progress_tracking/elbow_{data_hash}_{clustering_algorithm}_{max_clusters}_progress.csv"


def load_elbow_progress(clustering_algorithm, max_clusters, data_hash):
    """Load existing elbow progress from CSV file"""
    progress_file = get_elbow_progress_file_path(clustering_algorithm, max_clusters, data_hash)
    completed_k_values = set()
    
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    if len(row) >= 2:
                        k_value = int(row[0])
                        completed_k_values.add(k_value)
        except Exception as e:
            print(f"Warning: Could not load elbow progress file {progress_file}: {e}")
    
    return completed_k_values


def save_elbow_progress(clustering_algorithm, max_clusters, k_value, score, data_hash):
    """Save elbow progress to CSV file"""
    progress_file = get_elbow_progress_file_path(clustering_algorithm, max_clusters, data_hash)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)
    
    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(progress_file)
    
    try:
        with open(progress_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header if file is new
            if not file_exists:
                writer.writerow(['k_value', 'score', 'timestamp'])
            
            # Write progress data
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            writer.writerow([k_value, score, timestamp])
            
    except Exception as e:
        print(f"Warning: Could not save elbow progress to {progress_file}: {e}")


def calculate_optimal_k_from_scores(wcss_or_bic, cluster_range):
    """Calculate optimal k from WCSS/BIC scores using elbow method"""
    if len(wcss_or_bic) < 3:
        return 2
    
    # Calculate second derivative (elbow detection)
    if len(wcss_or_bic) >= 3:
        # Calculate first derivative
        first_derivative = np.diff(wcss_or_bic)
        # Calculate second derivative
        second_derivative = np.diff(first_derivative)
        
        # Find the elbow point (maximum second derivative)
        elbow_idx = np.argmax(second_derivative) + 2  # +2 because we took two diffs
        optimal_k = list(cluster_range)[elbow_idx]
        
        return optimal_k
    else:
        return 2


def Elbow_choose_clustering_algorithm(data, X, clustering_algorithm, n_clusters, parameter_dict, GMM_type="normal", num_processes_for_algo=1):   # X: Encoding and embedding, post-PCA, post-delivery
    pre_clustering_func = get_clustering_function(clustering_algorithm)

    # Parameters are expected to be in parameter_dict passed from Elbow_method
    if clustering_algorithm == 'Kmeans':
        algorithm_params = {
            'random_state': parameter_dict['random_state'],
            # 'n_init': parameter_dict['n_init'],
            'n_init': 10, # Use a smaller, fixed n_init for faster k selection in Elbow for K-Means
            'num_processes_for_algo': num_processes_for_algo
        }
        clustering = pre_clustering_func(data, X, n_clusters, **algorithm_params)
    elif clustering_algorithm == 'GMM': # General GMM, distinct from SGMM
        algorithm_params = {
            'random_state': parameter_dict['random_state'],
            'GMM_type': GMM_type, # GMM_type is specific to pre_clustering_GMM
            # 'reg_covar': parameter_dict['reg_covar'],
            'n_init': parameter_dict.get('n_init', 1), # Ensure n_init is passed for general GMM too
            'num_processes_for_algo': num_processes_for_algo
        }
        clustering = pre_clustering_func(data, X, n_clusters, **algorithm_params)
    elif clustering_algorithm == 'SGMM': # Spherical GMM
        algorithm_params = {
            'random_state': parameter_dict['random_state'],
            'reg_covar': parameter_dict['reg_covar'],
            'n_init': parameter_dict.get('n_init', 1), # Added n_init from parameter_dict
            'num_processes_for_algo': num_processes_for_algo
        }
        clustering = pre_clustering_func(data, X, n_clusters, **algorithm_params)
    elif clustering_algorithm in ['FCM', 'CK']:
        if clustering_algorithm == 'CK':
            # For CK, ensure n_init from the parameter_dict (which is 30 from Elbow_method's base_parameter_dict)
            # is passed as n_init_for_ck to pre_clustering_CK
            algorithm_params_ck = {
                'n_init_for_ck': parameter_dict.get('n_init', 30), # Default to 30 if not in dict somehow
                'num_processes_for_algo': num_processes_for_algo
            }
            clustering = pre_clustering_func(data, X, n_clusters, **algorithm_params_ck)
        else: # FCM or any other in this list that doesn't take n_init explicitly here, but pre_clustering_FCM should take num_processes_for_algo
            algorithm_params_fcm = {
                'num_processes_for_algo': num_processes_for_algo
            }
            clustering = pre_clustering_func(data, X, n_clusters, **algorithm_params_fcm)
    elif clustering_algorithm == 'Gmeans':
        algorithm_params = {
            'random_state': parameter_dict['random_state'],
            'max_clusters': parameter_dict['max_clusters'], # GMeans uses max_clusters from dict
            'tol': parameter_dict['tol'],
            'n_init': parameter_dict.get('n_init', 30), # Added n_init for Gmeans
            'num_processes_for_algo': num_processes_for_algo
        }
        # n_clusters (k from loop) is not passed directly to pre_clustering_Gmeans
        clustering = pre_clustering_func(data, X, **algorithm_params)
    elif clustering_algorithm == 'Xmeans':
        algorithm_params = {
            'random_state': parameter_dict['random_state'],
            'max_clusters': parameter_dict['max_clusters'], # XMeans uses max_clusters from dict
            'n_init': parameter_dict.get('n_init', 30), # Added n_init for Xmeans
            'num_processes_for_algo': num_processes_for_algo
        }
        # n_clusters (k from loop) is not passed directly to pre_clustering_Xmeans
        clustering = pre_clustering_func(data, X, **algorithm_params)
    elif clustering_algorithm == 'DBSCAN':
        algorithm_params = {
            # DBSCAN does not use random_state in its sklearn implementation
            'eps': parameter_dict['eps'],
            'count_samples': parameter_dict['count_samples'], # maps to min_samples
            'num_processes_for_algo': num_processes_for_algo
        }
        # n_clusters (k from loop) is not passed to pre_clustering_DBSCAN
        clustering = pre_clustering_func(data, X, **algorithm_params)
    elif clustering_algorithm == 'MShift':
        algorithm_params = {
            'random_state': parameter_dict['random_state'],
            'quantile': parameter_dict['quantile'],
            'n_samples': parameter_dict['n_samples'],
            'num_processes_for_algo': num_processes_for_algo
        }
        # n_clusters (k from loop) is not passed to pre_clustering_MShift
        clustering = pre_clustering_func(data, X, **algorithm_params)
    elif clustering_algorithm == 'NeuralGas':
        algorithm_params = {
            # NeuralGas pre_clustering doesn't take random_state in its current signature
            'n_start_nodes': parameter_dict['n_start_nodes'],
            'max_nodes': parameter_dict['max_nodes'],
            'step': parameter_dict['step'],
            'max_edge_age': parameter_dict['max_edge_age'],
            'num_processes_for_algo': num_processes_for_algo
        }
        # n_clusters (k from loop) is not passed to pre_clustering_NeuralGas
        clustering = pre_clustering_func(data, X, **algorithm_params)
    else: # KMedians and any other algorithm that takes n_clusters and random_state by default
        algorithm_params = {
            'random_state': parameter_dict['random_state'],
            'num_processes_for_algo': num_processes_for_algo
        }
        clustering = pre_clustering_func(data, X, n_clusters, **algorithm_params)

    return clustering


def Elbow_method(data, X, clustering_algorithm, max_clusters, parameter_dict=None, num_processes_for_algo=None):
    # Maintain complete parameter_dict for compatibility and to ensure all necessary params are available
    base_parameter_dict = {
        'random_state': 42,
        'n_init': 30, # For KMeans primarily
        'max_clusters': 1000, # For GMeans, XMeans primarily
        'tol': 1e-4, # For GMeans primarily
        'eps': 0.5, # For DBSCAN primarily
        'count_samples': 5, # For DBSCAN primarily (as min_samples)
        'quantile': 0.2, # For MShift primarily
        'n_samples': 500, # For MShift primarily
        'n_start_nodes': 2, # For NeuralGas primarily
        'max_nodes': 50, # For NeuralGas primarily
        'step': 0.2, # For NeuralGas primarily
        'max_edge_age': 50, # For NeuralGas primarily
        'epochs': 300, # For CANNwKNN (if used here, but usually tuned in Grid_search_all)
        'batch_size': 256, # For CANNwKNN
        'n_neighbors': 5, # For CANNwKNN
        'reg_covar': 1e-6 # For GMM and SGMM
    }
    if parameter_dict is not None:
        base_parameter_dict.update(parameter_dict) # Update with any user-provided params
    
    current_parameter_dict = base_parameter_dict.copy()

    wcss_or_bic = []  # Store WCSS (inertia) or BIC by number of clusters
    cluster_range = range(2, max_clusters + 1) # Start from 2 clusters for GMM/SGMM BIC calculation and most algos
    if not cluster_range:
        # Handle edge case where max_clusters < 2
        print("Warning: max_clusters is less than 2. Elbow method requires at least 2 clusters. Returning default k=2.")
        return {
            'optimal_cluster_n': 2,
            'best_parameter_dict': current_parameter_dict
        }

    # Generate data hash for unique identification
    data_hash = generate_data_hash(X)
    print(f"Data hash for progress tracking: {data_hash}")

    # Load existing progress
    completed_k_values = load_elbow_progress(clustering_algorithm, max_clusters, data_hash)
    if completed_k_values:
        print(f"Found existing elbow progress: {len(completed_k_values)} k-values already completed")
        print(f"Completed k-values: {sorted(list(completed_k_values))}")
    else:
        print("No existing elbow progress found")

    # Prepare tasks for parallel execution (skip completed k-values)
    tasks = []
    for k_task in cluster_range:
        if k_task not in completed_k_values:
            # Pass num_processes_for_algo to _calculate_score_for_k via args_tuple
            tasks.append((k_task, data, X, clustering_algorithm, current_parameter_dict, num_processes_for_algo))
    
    if not tasks:
        print("All k-values already completed. Loading results from progress file...")
        # Load all results from progress file
        k_score_pairs = []
        progress_file = get_elbow_progress_file_path(clustering_algorithm, max_clusters, data_hash)
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader, None)  # Skip header
                    for row in reader:
                        if len(row) >= 2:
                            k_value = int(row[0])
                            score = float(row[1])
                            k_score_pairs.append((k_value, score))
            except Exception as e:
                print(f"Error loading progress file: {e}")
                return {
                    'optimal_cluster_n': 2,
                    'best_parameter_dict': current_parameter_dict
                }
        
        # Sort results by k to ensure wcss_or_bic is in the correct order
        k_score_pairs.sort(key=lambda x: x[0])
        
        # Populate wcss_or_bic from loaded results
        for k, score in k_score_pairs:
            wcss_or_bic.append(score)
        
        # Continue with normal elbow calculation
        if not wcss_or_bic:
            print("No valid scores found in progress file. Returning default k=2.")
            return {
                'optimal_cluster_n': 2,
                'best_parameter_dict': current_parameter_dict
            }
        
        # Calculate optimal k using elbow method
        optimal_k = calculate_optimal_k_from_scores(wcss_or_bic, cluster_range)
        validation_candidates = get_validation_candidates(optimal_k, max_clusters)
        
        return {
            'optimal_cluster_n': optimal_k,
            'best_parameter_dict': current_parameter_dict,
            'validation_candidates': validation_candidates
        }

    # This will store (k, score) tuples, possibly out of order from parallel execution
    k_score_pairs = [] 

    if tasks:
        # Determine number of processes for the Pool
        if num_processes_for_algo == 0: # Use all available CPUs
            pool_processes = os.cpu_count()
            if pool_processes is None: # Fallback if os.cpu_count() returns None
                 pool_processes = 1 
        elif num_processes_for_algo is not None and num_processes_for_algo > 0: # User specified positive number
            pool_processes = num_processes_for_algo
        else: # Default: None or invalid value, use half CPUs or multiprocessing's default, ensure at least 1
            cpu_cores = os.cpu_count()
            if cpu_cores:
                pool_processes = max(1, cpu_cores // 2)
            else: # Fallback if os.cpu_count() returns None
                pool_processes = max(1, multiprocessing.cpu_count() // 2 if multiprocessing.cpu_count() else 1)
        
        # Ensure pool_processes does not exceed the number of tasks
        pool_processes = min(pool_processes, len(tasks))
        if pool_processes == 0 and len(tasks) > 0: pool_processes = 1 # Ensure at least 1 process if tasks exist

        print(f"[Elbow_method] Using {pool_processes} processes for {len(tasks)} k-values (num_processes_for_algo={num_processes_for_algo}).")
        try:
            with multiprocessing.Pool(processes=pool_processes) as pool:
                # pool.map will preserve order if tasks are ordered, 
                # but _calculate_score_for_k returns (k, score) so we can sort later if needed.
                # Using map as _calculate_score_for_k takes a single tuple argument.
                # MODIFIED: Wrap the pool.map iterator with tqdm for a progress bar and save progress
                result_iterator = pool.imap_unordered(_calculate_score_for_k, tasks)
                pbar = tqdm(result_iterator, total=len(tasks), desc=f"[Elbow for {clustering_algorithm}]")
                
                k_score_pairs = []
                try:
                    for result in pbar:
                        k_score_pairs.append(result)
                        # Save progress immediately after each k-value completion
                        k_value, score = result
                        save_elbow_progress(clustering_algorithm, max_clusters, k_value, score, data_hash)
                except KeyboardInterrupt:
                    print(f"\nThe [Elbow for {clustering_algorithm}] job was stopped by the user.")
                    print(f"[Elbow for {clustering_algorithm}] Interim saved file: Dataset_ex/progress_tracking/elbow_{data_hash}_{clustering_algorithm}_{max_clusters}_progress.csv")
                    print(f"[Elbow for {clustering_algorithm}] When restarting, it will resume from the point where it was interrupted.")
                    # Terminate pool and return current results
                    pool.terminate()
                    pool.join()
                    # Continue with current results

        except Exception as e:
            print(f"Error during parallel Elbow method processing: {e}. Falling back to sequential.")
            # Fallback to sequential execution
            k_score_pairs = [] # Clear partially filled results from try block
            #for k_seq in cluster_range:
            # MODIFIED: Add tqdm to the sequential fallback loop as well
            try:
                for task in tqdm(tasks, desc=f"[Elbow for {clustering_algorithm} (Sequential)]"):
                    # Call helper directly, passing num_processes_for_algo
                    #_, score_seq = _calculate_score_for_k((k_seq, data, X, clustering_algorithm, current_parameter_dict, num_processes_for_algo)) 
                    #k_score_pairs.append((k_seq, score_seq))
                    score_seq = _calculate_score_for_k(task) 
                    k_score_pairs.append(score_seq)
                    # Save progress immediately after each k-value completion
                    k_value, score = score_seq
                    save_elbow_progress(clustering_algorithm, max_clusters, k_value, score, data_hash)
            except KeyboardInterrupt:
                print(f"\nThe [Elbow for {clustering_algorithm} (Sequential)] job was stopped by the user.")
                print(f"[Elbow for {clustering_algorithm} (Sequential)] Interim saved file: Dataset_ex/progress_tracking/elbow_{data_hash}_{clustering_algorithm}_{max_clusters}_progress.csv")
                print(f"[Elbow for {clustering_algorithm} (Sequential)] When restarting, it will resume from the point where it was interrupted.")
                # Continue with current results
    else:
        print("[Elbow_method] No tasks to run for Elbow method (max_clusters might be < 2).")
        # This case is also handled by the `if not cluster_range:` block earlier, 
        # but good to have a log if tasks list is empty for other reasons.

    # Load existing completed results and merge with new results
    if completed_k_values:
        progress_file = get_elbow_progress_file_path(clustering_algorithm, max_clusters, data_hash)
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader, None)  # Skip header
                    for row in reader:
                        if len(row) >= 2:
                            k_value = int(row[0])
                            score = float(row[1])
                            # Only add if not already in k_score_pairs (avoid duplicates)
                            if not any(k == k_value for k, s in k_score_pairs):
                                k_score_pairs.append((k_value, score))
            except Exception as e:
                print(f"Warning: Could not load existing results: {e}")
    
    # Sort results by k to ensure wcss_or_bic is in the correct order for diff calculations
    k_score_pairs.sort(key=lambda x: x[0])
    
    # Populate wcss_or_bic from sorted results
    wcss_or_bic = [score for k_val, score in k_score_pairs]
    
    if len(wcss_or_bic) < 2: # Need at least 2 points to calculate differences
        print("Warning: Not enough valid scores to determine elbow. Returning default k=2 or max_clusters if 1.")
        optimal_k = 2 if max_clusters >= 2 else max_clusters if max_clusters ==1 else 1 # Ensure k is at least 1
        if max_clusters == 0 : optimal_k =1 # avoid k=0
        return {
            'optimal_cluster_n': optimal_k,
            'best_parameter_dict': current_parameter_dict
        }
    
    # Rate of change of slope; For GMM/SGMM, lower BIC is better → so reverse slope logic by finding max decrease (min -diff)
    # For WCSS (inertia), lower is better, so we want to find where the decrease starts to slow down (max second derivative)
    diff_scores = np.diff(wcss_or_bic)
    if len(diff_scores) < 1:
        print("Warning: Not enough score differences to determine elbow. Returning default k=2 or max_clusters if 1.")
        optimal_k = 2 if max_clusters >= 2 else max_clusters if max_clusters ==1 else 1
        if max_clusters == 0 : optimal_k =1
        return {
            'optimal_cluster_n': optimal_k,
            'best_parameter_dict': current_parameter_dict
        }

    second_diff = np.diff(diff_scores) 
    if len(second_diff) == 0:
        # If only two k values were tested (e.g., k=2, k=3), second_diff will be empty.
        # Default to the k with the better score or a predefined k.
        # For GMM/SGMM (BIC, lower is better), choose k with min score.
        # For WCSS (inertia, lower is better), choose k with min score.
        optimal_k_index = np.argmin(wcss_or_bic)
        optimal_k = cluster_range[optimal_k_index] 
        print("Warning: Only two k values tested or not enough points for second derivative. Optimal k chosen by min score.")
    else:
        # Apply dynamic threshold based on data complexity
        data_complexity = calculate_data_complexity(X)
        sensitivity_factor = get_sensitivity_factor(data_complexity, clustering_algorithm)
        
        if clustering_algorithm in ['GMM', 'SGMM']:
            # For BIC, we look for the point where the BIC score starts to increase after decreasing, or decreases less steeply.
            optimal_k_index = find_significant_elbow(second_diff, sensitivity_factor)
        else:
            # For WCSS (inertia), we look for the point where the rate of decrease slows down.
            optimal_k_index = find_significant_elbow(second_diff, sensitivity_factor)
        
        optimal_k = cluster_range[optimal_k_index] # Convert index back to k value

    # Apply minimum cluster guarantee for complex datasets
    min_k = get_minimum_clusters_guarantee(clustering_algorithm, len(X))
    if optimal_k < min_k:
        print(f"[Elbow_method] Optimal k={optimal_k} is below minimum threshold {min_k}. Using {min_k}.")
        optimal_k = min_k

    # Optional: Return additional candidates for validation
    # This allows the caller to perform Jaccard coefficient validation
    validation_candidates = get_validation_candidates(optimal_k, max_clusters)
    
    # Clean up progress file since optimization is complete
    progress_file = get_elbow_progress_file_path(clustering_algorithm, max_clusters, data_hash)
    if os.path.exists(progress_file):
        try:
            os.remove(progress_file)
            print(f"Cleaned up elbow progress file: {progress_file}")
        except Exception as e:
            print(f"Warning: Could not remove progress file {progress_file}: {e}")
    
    return {
        'optimal_cluster_n': optimal_k,
        'best_parameter_dict': current_parameter_dict,
        'validation_candidates': validation_candidates  # Additional k values to test
    }

def get_minimum_clusters_guarantee(clustering_algorithm, n_samples):
    """Get minimum number of clusters based on algorithm and sample size"""
    # Base minimum based on sample size
    if n_samples < 1000:
        base_min = 5
    elif n_samples < 10000:
        base_min = 20
    elif n_samples < 100000:
        base_min = 50
    else:
        base_min = 100
    
    # Algorithm-specific adjustments
    if clustering_algorithm in ['GMM', 'SGMM', 'kmeans', 'kmedians']:
        # These algorithms benefit from more clusters for complex data
        return min(150, max(base_min, n_samples // 1000))
    elif clustering_algorithm in ['DBSCAN', 'MShift']:
        # Density-based algorithms need fewer clusters
        return max(base_min // 2, 5)
    else:
        return base_min

def calculate_data_complexity(X):
    """Calculate data complexity score (0-1, higher = more complex)"""
    if len(X) == 0:
        return 0.5
    
    try:
        # Method 1: Feature correlation diversity
        if X.shape[1] > 1:
            # Calculate correlation matrix and its variance
            corr_matrix = np.corrcoef(X.T)
            # Remove diagonal and get unique correlations
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            correlations = corr_matrix[mask]
            corr_variance = np.var(correlations) if len(correlations) > 0 else 0
        else:
            corr_variance = 0
        
        # Method 2: Ratio of features to samples
        feature_sample_ratio = min(X.shape[1] / X.shape[0], 1.0)
        
        # Method 3: Variance distribution across features
        feature_variances = np.var(X, axis=0)
        var_std = np.std(feature_variances) if len(feature_variances) > 1 else 0
        normalized_var_std = min(var_std / (np.mean(feature_variances) + 1e-8), 2.0) / 2.0
        
        # Combine metrics (0-1 scale)
        complexity = (
            min(corr_variance * 2, 1.0) * 0.4 +
            feature_sample_ratio * 0.3 +
            normalized_var_std * 0.3
        )
        
        return min(max(complexity, 0.0), 1.0)
        
    except Exception as e:
        print(f"Warning: Error calculating data complexity: {e}. Using default 0.5.")
        return 0.5

def get_sensitivity_factor(data_complexity, clustering_algorithm):
    """Get sensitivity factor based on data complexity and algorithm"""
    # Base sensitivity: complex data = less sensitive (allow more clusters)
    if data_complexity > 0.7:
        base_sensitivity = 0.3  # Very low sensitivity for complex data
    elif data_complexity > 0.5:
        base_sensitivity = 0.6  # Medium sensitivity
    elif data_complexity > 0.3:
        base_sensitivity = 1.0  # Normal sensitivity
    else:
        base_sensitivity = 1.5  # High sensitivity for simple data
    
    # Algorithm-specific adjustments
    if clustering_algorithm in ['GMM', 'SGMM']:
        # These handle complexity better, can be more sensitive
        return base_sensitivity * 1.2
    elif clustering_algorithm in ['DBSCAN', 'MShift']:
        # These prefer fewer clusters, increase sensitivity
        return base_sensitivity * 1.5
    else:
        return base_sensitivity

def find_significant_elbow(second_diff, sensitivity_factor):
    """Find elbow point using relative change threshold"""
    if len(second_diff) == 0:
        return 0
    
    # Calculate dynamic threshold
    mean_change = np.mean(second_diff)
    std_change = np.std(second_diff)
    
    # Threshold = mean + (sensitivity_factor * std)
    # Lower sensitivity = higher threshold = fewer "significant" changes
    threshold = mean_change + (sensitivity_factor * std_change)
    
    # Find points above threshold
    significant_indices = np.where(second_diff > threshold)[0]
    
    if len(significant_indices) > 0:
        # Return first significant change (earlier elbow preferred)
        return significant_indices[0]
    else:
        # Fallback: return maximum if no significant changes
        return np.argmax(second_diff)

def get_validation_candidates(optimal_k, max_clusters, validation_range=20):
    """Get additional k values to validate the optimal_k choice"""
    # Test a range after the optimal_k to see if Jaccard improves
    start_k = optimal_k + 1
    end_k = min(optimal_k + validation_range, max_clusters)
    
    # Generate candidates: every 2-3 values to reduce computation
    candidates = []
    step = max(1, validation_range // 10)  # Adaptive step size
    
    for k in range(start_k, end_k + 1, step):
        candidates.append(k)
    
    # Always include a few immediate neighbors and the end point
    immediate_neighbors = [optimal_k + 1, optimal_k + 2]
    if end_k not in candidates:
        candidates.append(end_k)
    
    # Combine and sort
    all_candidates = list(set(candidates + immediate_neighbors))
    all_candidates = [k for k in all_candidates if k <= max_clusters and k > optimal_k]
    all_candidates.sort()
    
    print(f"[Elbow_method] Generated {len(all_candidates)} validation candidates: {all_candidates[:5]}..." if len(all_candidates) > 5 else f"[Elbow_method] Generated validation candidates: {all_candidates}")
    
    return all_candidates

# Helper function for Elbow_method parallel execution
def _calculate_score_for_k(args_tuple):
    """Calculates WCSS/BIC score for a single k value."""
    # Unpack num_processes_for_algo from args_tuple
    k, data_local, X_local, clustering_algorithm_local, current_parameter_dict_local, num_processes_for_algo_local = args_tuple
    score_val = np.inf # Default to a bad score

    try:
        # print(f"[DEBUG Elbow Worker] Processing k={k} for {clustering_algorithm_local} with num_processes_for_algo={num_processes_for_algo_local}") # Optional worker debug
        # Pass num_processes_for_algo_local to Elbow_choose_clustering_algorithm
        clustering_result = Elbow_choose_clustering_algorithm(data_local, X_local, clustering_algorithm_local, k, current_parameter_dict_local, num_processes_for_algo=num_processes_for_algo_local)
        
        if clustering_result is None or 'before_labeling' not in clustering_result or clustering_result['before_labeling'] is None:
            print(f"Warning: Clustering failed or returned no model for k={k} with {clustering_algorithm_local}. Using inf score.")
            return k, np.inf # score_val is already np.inf or explicitly set
            
        clustering_model = clustering_result['before_labeling']
        
        # The original code had a try-except for fit here, but pre_clustering usually handles fit/fit_predict.
        # We rely on inertia_ or bic() being available after Elbow_choose_clustering_algorithm.

        temp_score = None
        if clustering_algorithm_local in ['GMM', 'SGMM']:
            if hasattr(clustering_model, 'bic') and callable(getattr(clustering_model, 'bic')):
                try:
                    temp_score = clustering_model.bic(X_local)
                except Exception as e_bic:
                    print(f"Warning: Error calling bic() for {clustering_algorithm_local} model at k={k}: {e_bic}")
            else:
                print(f"Warning: bic() method not available or not callable for {clustering_algorithm_local} model at k={k}.")
        elif clustering_algorithm_local == 'CK':
            if hasattr(clustering_model, 'silhouette_score_'):
                silhouette_val = clustering_model.silhouette_score_
                if silhouette_val is not None and np.isfinite(silhouette_val):
                    # Silhouette Score is in [-1, 1], higher is better.
                    # To make it compatible with Elbow (lower is better & positive slope for elbow point):
                    # Transform to [0, 2] where 0 is best.
                    temp_score = 1.0 - silhouette_val 
                else:
                    print(f"Warning: Silhouette score is None or invalid for CK model at k={k} (value: {silhouette_val}). Using inf score for this k.")
            else:
                print(f"Warning: silhouette_score_ attribute not available for CK model at k={k}. Using inf score for this k.")
        else: # K-Means, etc.
            if hasattr(clustering_model, 'inertia_'):
                inertia_val = clustering_model.inertia_
                if inertia_val is not None and np.isfinite(inertia_val):
                    temp_score = inertia_val
                else:
                    print(f"Warning: Inertia is None or invalid for {clustering_algorithm_local} model at k={k} (value: {inertia_val}). Using inf score for this k.")
            else:
                print(f"Warning: inertia_ attribute not available for {clustering_algorithm_local} model at k={k}. Using inf score for this k.")
        
        if temp_score is not None and np.isfinite(temp_score):
            score_val = temp_score
        else:
            score_val = np.inf # Ensure bad score if temp_score ended up None or non-finite
            
    except np.linalg.LinAlgError as e:
        print(f"LinAlgError for k={k} with {clustering_algorithm_local}: {e}. Using inf score.")
    except ValueError as e:
        print(f"ValueError for k={k} with {clustering_algorithm_local}: {e}. Using inf score.")
    except Exception as e:
        # Catch any other unexpected errors from a specific k iteration
        print(f"Unexpected error for k={k} with {clustering_algorithm_local}: {e}. Using inf score.")
        
    # print(f"[DEBUG Elbow Worker] Finished k={k}, score={score_val}") # Optional
    return k, score_val