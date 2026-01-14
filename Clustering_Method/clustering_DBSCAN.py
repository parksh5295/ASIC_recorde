# input 'X' is X_reduced or X rows
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional)
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

import numpy as np
from sklearn.cluster import DBSCAN
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Grid_search import Grid_search_all
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify
from Clustering_Method.clustering_algorithm_chunked import dbscan_with_chunking
import logging

logger = logging.getLogger(__name__)


def clustering_DBSCAN_clustering(data, X, eps, count_samples, num_processes_for_algo=1, use_chunking=True, chunk_size=30000):  # Added num_processes_for_algo
    """
    Apply DBSCAN clustering with optional chunking for large datasets.
    
    Parameters:
    -----------
    use_chunking : bool, default=True
        If True and dataset is large, use chunked DBSCAN to avoid OOM errors
    chunk_size : int, default=30000
        Size of each chunk when chunking is enabled
    """
    n_samples = len(X)
    
    # Use chunking for large datasets to avoid memory issues
    if use_chunking and n_samples > chunk_size:
        logger.info(f"[DBSCAN] Dataset size {n_samples} > {chunk_size}, using chunked DBSCAN")
        clusters = dbscan_with_chunking(X, eps=eps, min_samples=count_samples, chunk_size=chunk_size, overlap_ratio=0.1)
        # clusters = dbscan_with_chunking(X, eps=eps, min_samples=count_samples, chunk_size=chunk_size, overlap_ratio=0.1, n_jobs=num_processes_for_algo)
        num_clusters = len(np.unique(clusters))
        dbscan = None  # No single model object when using chunking
    else:
        # Standard DBSCAN for smaller datasets
        logger.info(f"[DBSCAN] Dataset size {n_samples} <= {chunk_size}, using standard DBSCAN")
        dbscan = DBSCAN(eps=eps, min_samples=count_samples, n_jobs=num_processes_for_algo)
        clusters = dbscan.fit_predict(X)
        num_clusters = len(np.unique(clusters))
    
    return clusters, num_clusters, dbscan


def clustering_DBSCAN(data, X_reduced_features, original_labels_aligned, global_known_normal_samples_pca=None, threshold_value=0.3, num_processes_for_algo=1, eps=None): # MODIFIED: Added eps=None
    
    # --- MODIFIED: Use provided eps or run auto-tuning ---
    if eps is not None:
        # If eps is provided, skip GridSearch and use the given value
        print(f"[INFO DBSCAN] Using pre-defined 'eps' value: {eps}")
        parameter_dict = {'eps': eps, 'min_samples': 5} # Use a default for min_samples
    else:
        # If eps is not provided, run the original auto-tuning process
        print("[INFO DBSCAN] 'eps' not provided, starting auto-tuning via GridSearch...")
        # Pass num_processes_for_algo to Grid_search_all (Grid_search_all itself will need to be modified)
        parameter_dict = Grid_search_all(X_reduced_features, 'DBSCAN', num_processes_for_algo=num_processes_for_algo) 
    
    # Using internal defaults due to not passing parameter_dict when calling Grid_search_all
    # Default parameter_dict inside Grid_search_all in Grid_search.py: 'eps': 0.5, 'count_samples': 5
    eps_val = parameter_dict.get('eps', 0.5) 
    # Grid_search_all is optimized for 'min_samples', so check that key first.
    # 'count_samples' is a key of the default value inside Grid_search_all.
    min_samples_val = parameter_dict.get('min_samples', parameter_dict.get('count_samples', 5))

    print(f"DBSCAN: Using parameters eps={eps_val}, min_samples={min_samples_val}")
    
    # Pass num_processes_for_algo to clustering_DBSCAN_clustering
    predict_DBSCAN, num_clusters_actual, dbscan_model = clustering_DBSCAN_clustering(data, X_reduced_features, eps_val, min_samples_val, num_processes_for_algo=num_processes_for_algo)
    
    # Pass num_processes_for_algo to clustering_nomal_identify
    final_cluster_labels_from_cni, _, _ = clustering_nomal_identify(
        data_features_for_clustering=X_reduced_features,
        original_labels_aligned=original_labels_aligned,
        clusters_assigned=predict_DBSCAN,
        global_known_normal_samples_pca=global_known_normal_samples_pca,
        threshold_value=threshold_value,
        num_processes_for_algo=num_processes_for_algo,
        data_for_clustering=X_reduced_features # Pass the data for clustering
    )

    num_clusters_after_cni = len(np.unique(final_cluster_labels_from_cni))

    return {
        'Cluster_labeling': final_cluster_labels_from_cni,
        'Best_parameter_dict': parameter_dict,
        'raw_cluster_labels': predict_DBSCAN,
        'num_clusters': num_clusters_actual
    }


def pre_clustering_DBSCAN(data, X, eps, count_samples, num_processes_for_algo=1): # Added num_processes_for_algo
    # Pass num_processes_for_algo to clustering_DBSCAN_clustering
    cluster_labels, num_clusters_actual, dbscan = clustering_DBSCAN_clustering(data, X, eps, count_samples, num_processes_for_algo=num_processes_for_algo)
    
    return {
        'model_labels' : cluster_labels,
        'n_clusters' : num_clusters_actual,
        'before_labeling' : dbscan
    }
