# input 'X' is X_reduced or X rows
# Clustering Method: MeanShift
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional)
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import MeanShift, estimate_bandwidth
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Grid_search import Grid_search_all
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify
# from Clustering_Method.clustering_algorithm_chunked import meanshift_with_chunking  # COMMENTED OUT: MeanShift chunking not yet enabled
import logging

logger = logging.getLogger(__name__)


def clustering_MShift_clustering(data, X, state, quantile, n_samples, num_processes_for_algo=1, use_chunking=False, chunk_size=30000):  # Fundamental MeanShift clustering
    """
    Apply MeanShift clustering with optional chunking for large datasets.
    
    Parameters:
    -----------
    use_chunking : bool, default=False
        If True and dataset is large, use chunked MeanShift to avoid OOM errors.
        Currently DISABLED (False by default) - enable if MeanShift causes memory issues.
    chunk_size : int, default=30000
        Size of each chunk when chunking is enabled
    """
    # Estimate bandwidth based on the data
    bandwidth = estimate_bandwidth(X, quantile=quantile, n_samples=n_samples, random_state=state) # default; randomm_state=42, n_samples=500, quantile=0.2
    if bandwidth <= 0:
        bandwidth = 0.1  # Minimum safe value
    
    n_samples_total = len(X)
    
    # CHUNKING LOGIC (Currently commented out - MeanShift uses standard approach)
    '''
    if use_chunking and n_samples_total > chunk_size:
        logger.info(f"[MeanShift] Dataset size {n_samples_total} > {chunk_size}, using chunked MeanShift")
        clusters = meanshift_with_chunking(X, bandwidth=bandwidth, chunk_size=chunk_size, overlap_ratio=0.1)
        num_clusters = len(np.unique(clusters))
        MShift_model = None  # No single model object when using chunking
    else:
    '''
    # Standard MeanShift (always used for now)
    if n_samples_total > chunk_size:
        logger.info(f"[MeanShift] Dataset size {n_samples_total} > {chunk_size}, but chunking is DISABLED")
        logger.info(f"[MeanShift] Using standard MeanShift - may cause memory issues on very large datasets")
    
    # Use num_processes_for_algo for n_jobs in MeanShift
    MShift_model = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=num_processes_for_algo)
    clusters = MShift_model.fit_predict(X)

    num_clusters = len(np.unique(clusters))  # Counting the number of clusters
    
    return clusters, num_clusters, MShift_model


def clustering_MShift(data, X, aligned_original_labels, global_known_normal_samples_pca=None, threshold_value=0.3, num_processes_for_algo=1, quantile=None):
    if quantile is not None:
        print(f"[INFO] Mean Shift using pre-defined quantile={quantile}. Bypassing auto-tuning.")
        parameter_dict = {'quantile': quantile, 'n_samples': 100, 'random_state': 42}
    else:
        parameter_dict = Grid_search_all(X, 'MShift', num_processes_for_algo=num_processes_for_algo)

    random_state_val = parameter_dict.get('random_state', 42)
    quantile_val = parameter_dict.get('quantile', 0.15) 
    n_samples_val = parameter_dict.get('n_samples', 100)

    clusters, num_clusters, MShift_model_instance = clustering_MShift_clustering(data, X, state=random_state_val, quantile=quantile_val, n_samples=n_samples_val, num_processes_for_algo=num_processes_for_algo)

    print(f"\n[DEBUG MeanShift main_clustering] Param for CNI 'data_features_for_clustering' (X) - Shape: {X.shape}")

    # Pass X (features used for clustering) and aligned_original_labels to CNI
    final_cluster_labels_from_cni, _, _ = clustering_nomal_identify(
        data_features_for_clustering=X,
        original_labels_aligned=aligned_original_labels,
        clusters_assigned=clusters,
        global_known_normal_samples_pca=global_known_normal_samples_pca,
        threshold_value=threshold_value,
        num_processes_for_algo=num_processes_for_algo,
        data_for_clustering=X # Pass the data for clustering
    )

    return {
        'Cluster_labeling': final_cluster_labels_from_cni, # Use labels from CNI
        'Best_parameter_dict': parameter_dict,
        'raw_cluster_labels': clusters,
        'num_clusters': num_clusters
    }


# Additional classes for Grid Search
class MeanShiftWithDynamicBandwidth(BaseEstimator, ClusterMixin):
    def __init__(self, quantile=0.3, n_samples=500, bin_seeding=True, num_processes_for_algo=1):
        self.quantile = quantile
        self.n_samples = n_samples
        self.bin_seeding = bin_seeding
        self.num_processes_for_algo = num_processes_for_algo
        self.bandwidth = None
        self.model = None

    def fit(self, X, y=None):
        # Dynamically set bandwidth based on data
        self.bandwidth = estimate_bandwidth(X, quantile=self.quantile, n_samples=self.n_samples)

        # Set a stable minimum
        if self.bandwidth < 1e-3:
            print(f"Estimated bandwidth too small ({self.bandwidth:.5f}) → Adjusted to 0.001")
            self.bandwidth = 1e-3

        self.model = MeanShift(bandwidth=self.bandwidth, bin_seeding=self.bin_seeding, n_jobs=self.num_processes_for_algo)
        self.model.fit(X)

        self.labels_ = self.model.labels_
        
        return self

    def predict(self, X):
        return self.model.predict(X)
    

def pre_clustering_MShift(data, X, random_state, quantile, n_samples, num_processes_for_algo=1):
    cluster_labels, num_clusters_actual, MShift_model_obj = clustering_MShift_clustering(data, X, random_state, quantile, n_samples, num_processes_for_algo=num_processes_for_algo)

    return {
        'model_labels' : cluster_labels,
        'n_clusters' : num_clusters_actual, # Actual n_clusters from MeanShift
        'before_labeling' : MShift_model_obj
    }