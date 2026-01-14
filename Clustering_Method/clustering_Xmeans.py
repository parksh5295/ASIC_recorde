# input 'X' is X_reduced or X rows
# Clustering Algorithm: X-means; Autonomously tuning n_clusters in k-means
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional)
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import silhouette_score
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Grid_search import Grid_search_all
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify
import numpy as np
import joblib

# X-Means Clustering Function
def x_means_clustering(X, random_state, max_clusters, n_init=30, num_processes_for_algo=1):
    best_score = -1
    best_model = None
    best_k = 2
    for k in range(2, max_clusters + 1):
        #model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init, n_jobs=num_processes_for_algo)
        # CORRECTED: Removed the deprecated 'n_jobs' parameter. Parallelism is handled by joblib.
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        
        n_jobs_for_context = 1
        if num_processes_for_algo == 0:
            n_jobs_for_context = -1
        elif num_processes_for_algo is not None and num_processes_for_algo > 0:
            n_jobs_for_context = num_processes_for_algo

        labels = None
        with joblib.parallel_backend('loky', n_jobs=n_jobs_for_context):
             labels = model.fit_predict(X)

        if len(np.unique(labels)) < 2:
            continue
        
        silhouette_avg = silhouette_score(X, labels)
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_model = model
            best_k = k
    return best_model, best_k

def clustering_Xmeans_clustering(data, X, random_state, max_clusters, n_init=30, num_processes_for_algo=1):
    model, optimal_k = x_means_clustering(X, random_state, max_clusters, n_init=n_init, num_processes_for_algo=num_processes_for_algo)
    if model is None:
        print("[Warning clustering_Xmeans_clustering] x_means_clustering returned no model. Defaulting to k=2 or empty labels.")
        return np.array([]), 2

    clusters = model.labels_

    return clusters, optimal_k


def clustering_Xmeans(data, X, aligned_original_labels, global_known_normal_samples_pca=None, threshold_value=0.3, num_processes_for_algo=1, max_clusters=None):
    if max_clusters is not None:
        # This branch is called from Jaccard_Elbow_Method
        print(f"[INFO] X-Means using pre-defined max_clusters={max_clusters}. Bypassing auto-tuning.")
        parameter_dict = {'max_clusters': max_clusters, 'random_state': 42, 'n_init': 30}
        random_state_val = parameter_dict.get('random_state', 42)
        max_clusters_val = parameter_dict.get('max_clusters', 1000)
        n_init_val = parameter_dict.get('n_init', 30)

    else:
        # This branch is for standalone calls
        parameter_dict = Grid_search_all(X, 'Xmeans', num_processes_for_algo=num_processes_for_algo)

        if parameter_dict is None or parameter_dict.get('silhouette_score_from_grid', -1) == -1:
            print("Warning: Grid search for XMeans might have failed or returned default/no-op parameters.")
            if parameter_dict is None: parameter_dict = {}
            random_state_val = parameter_dict.get('random_state', 42)
            max_clusters_val = parameter_dict.get('max_clusters', 1000)
            n_init_val = parameter_dict.get('n_init', 30)
        else:
            random_state_val = parameter_dict.get('random_state')
            max_clusters_val = parameter_dict.get('max_clusters')
            n_init_val = parameter_dict.get('n_init', 30)

    clusters, num_clusters = clustering_Xmeans_clustering(data, X, 
                                                        random_state=random_state_val, 
                                                        max_clusters=max_clusters_val,
                                                        n_init=n_init_val,
                                                        num_processes_for_algo=num_processes_for_algo)

    # Debug cluster id (X is the data used for clustering)
    print(f"\n[DEBUG XMeans main_clustering] Param for CNI 'data_features_for_clustering' (X) - Shape: {X.shape}")
    # aligned_original_labels shape will be printed inside CNI
    # print(f"[DEBUG XMeans main_clustering] Param for CNI 'aligned_original_labels' - Shape: {aligned_original_labels.shape}")

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

    # final_cluster_labels_from_cni = clustering_nomal_identify(X, aligned_original_labels, clusters, num_clusters, global_known_normal_samples_pca=global_known_normal_samples_pca, threshold_value=threshold_value, num_processes_for_algo=num_processes_for_algo)

    # predict_Xmeans = data['cluster'] # Old way

    return {
        'Cluster_labeling': final_cluster_labels_from_cni,
        'Best_parameter_dict': parameter_dict,
        'raw_cluster_labels': clusters,
        'num_clusters': num_clusters
    }


# Auxiliary class for Grid Search
class XMeansWrapper(BaseEstimator, ClusterMixin):
    def __init__(self, random_state=42, max_clusters=10, n_init=30, num_processes_for_algo=1):
        self.random_state = random_state
        self.max_clusters = max_clusters
        self.n_init = n_init
        self.num_processes_for_algo = num_processes_for_algo
        self.model = None
        self.best_k = None

    def fit(self, X, y=None):
        # This now correctly calls the updated x_means_clustering which handles parallelism properly
        self.model, self.best_k = x_means_clustering(X, self.random_state, self.max_clusters, n_init=self.n_init, num_processes_for_algo=self.num_processes_for_algo)
        return self

    def predict(self, X):
        return self.model.predict(X) if self.model else None

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)
    

def pre_clustering_Xmeans(data, X, random_state, max_clusters, n_init=30, num_processes_for_algo=1):
    cluster_labels, num_clusters_optimal = clustering_Xmeans_clustering(data, X, random_state, max_clusters, n_init=n_init, num_processes_for_algo=num_processes_for_algo)

    return {
        'model_labels' : cluster_labels,
        'n_clusters' : num_clusters_optimal,
        'before_labeling' : cluster_labels
    }