# input 'X' is X_reduced or X rows
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional) overall dictionary
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

import numpy as np
from sklearn.cluster import KMeans
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Loop_elbow_gs import loop_tuning
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify
import joblib # Added for parallel_backend


def clustering_Kmeans(data, X, max_clusters, aligned_original_labels, global_known_normal_samples_pca=None, threshold_value=0.3, num_processes_for_algo=1, k=None): # main clustering, ADDED k=None
    
    best_parameter_dict = {'random_state': 42} # Default parameters
    
    if k is not None:
        # If k is provided, bypass hyperparameter tuning
        print(f"[INFO] K-Means using pre-defined k={k}. Bypassing auto-tuning.")
        n_clusters = k
    else:
        # Pass num_processes_for_algo to loop_tuning (loop_tuning itself will need to be modified to accept and use this)
        print("[INFO] K-Means finding optimal k using auto-tuning (Elbow/GridSearch)...")
        clustering_result_dict = loop_tuning(data, X, 'Kmeans', max_clusters, num_processes_for_algo=num_processes_for_algo)
        n_clusters = clustering_result_dict['optimal_cluster_n']
        best_parameter_dict = clustering_result_dict['best_parameter_dict']

    # Use num_processes_for_algo for n_jobs in KMeans
    # Ensure n_init in best_parameter_dict is > 1 for n_jobs to be effective, or set a default n_init like 10 if not present.
    n_init_val = best_parameter_dict.get('n_init', 10) # Default to 10 if not in dict
    kmeans = KMeans(n_clusters=n_clusters, random_state=best_parameter_dict['random_state'], n_init=n_init_val)
    
    clusters = None
    # Determine n_jobs for joblib.parallel_backend context
    n_jobs_for_context = 1 # Default to 1 (sequential)
    if num_processes_for_algo == 0: # 0 means use all available cores
        n_jobs_for_context = -1
    elif num_processes_for_algo is not None and num_processes_for_algo > 0:
        n_jobs_for_context = num_processes_for_algo
        
    # Use joblib.parallel_backend to control parallelism for KMeans n_init runs
    with joblib.parallel_backend('loky', n_jobs=n_jobs_for_context):
        clusters = kmeans.fit_predict(X)

    # Debug cluster id (data refers to original data, X is the data used for clustering)
    print(f"\n[DEBUG KMeans main_clustering] Param for CNI 'data_features_for_clustering' (X) - Shape: {X.shape}")
    # aligned_original_labels shape will be printed inside CNI
    # print(f"[DEBUG KMeans main_clustering] Param for CNI 'aligned_original_labels' - Shape: {aligned_original_labels.shape}")
    
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

    # predict_Kmeans = data['cluster'] # Old way

    return {
        'Cluster_labeling': final_cluster_labels_from_cni, # Use labels from CNI
        'Best_parameter_dict': best_parameter_dict,
        'raw_cluster_labels': clusters, # <-- ADDED: Return raw cluster IDs
        'num_clusters': n_clusters      # <-- ADDED: Return the number of clusters
    }


def pre_clustering_Kmeans(data, X, n_clusters, random_state, n_init, num_processes_for_algo=1): # Added num_processes_for_algo
    # Apply KMeans Clustering
    # Use num_processes_for_algo for n_jobs in KMeans. Ensure n_init > 1.
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    
    cluster_labels = None
    # Determine n_jobs for joblib.parallel_backend context
    n_jobs_for_context = 1 # Default to 1 (sequential)
    if num_processes_for_algo == 0: # 0 means use all available cores
        n_jobs_for_context = -1
    elif num_processes_for_algo is not None and num_processes_for_algo > 0:
        n_jobs_for_context = num_processes_for_algo
        
    # Use joblib.parallel_backend to control parallelism for KMeans n_init runs
    with joblib.parallel_backend('loky', n_jobs=n_jobs_for_context):
        cluster_labels = kmeans.fit_predict(X)

    # REMOVED call to clustering_nomal_identify
    # final_cluster_labels_from_cni = clustering_nomal_identify(X, aligned_original_labels, cluster_labels, n_clusters)
    # num_clusters_after_cni = len(np.unique(final_cluster_labels_from_cni))

    return {
        # 'Cluster_labeling' : final_cluster_labels_from_cni, # REMOVED CNI result
        'model_labels' : cluster_labels, # Model-generated labels (before CNI)
        'n_clusters' : n_clusters, # Number of clusters requested (can be different from unique labels in final_cluster_labels_from_cni)
        'before_labeling' : kmeans # Model object
    }