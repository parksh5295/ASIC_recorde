# Clustering Method = Spherical GMM
# input 'X' is X_reduced or X rows
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional)
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

import numpy as np
from sklearn.mixture import GaussianMixture
from utils.progressing_bar import progress_bar
from Tuning_hyperparameter.Elbow_method import Elbow_method
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify
from Clustering_Method.clustering_GMM import fit_gmm_with_retry


def clustering_SGMM(data, X, max_clusters, aligned_original_labels, global_known_normal_samples_pca=None, threshold_value=0.3, num_processes_for_algo=1, k=None):
    parameter_dict = {'random_state': 42} # Default parameters
    
    if k is not None:
        print(f"[INFO] SGMM using pre-defined k={k}. Bypassing auto-tuning.")
        n_clusters = k
    else:
        print("[INFO] SGMM finding optimal k using auto-tuning (Elbow Method)...")
        sgmm_specific_max_clusters = min(max_clusters, 50) # GMM/SGMM is tested up to 50
        
        # Call Elbow_method to get optimal_cluster_n and best_parameter_dict
        # Pass num_processes_for_algo to Elbow_method for its internal parallelization of k-value checks
        # The parameter_dict from Elbow_method will contain 'n_init', 'random_state', 'reg_covar' etc.
        after_elbow = Elbow_method(data, X, 'SGMM', sgmm_specific_max_clusters, num_processes_for_algo=num_processes_for_algo)
        n_clusters = after_elbow['optimal_cluster_n']
        parameter_dict = after_elbow['best_parameter_dict']

    # For final model fitting, use robust parameters from parameter_dict (especially n_init)
    # And do not override max_iter (or set to None) to use GaussianMixture's default
    final_n_init_val = parameter_dict.get('n_init', 30) # Default to 30 if not in dict for some reason
    final_reg_covar_init_val = parameter_dict.get('reg_covar', 1e-6)
    final_random_state = parameter_dict.get('random_state', 42) # Default random_state

    # print(f"[DEBUG clustering_SGMM] Final k={n_clusters}, using n_init={final_n_init_val}, reg_covar={final_reg_covar_init_val}")

    sgmm, clusters = fit_gmm_with_retry(
        X, 
        n_components=n_clusters, 
        covariance_type='spherical', 
        random_state=final_random_state,
        n_init_val=final_n_init_val, # Use full n_init for final model
        reg_covar_init=final_reg_covar_init_val,
        max_iter_override=None, # Use default max_iter for final model
        num_processes_for_algo=num_processes_for_algo # Though not used in fit_gmm_with_retry
    )
    
    # Debug cluster id (X is the data used for clustering)
    print(f"\n[DEBUG SGMM main_clustering] Param for CNI 'data_features_for_clustering' (X) - Shape: {X.shape}")
    # aligned_original_labels shape will be printed inside CNI
    # print(f"[DEBUG SGMM main_clustering] Param for CNI 'aligned_original_labels' - Shape: {original_labels_aligned.shape}")
    
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

    # Update parameter_dict with specific SGMM parameters if needed, though most are from Elbow already
    # parameter_dict['optimal_cluster_n_sgmm'] = n_clusters # Example

    return {
        'Cluster_labeling': final_cluster_labels_from_cni,
        'Best_parameter_dict': parameter_dict,
        'raw_cluster_labels': clusters,
        'num_clusters': n_clusters
    }


# Precept Function for Clustering Count Tuning Loop (used by Elbow_method)
def pre_clustering_SGMM(data, X, n_clusters, random_state, reg_covar=1e-6, n_init=1, num_processes_for_algo=1):
    # For Elbow method, use smaller n_init and max_iter to speed up k selection.
    # The main clustering_SGMM function will do a more thorough fit later.
    quick_n_init = 5  # Reduced n_init for speed during Elbow
    quick_max_iter = 50 # Reduced max_iter for speed during Elbow

    # print(f"[DEBUG pre_clustering_SGMM] k={n_clusters}, using quick_n_init={quick_n_init}, quick_max_iter={quick_max_iter}, reg_covar={reg_covar}")

    # Original n_init received (e.g. 30 from Elbow_method) is now overridden by quick_n_init for this pre-clustering step.
    # The reg_covar is passed as is from Elbow_method's parameter_dict.
    sgmm, cluster_labels = fit_gmm_with_retry(
        X, 
        n_components=n_clusters, 
        covariance_type='spherical', 
        random_state=random_state, 
        reg_covar_init=reg_covar, # Use reg_covar passed from Elbow
        n_init_val=quick_n_init,    # Use smaller n_init for pre-clustering
        max_iter_override=quick_max_iter, # Use smaller max_iter for pre-clustering
        num_processes_for_algo=num_processes_for_algo # Though not used in fit_gmm_with_retry
    )

    # predict_SGMM = clustering_nomal_identify(data, cluster_labels, n_clusters)
    # num_clusters = len(np.unique(predict_SGMM))  # Counting the number of clusters

    return {
        'model_labels' : cluster_labels,
        'n_clusters' : n_clusters, # n_clusters requested
        'before_labeling' : sgmm
    }