# Modules for determining how to cluster
# Output: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}; Name: Clustering

import numpy as np
from Clustering_Method.clustering_Kmeans import clustering_Kmeans
from Clustering_Method.clustering_Kmedians import clustering_Kmedians
from Clustering_Method.clustering_GMM import clustering_GMM
from Clustering_Method.clustering_SGMM import clustering_SGMM
from Clustering_Method.clustering_Gmeans import clustering_Gmeans
from Clustering_Method.clustering_Xmeans import clustering_Xmeans
from Clustering_Method.clustering_DBSCAN import clustering_DBSCAN
from Clustering_Method.clustering_MShift import clustering_MShift
from Clustering_Method.clustering_FCM import clustering_FCM
from Clustering_Method.clustering_CK import clustering_CK
from Clustering_Method.clustering_NeuralGas import clustering_NeuralGas
from Clustering_Method.clustering_CLARA import clustering_CLARA


def choose_clustering_algorithm(data, X_reduced_features, original_labels_aligned, clustering_algorithm_choice, max_clusters=1000, global_known_normal_samples_pca=None, threshold_value=0.3, num_processes_for_algo=1):
    '''
    parameter_dict = {'random_state' : random_state, 'n_init' : n_init, 'max_clusters' : max_clusters, 'tol' : tol, 'eps' : eps,
                        'count_samples' : count_samples, 'quantile' : quantile, 'n_samples' : n_samples, 'n_start_nodes' : n_start_nodes,
                        'max_nodes' : max_nodes, 'step' : step, 'max_edge_age' : max_edge_age, 'epochs' : epochs,
                        'batch_size': batch_size, 'n_neighbors' : n_neighbors
    }
    '''
    GMM_type = None
    clustering = None # Initialize clustering variable

    # Pass original_labels_aligned to each clustering function call.
    # The 'data' argument is kept as it might be used by Elbow_method or other hyperparameter tuning logic.
    # X_reduced_features is the actual input for clustering.

    if clustering_algorithm_choice in ['Kmeans', 'kmeans']:
        print(f"\n[DEBUG AutoTune {clustering_algorithm_choice}] Calling clustering_Kmeans with:")
        print(f"[DEBUG AutoTune {clustering_algorithm_choice}] - global_known_normal_samples_pca type: {type(global_known_normal_samples_pca)}")
        print(f"[DEBUG AutoTune {clustering_algorithm_choice}] - global_known_normal_samples_pca is None: {global_known_normal_samples_pca is None}")
        if global_known_normal_samples_pca is not None:
            print(f"[DEBUG AutoTune {clustering_algorithm_choice}] - global_known_normal_samples_pca shape: {global_known_normal_samples_pca.shape}")
            print(f"[DEBUG AutoTune {clustering_algorithm_choice}] - global_known_normal_samples_pca contains NaN: {np.any(np.isnan(global_known_normal_samples_pca)) if isinstance(global_known_normal_samples_pca, np.ndarray) else 'N/A'}")
        print(f"[DEBUG AutoTune {clustering_algorithm_choice}] - threshold_value: {threshold_value}")
        print(f"[DEBUG AutoTune {clustering_algorithm_choice}] - X_reduced_features shape: {X_reduced_features.shape}")
        print(f"[DEBUG AutoTune {clustering_algorithm_choice}] - original_labels_aligned shape: {original_labels_aligned.shape}")
        
        clustering = clustering_Kmeans(data, X_reduced_features, max_clusters, original_labels_aligned, global_known_normal_samples_pca=global_known_normal_samples_pca, threshold_value=threshold_value, num_processes_for_algo=num_processes_for_algo)

    elif clustering_algorithm_choice in ['Kmedians', 'kmedians']:
        clustering = clustering_Kmedians(data, X_reduced_features, max_clusters, original_labels_aligned, global_known_normal_samples_pca=global_known_normal_samples_pca, threshold_value=threshold_value, num_processes_for_algo=num_processes_for_algo)

    elif clustering_algorithm_choice.upper().startswith('GMM'):
        print(f"\n[DEBUG AutoTune {clustering_algorithm_choice}] Calling clustering_GMM with:")
        print(f"[DEBUG AutoTune {clustering_algorithm_choice}] - global_known_normal_samples_pca type: {type(global_known_normal_samples_pca)}")
        print(f"[DEBUG AutoTune {clustering_algorithm_choice}] - global_known_normal_samples_pca is None: {global_known_normal_samples_pca is None}")
        if global_known_normal_samples_pca is not None:
            print(f"[DEBUG AutoTune {clustering_algorithm_choice}] - global_known_normal_samples_pca shape: {global_known_normal_samples_pca.shape}")
            print(f"[DEBUG AutoTune {clustering_algorithm_choice}] - global_known_normal_samples_pca contains NaN: {np.any(np.isnan(global_known_normal_samples_pca)) if isinstance(global_known_normal_samples_pca, np.ndarray) else 'N/A'}")
        print(f"[DEBUG AutoTune {clustering_algorithm_choice}] - threshold_value: {threshold_value}")
        print(f"[DEBUG AutoTune {clustering_algorithm_choice}] - X_reduced_features shape: {X_reduced_features.shape}")
        print(f"[DEBUG AutoTune {clustering_algorithm_choice}] - original_labels_aligned shape: {original_labels_aligned.shape}")
        
        parts = clustering_algorithm_choice.split('_')
        if len(parts) == 1 and parts[0].upper() == 'GMM': # Only "GMM"
            # GMM_type = input("Please enter the GMM type, i.e. normal, full, tied, diag: ") # Commented out
            GMM_type = "normal"  # Default to "normal"
            print(f"[INFO] GMM algorithm selected (Autotune path). Defaulting to GMM type: {GMM_type}")
        elif len(parts) == 2 and parts[0].upper() == 'GMM' and parts[1].lower() in ['normal', 'full', 'tied', 'diag']:
            GMM_type = parts[1].lower()
            print(f"[INFO] Using GMM type '{GMM_type}' from algorithm choice: {clustering_algorithm_choice} (Autotune path)")
        else:
            print(f"Unsupported GMM specification: {clustering_algorithm_choice} (Autotune path)")
            raise Exception(f"Unsupported GMM specification: {clustering_algorithm_choice}")
        
        clustering = clustering_GMM(data, X_reduced_features, max_clusters, GMM_type, original_labels_aligned, global_known_normal_samples_pca=global_known_normal_samples_pca, threshold_value=threshold_value, num_processes_for_algo=num_processes_for_algo)

    elif clustering_algorithm_choice == 'SGMM':
        clustering = clustering_SGMM(data, X_reduced_features, max_clusters, original_labels_aligned, global_known_normal_samples_pca=global_known_normal_samples_pca, threshold_value=threshold_value, num_processes_for_algo=num_processes_for_algo)

    elif clustering_algorithm_choice in ['Gmeans', 'gmeans']:
        clustering = clustering_Gmeans(data, X_reduced_features, original_labels_aligned, global_known_normal_samples_pca=global_known_normal_samples_pca, threshold_value=threshold_value, num_processes_for_algo=num_processes_for_algo)

    elif clustering_algorithm_choice in ['Xmeans', 'xmeans']:
        clustering = clustering_Xmeans(data, X_reduced_features, original_labels_aligned, global_known_normal_samples_pca=global_known_normal_samples_pca, threshold_value=threshold_value, num_processes_for_algo=num_processes_for_algo)

    elif clustering_algorithm_choice == 'DBSCAN':
        clustering = clustering_DBSCAN(data, X_reduced_features, original_labels_aligned, global_known_normal_samples_pca=global_known_normal_samples_pca, threshold_value=threshold_value, num_processes_for_algo=num_processes_for_algo)

    elif clustering_algorithm_choice == 'MShift':
        clustering = clustering_MShift(data, X_reduced_features, original_labels_aligned, global_known_normal_samples_pca=global_known_normal_samples_pca, threshold_value=threshold_value, num_processes_for_algo=num_processes_for_algo)

    elif clustering_algorithm_choice == 'FCM':
        clustering = clustering_FCM(data, X_reduced_features, max_clusters, original_labels_aligned, global_known_normal_samples_pca=global_known_normal_samples_pca, threshold_value=threshold_value, num_processes_for_algo=num_processes_for_algo)

    elif clustering_algorithm_choice == 'CK':
        clustering = clustering_CK(data, X_reduced_features, max_clusters, original_labels_aligned, global_known_normal_samples_pca=global_known_normal_samples_pca, threshold_value=threshold_value, num_processes_for_algo=num_processes_for_algo)

    elif clustering_algorithm_choice == 'NeuralGas':
        clustering = clustering_NeuralGas(data, X_reduced_features, original_labels_aligned, global_known_normal_samples_pca=global_known_normal_samples_pca, threshold_value=threshold_value, num_processes_for_algo=num_processes_for_algo)

    elif clustering_algorithm_choice == 'CLARA':
        # For Autotune, CLARA's 'k' is determined by an Elbow method upstream.
        # We pass max_clusters as the k value here.
        clustering = clustering_CLARA(data, X_reduced_features, original_labels_aligned, global_known_normal_samples_pca=global_known_normal_samples_pca, threshold_value=threshold_value, num_processes_for_algo=num_processes_for_algo, k=max_clusters)

    else:
        print(f"Unsupported algorithm: {clustering_algorithm_choice}")
        raise Exception(f"Unsupported clustering algorithm: {clustering_algorithm_choice}")

    if clustering is None:
        # This check might be problematic if CANNwKNN is expected to return None or a different structure not caught by this.
        # However, CANNwKNN was modified to return a dict similar to others.
        raise Exception(f"Clustering result is None for algorithm: {clustering_algorithm_choice}")

    return clustering, GMM_type


def choose_clustering_algorithm_for_cache(data, X_reduced_features, original_labels_aligned, clustering_algorithm_choice, max_clusters=1000, global_known_normal_samples_pca=None, threshold_value=0.3, num_processes_for_algo=1, dbscan_eps=None, k=None):
    """
    A new version of choose_clustering_algorithm that is designed to return the raw,
    unprocessed cluster labels, which are needed for the caching mechanism that allows
    for rapid evaluation of different thresholds.
    un-re-labeled cluster IDs, which is necessary for the caching mechanism.

    This function will be expanded to handle all clustering types, but starts with K-Means.
    """
    GMM_type = None
    clustering_result = None

    if clustering_algorithm_choice in ['Kmeans', 'kmeans']:
        # This now returns a richer dictionary including raw labels and num_clusters
        clustering_result = clustering_Kmeans(
            data, X_reduced_features, max_clusters, original_labels_aligned,
            global_known_normal_samples_pca=global_known_normal_samples_pca,
            threshold_value=threshold_value,
            num_processes_for_algo=num_processes_for_algo,
            k=k # Pass k value down
        )
    elif clustering_algorithm_choice in ['Kmedians', 'kmedians']:
        clustering_result = clustering_Kmedians(
            data, X_reduced_features, max_clusters, original_labels_aligned,
            global_known_normal_samples_pca=global_known_normal_samples_pca,
            threshold_value=threshold_value,
            num_processes_for_algo=num_processes_for_algo,
            k=k # Pass k value down
        )
    elif clustering_algorithm_choice.upper().startswith('GMM'):
        parts = clustering_algorithm_choice.split('_')
        if len(parts) == 1 and parts[0].upper() == 'GMM':
            GMM_type = "normal"
        elif len(parts) == 2 and parts[0].upper() == 'GMM' and parts[1].lower() in ['normal', 'full', 'tied', 'diag']:
            GMM_type = parts[1].lower()
        else:
            raise Exception(f"Unsupported GMM specification for caching: {clustering_algorithm_choice}")
        
        clustering_result = clustering_GMM(
            data, X_reduced_features, max_clusters, GMM_type, original_labels_aligned,
            global_known_normal_samples_pca=global_known_normal_samples_pca,
            threshold_value=threshold_value,
            num_processes_for_algo=num_processes_for_algo,
            k=k # Pass k value down
        )
    elif clustering_algorithm_choice == 'DBSCAN':
        clustering_result = clustering_DBSCAN(
            data, X_reduced_features, original_labels_aligned,
            global_known_normal_samples_pca=global_known_normal_samples_pca,
            threshold_value=threshold_value,
            num_processes_for_algo=num_processes_for_algo,
            eps=dbscan_eps # Pass eps value down
        )
    elif clustering_algorithm_choice == 'MShift':
        clustering_result = clustering_MShift(
            data, X_reduced_features, original_labels_aligned,
            global_known_normal_samples_pca=global_known_normal_samples_pca,
            threshold_value=threshold_value,
            num_processes_for_algo=num_processes_for_algo
        )
    elif clustering_algorithm_choice == 'FCM':
        clustering_result = clustering_FCM(
            data, X_reduced_features, max_clusters, original_labels_aligned,
            global_known_normal_samples_pca=global_known_normal_samples_pca,
            threshold_value=threshold_value,
            num_processes_for_algo=num_processes_for_algo,
            k=k # Pass k value down
        )
    elif clustering_algorithm_choice == 'CK':
        clustering_result = clustering_CK(
            data, X_reduced_features, max_clusters, original_labels_aligned,
            global_known_normal_samples_pca=global_known_normal_samples_pca,
            threshold_value=threshold_value,
            num_processes_for_algo=num_processes_for_algo,
            k=k # Pass k value down
        )
    elif clustering_algorithm_choice == 'NeuralGas':
        clustering_result = clustering_NeuralGas(
            data, X_reduced_features, original_labels_aligned,
            global_known_normal_samples_pca=global_known_normal_samples_pca,
            threshold_value=threshold_value,
            num_processes_for_algo=num_processes_for_algo
        )
    elif clustering_algorithm_choice == 'SGMM':
        clustering_result = clustering_SGMM(
            data, X_reduced_features, max_clusters, original_labels_aligned,
            global_known_normal_samples_pca=global_known_normal_samples_pca,
            threshold_value=threshold_value,
            num_processes_for_algo=num_processes_for_algo,
            k=k # Pass k value down
        )
    elif clustering_algorithm_choice in ['Gmeans', 'gmeans']:
        clustering_result = clustering_Gmeans(
            data, X_reduced_features, original_labels_aligned,
            global_known_normal_samples_pca=global_known_normal_samples_pca,
            threshold_value=threshold_value,
            num_processes_for_algo=num_processes_for_algo
        )
    elif clustering_algorithm_choice in ['Xmeans', 'xmeans']:
        clustering_result = clustering_Xmeans(
            data, X_reduced_features, original_labels_aligned,
            global_known_normal_samples_pca=global_known_normal_samples_pca,
            threshold_value=threshold_value,
            num_processes_for_algo=num_processes_for_algo
        )
    # =================================================================================
    # TODO: Add elif blocks for all other clustering algorithms (CANNwKNN, etc.)
    # Each block should call its respective clustering function, which in turn needs
    # to be refactored to return the 'raw_cluster_labels' and 'num_clusters' keys.
    # For now, we will raise an error for other types.
    # =================================================================================
    else:
        print(f"Unsupported algorithm for caching: {clustering_algorithm_choice}")
        raise NotImplementedError(f"The caching function does not yet support '{clustering_algorithm_choice}'. Please implement it.")

    if clustering_result is None:
        raise Exception(f"Clustering result is None for algorithm: {clustering_algorithm_choice}")

    # The GMM_type is not handled here yet, but will be needed for GMM.
    return clustering_result, GMM_type
