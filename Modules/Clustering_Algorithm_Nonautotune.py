# A clustering module that uses default values, where each parameter is not optimized
# a control group for this clustering module.
# Output: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}; Name: Clustering

from Clustering_Method.clustering_Kmeans import pre_clustering_Kmeans
from Clustering_Method.clustering_Kmedians import pre_clustering_Kmedians
from Clustering_Method.clustering_GMM import pre_clustering_GMM
from Clustering_Method.clustering_SGMM import pre_clustering_SGMM
from Clustering_Method.clustering_Gmeans import pre_clustering_Gmeans
from Clustering_Method.clustering_Xmeans import pre_clustering_Xmeans
from Clustering_Method.clustering_DBSCAN import pre_clustering_DBSCAN
from Clustering_Method.clustering_MShift import pre_clustering_MShift
from Clustering_Method.clustering_FCM import pre_clustering_FCM
from Clustering_Method.clustering_CK import pre_clustering_CK
from Clustering_Method.clustering_NeuralGas import pre_clustering_NeuralGas
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify
from Clustering_Method.clustering_CLARA import pre_clustering_CLARA


def choose_clustering_algorithm_Non_optimization(data, X_reduced_features, original_labels_aligned, clustering_algorithm_choice, global_known_normal_samples_pca=None, threshold_value=0.3, num_processes_for_algo=1):
    parameter_dict = {'random_state' : 42, 'n_init' : 30, 'max_clusters' : 1000, 'tol' : 1e-4, 'eps' : 0.5, 'count_samples' : 5,
                        'quantile' : 0.2, 'n_samples' : 500, 'n_start_nodes' : 2, 'max_nodes' : 50, 'step' : 0.2,
                        'max_edge_age' : 50, 'epochs' : 300, 'batch_size' : 256, 'n_neighbors' : 5, 'n_clusters' : 1000
                        }

    GMM_type = None
    pre_clustering_result = None # To store result from pre_clustering functions

    # Pass original_labels_aligned to each pre_clustering function call.
    # The 'data' and 'X_reduced_features' arguments are passed as required by pre_clustering functions.

    if clustering_algorithm_choice in ['Kmeans', 'kmeans']:
        pre_clustering_result = pre_clustering_Kmeans(data, X_reduced_features, parameter_dict['n_clusters'], parameter_dict['random_state'], parameter_dict['n_init'], num_processes_for_algo=num_processes_for_algo)

    elif clustering_algorithm_choice in ['Kmedians', 'kmedians']:
        pre_clustering_result = pre_clustering_Kmedians(data, X_reduced_features, parameter_dict['n_clusters'], parameter_dict['random_state'], num_processes_for_algo=num_processes_for_algo)

    # Enhanced GMM type handling for Non-Autotune
    elif clustering_algorithm_choice.upper().startswith('GMM'):
        parts = clustering_algorithm_choice.split('_')
        if len(parts) == 1 and parts[0].upper() == 'GMM': # Only "GMM"
            # GMM_type = input("Please enter the GMM type, i.e. normal, full, tied, diag: ") # 주석 처리
            GMM_type = "normal" # Default to normal
            print(f"[INFO] GMM algorithm selected (Non-Autotune). Defaulting to GMM type: {GMM_type}")
        elif len(parts) == 2 and parts[0].upper() == 'GMM' and parts[1].lower() in ['normal', 'full', 'tied', 'diag']:
            GMM_type = parts[1].lower()
            print(f"[INFO] Using GMM type '{GMM_type}' from algorithm choice: {clustering_algorithm_choice} (Non-Autotune)")
        else:
            print(f"Unsupported GMM specification: {clustering_algorithm_choice} (Non-Autotune)")
            raise Exception(f"Unsupported GMM specification: {clustering_algorithm_choice}")
        
        pre_clustering_result = pre_clustering_GMM(data, X_reduced_features, parameter_dict['n_clusters'], parameter_dict['random_state'], GMM_type, n_init=parameter_dict['n_init'], num_processes_for_algo=num_processes_for_algo)

    elif clustering_algorithm_choice == 'SGMM':
        pre_clustering_result = pre_clustering_SGMM(data, X_reduced_features, 
                                                    parameter_dict['n_clusters'], 
                                                    parameter_dict['random_state'],
                                                    n_init=parameter_dict['n_init'], num_processes_for_algo=num_processes_for_algo)

    elif clustering_algorithm_choice in ['Gmeans', 'gmeans']:
        pre_clustering_result = pre_clustering_Gmeans(data, X_reduced_features, parameter_dict['random_state'], parameter_dict['max_clusters'], parameter_dict['tol'], n_init=parameter_dict['n_init'], num_processes_for_algo=num_processes_for_algo)

    elif clustering_algorithm_choice in ['Xmeans', 'xmeans']:
        pre_clustering_result = pre_clustering_Xmeans(data, X_reduced_features, parameter_dict['random_state'], parameter_dict['max_clusters'], n_init=parameter_dict['n_init'], num_processes_for_algo=num_processes_for_algo)

    elif clustering_algorithm_choice == 'DBSCAN':
        pre_clustering_result = pre_clustering_DBSCAN(data, X_reduced_features, parameter_dict['eps'], parameter_dict['count_samples'], num_processes_for_algo=num_processes_for_algo)

    elif clustering_algorithm_choice == 'MShift':
        pre_clustering_result = pre_clustering_MShift(data, X_reduced_features, parameter_dict['random_state'], parameter_dict['quantile'], parameter_dict['n_samples'], num_processes_for_algo=num_processes_for_algo)

    elif clustering_algorithm_choice == 'FCM':
        pre_clustering_result = pre_clustering_FCM(data, X_reduced_features, parameter_dict['n_clusters'], num_processes_for_algo=num_processes_for_algo)

    elif clustering_algorithm_choice == 'CK':
        pre_clustering_result = pre_clustering_CK(data, X_reduced_features, parameter_dict['n_clusters'], n_init_for_ck=parameter_dict['n_init'], num_processes_for_algo=num_processes_for_algo)

    elif clustering_algorithm_choice == 'NeuralGas':
        pre_clustering_result = pre_clustering_NeuralGas(data, X_reduced_features, parameter_dict['n_start_nodes'], parameter_dict['max_nodes'], parameter_dict['step'], parameter_dict['max_edge_age'], num_processes_for_algo=num_processes_for_algo)

    elif clustering_algorithm_choice == 'CLARA':
        pre_clustering_result = pre_clustering_CLARA(data, X_reduced_features, parameter_dict['n_clusters'], num_processes_for_algo=num_processes_for_algo)

    elif clustering_algorithm_choice in ['CANNwKNN', 'CANN']:
        print(f"[INFO] CANNwKNN/CANN selected for Non-optimization. Passing global_known_normal_samples_pca for consistency.")
        pre_clustering_result = pre_clustering_CANNwKNN(data, X_reduced_features, parameter_dict['epochs'], parameter_dict['batch_size'], parameter_dict['n_neighbors'], num_processes_for_algo=num_processes_for_algo)
        
        final_cluster_labels = pre_clustering_result['model_labels']
        
        # For CANNwKNN, the parameter_dict is fixed and known.
        # The 'Best_parameter_dict' is more relevant for autotuning.
        # Here, we just return the fixed dict used.
        return {
            'Cluster_labeling': final_cluster_labels,
            'Best_parameter_dict': {
                'epochs': parameter_dict['epochs'],
                'batch_size': parameter_dict['batch_size'],
                'n_neighbors': parameter_dict['n_neighbors']
            }
        }, GMM_type

    else:
        print("Unsupported algorithm")
        raise Exception("Unsupported clustering algorithms")
    
    if pre_clustering_result is None or 'model_labels' not in pre_clustering_result or 'n_clusters' not in pre_clustering_result:
        raise Exception(f"Pre-clustering for {clustering_algorithm_choice} failed or returned unexpected format.")

    model_labels = pre_clustering_result['model_labels']
    n_clusters_actual = pre_clustering_result['n_clusters'] # Actual number of clusters from pre_clustering

    print(f"\n[DEBUG NonAutoTune {clustering_algorithm_choice}] Param for CNI 'data_features_for_clustering' (X_reduced_features) - Shape: {X_reduced_features.shape}")
    
    final_cluster_labels_from_cni = clustering_nomal_identify(
        X_reduced_features, 
        original_labels_aligned, 
        model_labels, 
        n_clusters_actual, 
        global_known_normal_samples_pca=global_known_normal_samples_pca,
        threshold_value=threshold_value,
        num_processes_for_algo=num_processes_for_algo
    )

    # For Non-optimization, Best_parameter_dict is just the fixed parameter_dict.
    # We can select relevant keys if needed, or return the whole dict.
    # For simplicity, returning the whole dict.
    return {
        'Cluster_labeling': final_cluster_labels_from_cni,
        'Best_parameter_dict': parameter_dict 
    }, GMM_type