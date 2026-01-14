import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import pairwise_distances
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify
import logging


logger = logging.getLogger(__name__)

def _calculate_dissimilarity_chunked(X, medoids, chunk_size=50000):
    """
    Calculate total dissimilarity (cost) by processing data in chunks
    to avoid creating a massive distance matrix in memory.
    """
    total_dissimilarity = 0
    n_samples = X.shape[0]
    
    for i in range(0, n_samples, chunk_size):
        end = i + chunk_size
        chunk = X[i:end]
        distances = pairwise_distances(chunk, medoids)
        min_distances = np.min(distances, axis=1)
        total_dissimilarity += np.sum(min_distances)
        
    return total_dissimilarity

def _assign_labels_chunked(X, medoids, chunk_size=50000):
    """
    Assign labels to all data points by processing in chunks to manage memory.
    """
    n_samples = X.shape[0]
    labels = np.empty(n_samples, dtype=int)

    for i in range(0, n_samples, chunk_size):
        end = i + chunk_size
        chunk = X[i:end]
        distances = pairwise_distances(chunk, medoids)
        labels[i:end] = np.argmin(distances, axis=1)
    
    return labels

def run_clara_once(X, k, number_samples, random_state=None):
    """
    Run one iteration of CLARA: sample a subset, run KMedoids, 
    then calculate the cost on the full dataset.
    """
    # rng = np.random.default_rng(random_state)
    # Use legacy random for better reproducibility across different CPU architectures
    # default_rng() can give different results on different hardware due to SIMD optimizations
    n = X.shape[0]
    
    if number_samples > n:
        number_samples = n

    # Step 1: Draw a sample from the dataset.
    # sample_indices = rng.choice(n, size=number_samples, replace=False)
    if random_state is not None:
        np.random.seed(random_state)
    sample_indices = np.random.choice(n, size=number_samples, replace=False)
    X_sample = X[sample_indices]

    # Step 2: Run KMedoids (PAM) on the sample to find medoids.
    # n_init=1 is sufficient as CLARA's multiple iterations serve a similar purpose.
    # kmedoids = KMedoids(n_clusters=k, method="pam", max_iter=1000, random_state=random_state) # Slow and no heuristics
    kmedoids = KMedoids(n_clusters=k, method="alternate", max_iter=1000, random_state=random_state)
    #kmedoids = KMedoids(n_clusters=k, method="alternate", tol=1e-3 max_iter=1000, random_state=random_state)
    kmedoids.fit(X_sample)
    medoids = kmedoids.cluster_centers_

    # Step 3: Calculate the total dissimilarity (cost) of these medoids on the ENTIRE dataset.
    # This is the crucial step to evaluate the quality of the sample's medoids.
    # We use a chunked calculation to prevent memory errors.
    total_dissimilarity = _calculate_dissimilarity_chunked(X, medoids)
    
    return medoids, total_dissimilarity


def clustering_CLARA(data, X, aligned_original_labels, global_known_normal_samples_pca,
                     threshold_value, num_processes_for_algo, k,
                     number_samples=None, number_iterations=5, random_state=42):
    """
    Custom and memory-efficient implementation of CLARA.
    """
    logger.info(f"[CLARA Custom] Starting CLARA with k={k}...")

    if number_samples is None:
        number_samples = min(X.shape[0], 40 + 2 * k)

    best_dissimilarity = np.inf
    best_medoids = None
    
    # In each iteration, we find a set of medoids and see if it's the best set so far.
    for it in range(number_iterations):
        # We need a different random state for each sampling iteration
        iter_random_state = random_state + it if random_state is not None else None
        medoids, total_dissimilarity = run_clara_once(X, k, number_samples, iter_random_state)
        
        if total_dissimilarity < best_dissimilarity:
            best_dissimilarity = total_dissimilarity
            best_medoids = medoids

    # After finding the best medoids, assign all data points to them.
    # This is also done in chunks to manage memory.
    if best_medoids is not None:
        raw_cluster_labels = _assign_labels_chunked(X, best_medoids)
    else:
        logger.error(f"[CLARA Custom] Failed to find any valid medoids for k={k}")
        raw_cluster_labels = None

    # Apply CNI to the final labels
    try:
        if raw_cluster_labels is None:
            raise ValueError("Clustering failed, labels are None.")

        final_labels, jaccard_score, _ = clustering_nomal_identify(
            data_features_for_clustering=X,
            clusters_assigned=raw_cluster_labels,
            original_labels_aligned=aligned_original_labels,
            global_known_normal_samples_pca=global_known_normal_samples_pca,
            threshold_value=threshold_value,
            num_processes_for_algo=num_processes_for_algo,
            data_for_clustering=X
        )

        logger.info(f"[CLARA Custom] Completed with k={k}. Jaccard: {jaccard_score:.4f}")

        return {
            'raw_cluster_labels': raw_cluster_labels,
            'Cluster_labeling': final_labels,
            'Jaccard': jaccard_score,
            'Best_parameter_dict': {
                'k': k, # or 'n_clusters' for consistency in grid search
                'n_clusters': k,
                'number_samples': number_samples,
                'number_iterations': number_iterations
            }
        }

    except Exception as e:
        logger.error(f"[CLARA Custom] Error during clustering with k={k}: {e}")
        return {
            'raw_cluster_labels': None,
            'Cluster_labeling': np.array([]),
            'Jaccard': 0.0,
            'Best_parameter_dict': {}
        }


def pre_clustering_CLARA(data, X, k,
                         num_processes_for_algo=1,
                         number_samples=None,
                         number_iterations=5,
                         random_state=42):
    """
    Pre-clustering wrapper for custom CLARA without CNI adjustment.
    """
    logger.info(f"[CLARA Custom Pre-clustering] Starting with k={k}...")

    if number_samples is None:
        number_samples = min(X.shape[0], 40 + 2 * k)

    best_dissimilarity = np.inf
    best_medoids = None

    for it in range(number_iterations):
        iter_random_state = random_state + it if random_state is not None else None
        medoids, total_dissimilarity = run_clara_once(X, k, number_samples, iter_random_state)
        if total_dissimilarity < best_dissimilarity:
            best_dissimilarity = total_dissimilarity
            best_medoids = medoids

    if best_medoids is None:
        logger.error(f"[CLARA Custom Pre-clustering] Failed for k={k}")
        return {'model_labels': None, 'n_clusters': 0}
        
    raw_cluster_labels = _assign_labels_chunked(X, best_medoids)

    return {
        'model_labels': raw_cluster_labels,
        'n_clusters': len(np.unique(raw_cluster_labels))
    }