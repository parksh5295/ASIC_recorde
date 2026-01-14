import numpy as np
from collections import Counter
from math import log10
from scipy.stats import hypergeom
import pandas as pd


def compute_surrogate_score_optimized(X, labels, known_normal_idx, alpha=0.6, beta=0.25, gamma=0.15, thresholds=[0.5,0.6,0.7,0.8,0.9]):
    """
    An optimized version of the surrogate score calculation using vectorized
    operations with pandas and numpy for improved performance.
    """
    n_samples = X.shape[0]
    total_kn = len(known_normal_idx)

    # --- 1. Cluster Summary (Optimized) ---
    # Use pandas for efficient, vectorized grouping and aggregation
    df = pd.DataFrame({'label': labels})
    df['is_kn'] = df.index.isin(known_normal_idx)

    # Aggregate size and known normal counts per cluster
    cluster_info_df = df.groupby('label').agg(
        size=('label', 'size'),
        kn=('is_kn', 'sum')
    )
    # This check is important for cases where no known normals are present
    if 'kn' not in cluster_info_df.columns:
        cluster_info_df['kn'] = 0

    # --- 2. Best F1 Score (Optimized) ---
    # Vectorized calculation of normal ratio
    cluster_info_df['normal_ratio'] = cluster_info_df['kn'] / cluster_info_df['size']
    
    best_f1 = -1.0
    best_t = None
    best_assigned_clusters = []

    if total_kn > 0:
        for t in thresholds:
            assigned_normal_clusters = cluster_info_df[cluster_info_df['normal_ratio'] >= t]
            
            kn_in_assigned = assigned_normal_clusters['kn'].sum()
            total_in_assigned = assigned_normal_clusters['size'].sum()
            
            recall = kn_in_assigned / total_kn
            precision = kn_in_assigned / total_in_assigned if total_in_assigned > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
                best_assigned_clusters = assigned_normal_clusters.index.tolist()

    # --- 3. Compactness (Optimized) ---
    # Combine X and labels for efficient processing
    X_df = pd.DataFrame(X)
    X_df['label'] = labels
    
    # Calculate centroids for all clusters at once
    centroids = X_df.groupby('label').transform('mean')
    
    # Calculate squared distances for all points and then mean per cluster
    squared_distances = ((X_df.drop('label', axis=1) - centroids) ** 2).sum(axis=1)
    mean_dist_per_cluster = squared_distances.groupby(labels).mean()
    
    # Average compactness over all clusters
    avg_compactness = mean_dist_per_cluster.mean()
    compact_score = 1.0 / (1.0 + avg_compactness)

    # --- 4. Enrichment Score (using scipy which is fast) ---
    # hypergeom.sf is already vectorized-friendly and fast
    p_values = hypergeom.sf(cluster_info_df['kn'] - 1, n_samples, total_kn, cluster_info_df['size'])
    # Guard against p=0 -> log(0) error
    p_values[p_values == 0] = 1e-300
    log_p_values = -np.log10(p_values)
    enrich = log_p_values.sum()
    enrich_score = np.tanh(enrich / (1.0 + enrich))

    # --- 5. Combine scores ---
    final_score = alpha * best_f1 + beta * compact_score + gamma * enrich_score

    return {
        "final": final_score,
        "f1_known": best_f1,
        "compact_score": compact_score,
        "enrich_score": enrich_score,
        "best_t": best_t,
        "best_assigned_clusters": best_assigned_clusters
    }


# --- ADAPTER FUNCTION ---
# This function acts as a bridge between the new surrogate score logic
# and the existing Jaccard_Elbow_Method.py which expects this function signature.
def compute_surrogate_score(X, labels, known_normal_idx, **kwargs):
    """
    Adapter function to call the new, OPTIMIZED surrogate score logic and return
    only the final score as a single float, maintaining compatibility.
    Accepts **kwargs to ignore any extra parameters from older calls.
    """
    # Call the new, detailed and OPTIMIZED surrogate function
    result_dict = compute_surrogate_score_optimized(X, labels, known_normal_idx)
    
    # Extract and return only the final score
    return result_dict['final']
