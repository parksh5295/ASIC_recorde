import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from math import log10


def get_default_pi(file_type):
    """
    Returns a default pi value (prior assumption of normal ratio) based on the dataset file_type.
    These are placeholder values and can be adjusted based on domain knowledge.
    """
    pi_defaults = {
        'MiraiBotnet': 0.56,
        'DARPA98': 0.38,
        'Kitsune': 0.54,
        'CICIDS2017': 0.8
        # Add other dataset-specific pi values here
    }
    # Return the specific value, or a general default of 0.9 if not found
    return pi_defaults.get(file_type, 0.9)


# --- Elkan & Noto-based functions (lightweight version) ---
def estimate_posterior_p_benign(cluster_info, pi=0.9):
    """
    Estimate posterior P(y=benign|cluster) using Elkan & Noto calibration.
    This is a lightweight version that does not require an ML model.
    """
    size = cluster_info['size'].copy()
    size[size == 0] = 1e-9  # avoid zero-div
    p_obs = cluster_info['kn'] / size
    p_post = (p_obs / max(pi, 1e-6)).clip(0, 1)
    return p_post

def cluster_level_estimates(cluster_info, p_post):
    """ Compute expected benign count per cluster. """
    exp_benign = p_post * cluster_info['size']
    cluster_info = cluster_info.copy()
    cluster_info['exp_benign'] = exp_benign
    return cluster_info

def expected_jaccard_by_prefix_optimized(cluster_info):
    """
    Optimized Jaccard calculation using vectorized cumsum.
    """
    cluster_info = cluster_info.copy()
    size_no_zero = cluster_info['size'].clip(lower=1e-9)
    cluster_info['benign_density'] = cluster_info['exp_benign'] / size_no_zero
    sorted_clusters = cluster_info.sort_values(by='benign_density', ascending=False)
    exp_total_benign = cluster_info['exp_benign'].sum()
    if exp_total_benign == 0:
        return 0.0
    cum_exp_benign = sorted_clusters['exp_benign'].cumsum()
    cum_size = sorted_clusters['size'].cumsum()
    union = cum_size + (exp_total_benign - cum_exp_benign)
    union[union == 0] = 1e-9
    jaccard_scores = cum_exp_benign / union
    return jaccard_scores.max()


# --- Main surrogate scoring ---
def compute_surrogate_score_optimized(X, labels, known_normal_idx,
                                      alpha=0.5, beta=0.2,
                                      gamma=0.15, delta=0.15,
                                      thresholds=[0.5,0.6,0.7,0.8,0.9],
                                      pi=None, file_type=None):
    """
    Optimized surrogate score calculation combining F1, compactness, enrichment, and estimated Jaccard.
    """
    # If pi is not provided, get the default value based on file_type
    if pi is None:
        pi = get_default_pi(file_type)

    # Robustness check for known_normal_idx
    if known_normal_idx is None or len(known_normal_idx) == 0:
        known_normal_idx = []
    n_samples = X.shape[0]
    total_kn = len(known_normal_idx)

    # --- 1. Cluster Summary ---
    df = pd.DataFrame({'label': labels})
    df['is_kn'] = df.index.isin(known_normal_idx)
    cluster_info_df = df.groupby('label').agg(
        size=('label', 'size'),
        kn=('is_kn', 'sum')
    )
    if 'kn' not in cluster_info_df.columns:
        cluster_info_df['kn'] = 0

    # --- 2. Best F1 ---
    size_series = cluster_info_df['size'].copy()
    size_series[size_series == 0] = 1e-9
    cluster_info_df['normal_ratio'] = cluster_info_df['kn'] / size_series
    best_f1, best_t, best_assigned_clusters = 0.0, None, []
    if total_kn > 0:
        for t in thresholds:
            assigned = cluster_info_df[cluster_info_df['normal_ratio'] >= t]
            kn_in = assigned['kn'].sum()
            total_in = assigned['size'].sum()
            recall = kn_in / total_kn
            precision = kn_in / total_in if total_in > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, t
                best_assigned_clusters = assigned.index.tolist()

    # --- 3. Compactness ---
    X_df = pd.DataFrame(X)
    X_df['label'] = labels
    centroids = X_df.groupby('label').transform('mean')
    sq_dist = ((X_df.drop('label', axis=1) - centroids) ** 2).sum(axis=1)
    mean_dist = sq_dist.groupby(labels).mean()
    compact_score = 1.0 / (1.0 + mean_dist.mean())

    # --- 4. Enrichment ---
    enrich_score = 0.0
    if total_kn > 0:
        clusters_with_kn = cluster_info_df[cluster_info_df['kn'] > 0]
        if not clusters_with_kn.empty:
            p_values = hypergeom.sf(clusters_with_kn['kn'] - 1, n_samples, total_kn, clusters_with_kn['size'])
            p_values[p_values == 0] = 1e-300
            log_p = -np.log10(p_values)
            enrich = log_p.sum()
            enrich_score = np.tanh(enrich / (1.0 + enrich))

    # --- 5. Expected Jaccard (Elkan & Noto) ---
    est_jaccard = 0.0
    if total_kn > 0:
        p_post = estimate_posterior_p_benign(cluster_info_df, pi=pi)
        cluster_est = cluster_level_estimates(cluster_info_df, p_post)
        est_jaccard = expected_jaccard_by_prefix_optimized(cluster_est)

    # --- 6. Combine ---
    final_score = (alpha * best_f1 +
                   beta * compact_score +
                   gamma * enrich_score +
                   delta * est_jaccard)

    return {
        "final": final_score,
        "f1_known": best_f1,
        "compact_score": compact_score,
        "enrich_score": enrich_score,
        "est_jaccard": est_jaccard,
        "best_t": best_t,
        "best_assigned_clusters": best_assigned_clusters
    }

# --- Adapter ---
def compute_surrogate_score(X, labels, known_normal_idx, **kwargs):
    """
    Adapter function to maintain backward compatibility with the existing system,
    which expects a simpler function signature.
    """
    # Extract file_type from kwargs if it exists, to be passed for pi calculation
    file_type = kwargs.get('file_type')
    
    # Call the main optimized function with default hyperparameters
    result_dict = compute_surrogate_score_optimized(X, labels, known_normal_idx, file_type=file_type)
    
    # The old system expects a single float value, so we return the 'final' score.
    return result_dict['final']
