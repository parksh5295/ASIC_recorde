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

# --- Elkan & Noto 기반 함수들 ---
def estimate_posterior_p_benign(cluster_info, pi=0.9):
    """
    Estimate posterior P(y=benign|cluster) using Elkan & Noto calibration.
    cluster_info: DataFrame with columns ['size', 'kn']
    pi: prior probability of benign (assumed known or chosen)
    """
    size = cluster_info['size'].copy()
    size[size == 0] = 1e-9  # avoid zero-div
    p_obs = cluster_info['kn'] / size
    # Elkan & Noto: P(y=1|x) ≈ P(s=1|x) / pi
    p_post = (p_obs / max(pi, 1e-6)).clip(0, 1)
    return p_post


def cluster_level_estimates(cluster_info, p_post):
    """
    Compute expected benign count per cluster using posterior probabilities.
    """
    exp_benign = p_post * cluster_info['size']
    cluster_info = cluster_info.copy()
    cluster_info['exp_benign'] = exp_benign
    return cluster_info


def expected_jaccard_by_prefix(cluster_info):
    """
    Sort clusters by benign density and compute expected Jaccard via prefix scan.
    Returns the maximum expected Jaccard over prefixes.
    """
    # benign density = expected benign fraction
    cluster_info = cluster_info.copy()
    cluster_info['benign_density'] = (
        cluster_info['exp_benign'] / cluster_info['size'].clip(lower=1e-9)
    )

    # sort clusters by benign density (descending)
    sorted_clusters = cluster_info.sort_values(
        by='benign_density', ascending=False
    )

    exp_total_benign = cluster_info['exp_benign'].sum()
    best_jaccard = 0.0

    cum_exp_benign = 0.0
    cum_size = 0

    for _, row in sorted_clusters.iterrows():
        cum_exp_benign += row['exp_benign']
        cum_size += row['size']
        inter = cum_exp_benign
        union = cum_size + (exp_total_benign - cum_exp_benign)
        jaccard = inter / union if union > 0 else 0.0
        best_jaccard = max(best_jaccard, jaccard)

    return best_jaccard


# --- Main surrogate scoring ---
def compute_surrogate_score_optimized(X, labels, known_normal_idx,
                                      alpha=0.5, beta=0.2,
                                      gamma=0.15, delta=0.15,
                                      thresholds=[0.5,0.6,0.7,0.8,0.9],
                                      pi=None, file_type=None):

    # If pi is not provided, get the default value based on file_type
    if pi is None:
        pi = get_default_pi(file_type)

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
    cluster_info_df['normal_ratio'] = cluster_info_df['kn'] / cluster_info_df['size']
    best_f1, best_t, best_assigned_clusters = 0.0, None, []
    if total_kn > 0:
        for t in thresholds:
            assigned = cluster_info_df[cluster_info_df['normal_ratio'] >= t]
            kn_in = assigned['kn'].sum()
            total_in = assigned['size'].sum()
            recall = kn_in / total_kn if total_kn > 0 else 0.0
            precision = kn_in / total_in if total_in > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision+recall)>0 else 0.0
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
    p_values = hypergeom.sf(cluster_info_df['kn'] - 1, n_samples, total_kn, cluster_info_df['size'])
    p_values[p_values == 0] = 1e-300
    log_p = -np.log10(p_values)
    enrich_score = np.tanh(log_p.sum() / (1.0 + log_p.sum()))

    # --- 5. Expected Jaccard (Elkan & Noto) ---
    if total_kn > 0:
        p_post = estimate_posterior_p_benign(cluster_info_df, pi=pi)
        cluster_est = cluster_level_estimates(cluster_info_df, p_post)
        est_jaccard = expected_jaccard_by_prefix(cluster_est)
    else:
        est_jaccard = 0.0

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
    # Extract file_type from kwargs if it exists, to be passed for pi calculation
    file_type = kwargs.get('file_type')

    result_dict = compute_surrogate_score_optimized(X, labels, known_normal_idx, file_type=file_type)
    return result_dict['final']
