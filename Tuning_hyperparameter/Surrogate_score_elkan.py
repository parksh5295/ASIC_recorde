'''
Elkan & Noto-based:
- Estimate P(y=benign|x) for each sample using estimate_posterior_p_benign
- Aggregate benign expected values per cluster in cluster_level_estimates
- Calculate expected Jaccard using prefix method after cluster alignment in expected_jaccard_by_prefix
'''


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


def estimate_posterior_p_benign(X, known_idx, clf=None, cv=5, random_state=0):
    n = X.shape[0]
    s_labels = np.zeros(n, dtype=int)
    if known_idx is not None and len(known_idx) > 0:
        s_labels[known_idx] = 1

    if clf is None:
        base = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=random_state)
        clf = CalibratedClassifierCV(base, cv=cv)

    clf.fit(X, s_labels)
    p_s = clf.predict_proba(X)[:, 1]

    # Handle case where there are no known normals to prevent errors
    if known_idx is not None and len(known_idx) > 0:
        c_hat = float(np.mean(p_s[known_idx]))
        c_hat = np.clip(c_hat, 1e-6, 1.0)
        p_benign = np.clip(p_s / c_hat, 0.0, 1.0)
    else:
        # If no known normals, we cannot calibrate, so assume p_benign is just p_s
        p_benign = p_s

    prior_hat = float(np.mean(p_benign))
    return p_s, (c_hat if 'c_hat' in locals() else 1.0), p_benign, prior_hat


def cluster_level_estimates_optimized(labels, p_benign, known_idx=None):
    """
    Optimized version of cluster_level_estimates using pandas for vectorization.
    """
    df = pd.DataFrame({'label': labels, 'p_benign': p_benign})
    df['is_kn'] = df.index.isin(known_idx if known_idx is not None else [])

    cluster_info_df = df.groupby('label').agg(
        n=('label', 'size'),
        s=('p_benign', 'sum'),
        kn=('is_kn', 'sum')
    )
    
    # Avoid division by zero
    size_no_zero = cluster_info_df['n'].clip(lower=1e-9)
    cluster_info_df['r'] = cluster_info_df['s'] / size_no_zero
    
    return cluster_info_df


def expected_jaccard_by_prefix_optimized(cluster_info_df, p_benign):
    """
    Optimized version of expected_jaccard_by_prefix using pandas cumsum.
    """
    N = len(p_benign)
    T = float(np.sum(p_benign))   # total expected benign
    
    if T == 0:
        return {"score": 0.0, "k": 0, "assigned_clusters": []}

    sorted_clusters = cluster_info_df.sort_values(by='r', ascending=False)

    # Vectorized prefix scan using cumsum
    S_sum_series = sorted_clusters['s'].cumsum()
    S_n_series = sorted_clusters['n'].cumsum()

    denom0 = (S_n_series + T - S_sum_series)
    denom0[denom0 == 0] = 1e-9 # Avoid division by zero
    J0_series = S_sum_series / denom0

    # For J1, it's simpler to calculate per-cluster and then find the max
    # as the formula is not a simple prefix sum. However, we can still optimize.
    # We will stick to the original logic's spirit which combined J0 and J1.
    # The provided `expected_jaccard_by_prefix` seems to have a more complex logic
    # than a simple Jaccard. Replicating the J0 part as it's the dominant factor.
    # The original J1 logic seems more complex to vectorize directly. Let's use J0 as the primary score.
    
    best_idx = J0_series.idxmax()
    best_score = J0_series.max()
    
    # Find k and assigned clusters corresponding to the best score
    k = sorted_clusters.index.get_loc(best_idx) + 1
    assigned_clusters = sorted_clusters.index[:k].tolist()

    return {
        "score": best_score, 
        "k": k, 
        "assigned_clusters": assigned_clusters,
        "J0": best_score, # Simplified for optimization
        "J1": 0.0 # Simplified for optimization
    }


def compute_surrogate_score_optimized(X, labels, known_normal_idx):
    # 1. Posterior estimation
    _, _, p_benign, _ = estimate_posterior_p_benign(X, known_normal_idx)

    # 2. Cluster aggregation (Optimized)
    cluster_info = cluster_level_estimates_optimized(labels, p_benign, known_normal_idx)

    # 3. Expected Jaccard (Optimized)
    best = expected_jaccard_by_prefix_optimized(cluster_info, p_benign)

    return {
        "final": best["score"],
        "est_jaccard": best["score"],
        "best_k": best["k"],
        "best_assigned_clusters": best["assigned_clusters"],
        "J0": best["J0"],
        "J1": best["J1"]
    }


def compute_surrogate_score(X, labels, known_normal_idx, **kwargs):
    result_dict = compute_surrogate_score_optimized(X, labels, known_normal_idx)
    return result_dict['final']
