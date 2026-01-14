import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


# 1. Cluster center utilities

def compute_centroid(X):
    return np.mean(X, axis=0)


def compute_medoid(X):
    D = pairwise_distances(X)
    return X[np.argmin(D.sum(axis=1))]


# 2. Intra-cluster inconsistency detection

def distance_based_outliers(
    X,
    cluster_labels,
    method="centroid",
    percentile=97.5
):
    inconsistent = defaultdict(set)

    for c in np.unique(cluster_labels):
        idx = np.where(cluster_labels == c)[0]
        Xc = X[idx]

        if len(Xc) < 5:
            continue

        center = compute_centroid(Xc) if method == "centroid" else compute_medoid(Xc)
        dists = np.linalg.norm(Xc - center, axis=1)
        thr = np.percentile(dists, percentile)

        for i, d in zip(idx, dists):
            if d > thr:
                inconsistent[c].add(i)

    return inconsistent


def feature_deviation_outliers(
    X,
    cluster_labels,
    feature_weights=None,
    percentile=95,
    eps=1e-6
):
    inconsistent = defaultdict(set)
    feature_weights = (
        np.ones(X.shape[1]) if feature_weights is None else feature_weights
    )

    for c in np.unique(cluster_labels):
        idx = np.where(cluster_labels == c)[0]
        Xc = X[idx]

        if len(Xc) < 5:
            continue

        mu = Xc.mean(axis=0)
        sigma = Xc.std(axis=0) + eps

        deviation = np.abs((Xc - mu) / sigma)
        scores = (deviation * feature_weights).mean(axis=1)
        thr = np.percentile(scores, percentile)

        for i, s in zip(idx, scores):
            if s > thr:
                inconsistent[c].add(i)

    return inconsistent


def assign_soft_labels(labels, inconsistent_indices, uncertain_label=-1):
    # This does NOT flip labels; it only assigns a temporary -1.
    labels = labels.copy()
    for idx in inconsistent_indices:
        labels[idx] = uncertain_label
    return labels


# 3. Conservative relabeling logic (Main Logic)

def conservative_relabel(
    cluster_labels,
    inconsistent_by_cluster,
    max_inconsistent_ratio=0.1,
    benign_ratio_threshold=0.7
):
    """
    Rules:
    - attack -> benign flip: NEVER
    - Inconsistent samples from benign-heavy clusters are flipped to attack
    - If the inconsistent ratio is too high, the cluster refinement is aborted
    """

    refined = cluster_labels.copy()

    for c, inconsistent_idx in inconsistent_by_cluster.items():
        idx = np.where(cluster_labels == c)[0]
        cluster_size = len(idx)

        if cluster_size == 0:
            continue

        ratio = len(inconsistent_idx) / cluster_size
        if ratio > max_inconsistent_ratio:
            # refinement abort for this cluster
            continue

        # Cluster majority determination
        benign_ratio = np.mean(cluster_labels[idx] == 0)

        print(
            f"[Refine Debug] Cluster {c}: "
            f"size={cluster_size}, "
            f"inconsistent={len(inconsistent_idx)}, "
            f"ratio={ratio:.3f}, "
            f"benign_ratio={benign_ratio:.3f}"
        )

        # Flips are permitted only in benign-heavy clusters
        if benign_ratio >= benign_ratio_threshold:
            for i in inconsistent_idx:
                # Only 0 -> 1 allowed
                if refined[i] == 0:
                    refined[i] = 1
    
    before = np.sum(cluster_labels == 1)
    after  = np.sum(refined == 1)
    print(f"[Refine Debug] Attack count: {before} -> {after} (Δ={after-before})")

    return refined


# 4. Unified refinement interface (pipeline entry point)

def refine_clusters(
    X,
    cluster_labels,
    feature_weights=None,
    config=None
):
    """
    Final refinement entry.
    Always returns refined {0,1} labels.
    """

    config = config or {}

    # Step 1: detect inconsistent samples
    inconsistent = defaultdict(set)

    if config.get("distance", True):
        out_d = distance_based_outliers(
            X,
            cluster_labels,
            method=config.get("center", "centroid"),
            percentile=config.get("distance_percentile", 80.0),
        )
        for c in out_d:
            inconsistent[c] |= out_d[c]

    if config.get("feature_deviation", True):
        out_f = feature_deviation_outliers(
            X,
            cluster_labels,
            feature_weights=feature_weights,
            percentile=config.get("feature_percentile", 85),
        )
        for c in out_f:
            inconsistent[c] |= out_f[c]

    # Step 2: conservative relabeling
    refined_labels = conservative_relabel(
        cluster_labels,
        inconsistent,
        max_inconsistent_ratio=config.get("max_inconsistent_ratio", 0.25),
        benign_ratio_threshold=config.get("benign_ratio_threshold", 0.65),
    )

    return refined_labels
