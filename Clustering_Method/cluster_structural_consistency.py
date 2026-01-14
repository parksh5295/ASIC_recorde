import numpy as np
from collections import defaultdict, Counter


# 1. Core feature extraction

def extract_core_feature_mask(
    X,
    top_ratio=0.3,
    variance_threshold=0.5
):
    """
    Identify core features of a cluster based on low variance.
    Returns a boolean mask indicating core dimensions.
    """
    variances = np.var(X, axis=0)
    threshold = np.percentile(variances, variance_threshold * 100)
    core_mask = variances <= threshold

    # Safety: ensure at least some features are selected
    if core_mask.sum() < max(1, int(len(core_mask) * top_ratio)):
        idx = np.argsort(variances)[: int(len(core_mask) * top_ratio)]
        core_mask[:] = False
        core_mask[idx] = True

    return core_mask


def compute_core_signature(X_core):
    """
    Compute cluster core signature as median values.
    """
    return np.median(X_core, axis=0)


# 2. Core consistency scoring

def core_consistency_score(
    X_core,
    core_signature,
    tolerance=0.5
):
    """
    Compute consistency score per sample.
    Score ∈ [0,1], higher = more consistent.
    """
    deviation = np.abs(X_core - core_signature)
    consistent = deviation <= tolerance
    return consistent.mean(axis=1)


# 3. Cluster-level consistency analysis

def detect_core_inconsistent_samples(
    X,
    cluster_labels,
    min_cluster_size=10,
    consistency_threshold=0.6,
    tolerance=0.5
):
    """
    Returns:
        inconsistent_by_cluster: {cluster_id: set(sample_indices)}
    """
    inconsistent = defaultdict(set)

    for c in np.unique(cluster_labels):
        idx = np.where(cluster_labels == c)[0]
        if len(idx) < min_cluster_size:
            continue

        Xc = X[idx]

        core_mask = extract_core_feature_mask(Xc)
        Xc_core = Xc[:, core_mask]

        core_signature = compute_core_signature(Xc_core)
        scores = core_consistency_score(
            Xc_core,
            core_signature,
            tolerance=tolerance
        )

        for i, s in zip(idx, scores):
            if s < consistency_threshold:
                inconsistent[c].add(i)

    return inconsistent


# 4. Conservative relabeling (NO LABEL LEAKAGE)

def conservative_core_relabel(
    cluster_labels,
    inconsistent_by_cluster,
    max_inconsistent_ratio=0.3
):
    """
    Rules:
    - Only 0 -> 1 flips allowed
    - If a cluster is too unstable, refinement is skipped
    """
    refined = cluster_labels.copy()

    for c, bad_idx in inconsistent_by_cluster.items():
        cluster_idx = np.where(cluster_labels == c)[0]
        size = len(cluster_idx)

        if size == 0:
            continue

        ratio = len(bad_idx) / size
        if ratio > max_inconsistent_ratio:
            continue

        for i in bad_idx:
            if refined[i] == 0:
                refined[i] = 1

        print(
            f"[CoreRefine] Cluster {c}: "
            f"size={size}, "
            f"inconsistent={len(bad_idx)}, "
            f"ratio={ratio:.3f}"
        )

    before = np.sum(cluster_labels == 1)
    after = np.sum(refined == 1)
    print(f"[CoreRefine] Attack count: {before} -> {after} (Δ={after-before})")

    return refined


# 5. Unified entry point (pipeline hook)

def refine_clusters_core_consistency(
    X,
    cluster_labels,
    config=None
):
    """
    Core-consistency-based refinement.
    NO label usage.
    """
    config = config or {}

    inconsistent = detect_core_inconsistent_samples(
        X,
        cluster_labels,
        min_cluster_size=config.get("min_cluster_size", 10),
        consistency_threshold=config.get("consistency_threshold", 0.6),
        tolerance=config.get("tolerance", 0.5),
    )

    refined = conservative_core_relabel(
        cluster_labels,
        inconsistent,
        max_inconsistent_ratio=config.get("max_inconsistent_ratio", 0.3),
    )

    return refined
