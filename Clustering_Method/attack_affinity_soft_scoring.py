import numpy as np
from sklearn.metrics import pairwise_distances
from collections import defaultdict


# 1. Attack core extraction (NO label leakage)

def extract_attack_core(
    X,
    cluster_labels,
    attack_label=1,
    core_percentile=30
):
    """
    Extract attack core samples using intra-cluster compactness.
    """
    idx = np.where(cluster_labels == attack_label)[0]
    if len(idx) < 5:
        return set()

    Xc = X[idx]
    center = Xc.mean(axis=0)
    dists = np.linalg.norm(Xc - center, axis=1)

    thr = np.percentile(dists, core_percentile)
    return set(idx[dists <= thr])


# 2. Sample-level soft attack affinity score

def compute_attack_affinity_scores(
    X,
    cluster_labels,
    attack_label=1,
    benign_label=0,
    k=10,
):
    """
    Compute soft attack affinity score for benign-labeled samples.

    score(x) = (# of kNN in attack core) / k
    """

    attack_core = extract_attack_core(
        X,
        cluster_labels,
        attack_label=attack_label,
    )

    if len(attack_core) == 0:
        return {}

    benign_idx = np.where(cluster_labels == benign_label)[0]

    X_b = X[benign_idx]
    X_a = X[list(attack_core)]

    D = pairwise_distances(X_b, X_a)

    soft_scores = {}
    for i, row in enumerate(D):
        nn = np.argsort(row)[:k]
        soft_scores[benign_idx[i]] = len(nn) / k

    return soft_scores


# 3. Local density estimation (stability check)

def local_density(
    X,
    idx,
    radius_percentile=10
):
    """
    Estimate local density by inverse average distance
    """
    x = X[idx].reshape(1, -1)
    D = pairwise_distances(x, X)[0]
    thr = np.percentile(D, radius_percentile)
    neighbors = D[D <= thr]

    if len(neighbors) == 0:
        return 0.0

    return 1.0 / (neighbors.mean() + 1e-6)


# 4. Deferred flip gate (ULTRA conservative)

def deferred_flip_gate(
    X,
    cluster_labels,
    soft_scores,
    attack_label=1,
    benign_label=0,
    soft_threshold=0.99,
    density_threshold=0.05,
    max_flips_ratio=0.01,
    verbose=True,
):
    """
    Convert soft scores to hard flips only when extremely confident.
    """

    refined = cluster_labels.copy()
    candidates = []

    for idx, score in soft_scores.items():
        if score >= soft_threshold:
            density = local_density(X, idx)
            if density <= density_threshold:
                candidates.append(idx)

    # Absolute safety cap
    max_flips = max(1, int(len(cluster_labels) * max_flips_ratio))
    candidates = candidates[:max_flips]

    for idx in candidates:
        if refined[idx] == benign_label:
            refined[idx] = attack_label

    if verbose:
        before = np.sum(cluster_labels == attack_label)
        after = np.sum(refined == attack_label)
        print(
            f"[AAP-SoftGate] flips={len(candidates)}, "
            f"attack {before} → {after} (Δ={after-before})"
        )

    return refined, candidates


# 5. Unified pipeline entry (SAFE DEFAULT)

def apply_attack_affinity_soft_scoring(
    X,
    cluster_labels,
    enabled=True,
    config=None,
):
    """
    Pipeline entry point.

    Returns:
    - refined_labels (mostly unchanged)
    - soft_scores (for downstream signature / alert logic)
    - flipped_indices (usually empty)
    """

    if not enabled:
        return cluster_labels, {}, []

    config = config or {}

    soft_scores = compute_attack_affinity_scores(
        X,
        cluster_labels,
        attack_label=config.get("attack_label", 1),
        benign_label=config.get("benign_label", 0),
        k=config.get("k", 10),
    )

    refined_labels, flipped = deferred_flip_gate(
        X,
        cluster_labels,
        soft_scores,
        attack_label=config.get("attack_label", 1),
        benign_label=config.get("benign_label", 0),
        soft_threshold=config.get("soft_threshold", 0.99),
        density_threshold=config.get("density_threshold", 0.05),
        max_flips_ratio=config.get("max_flips_ratio", 0.01),
        verbose=config.get("verbose", True),
    )

    return refined_labels, soft_scores, flipped
