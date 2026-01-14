import numpy as np
from collections import defaultdict
from sklearn.metrics import pairwise_distances


# 1. Core extraction utilities

def compute_centroid(X):
    return np.mean(X, axis=0)


def extract_cluster_core(
    X,
    cluster_labels,
    target_label=1,
    core_percentile=40
):
    """
    Extract core samples from a target cluster (e.g., attack cluster)
    based purely on intra-cluster distance.
    
    Parameters
    ----------
    X : ndarray (N, D)
    cluster_labels : ndarray (N,)
    target_label : int
        Cluster label considered as attack (default=1)
    core_percentile : float
        Lower percentile → tighter core
    
    Returns
    -------
    core_indices : set
        Indices of core samples
    """

    idx = np.where(cluster_labels == target_label)[0]
    if len(idx) < 5:
        return set()

    Xc = X[idx]
    center = compute_centroid(Xc)
    dists = np.linalg.norm(Xc - center, axis=1)

    thr = np.percentile(dists, core_percentile)
    core_indices = set(idx[dists <= thr])

    return core_indices


# 2. Propagation score computation

def propagation_score_knn(
    X,
    benign_indices,
    attack_core_indices,
    k=10
):
    """
    Propagation score based on kNN overlap with attack core.
    
    score(x) = (# of neighbors in attack core) / k
    """

    if len(attack_core_indices) == 0:
        return {}

    attack_core_indices = list(attack_core_indices)
    benign_indices = list(benign_indices)

    X_benign = X[benign_indices]
    X_attack = X[attack_core_indices]

    D = pairwise_distances(X_benign, X_attack)

    scores = {}
    for i, row in enumerate(D):
        nn = np.argsort(row)[:k]
        scores[benign_indices[i]] = len(nn) / k

    return scores


def propagation_score_distance(
    X,
    benign_indices,
    attack_core_indices
):
    """
    Propagation score based on minimum distance to attack core.
    """

    if len(attack_core_indices) == 0:
        return {}

    X_benign = X[list(benign_indices)]
    X_attack = X[list(attack_core_indices)]

    D = pairwise_distances(X_benign, X_attack)
    min_dists = D.min(axis=1)

    scores = {}
    for idx, d in zip(benign_indices, min_dists):
        scores[idx] = d

    return scores


# 3. Benign cluster edge detection

def benign_edge_samples(
    X,
    cluster_labels,
    benign_label=0,
    edge_percentile=70
):
    """
    Identify edge (non-core) samples inside benign clusters.
    """

    idx = np.where(cluster_labels == benign_label)[0]
    if len(idx) < 5:
        return set()

    Xb = X[idx]
    center = compute_centroid(Xb)
    dists = np.linalg.norm(Xb - center, axis=1)

    thr = np.percentile(dists, edge_percentile)
    edge_indices = set(idx[dists >= thr])

    return edge_indices


# 4. Core-consistency refinement logic

def cluster_core_consistency_refine(
    X,
    cluster_labels,
    attack_label=1,
    benign_label=0,
    core_percentile=40,
    edge_percentile=70,
    k=10,
    propagation_threshold=0.3,
    mode="knn",
    verbose=True
):
    """
    Propagation-aware refinement without label leakage.

    Rules:
    - Only benign → attack flip allowed
    - Only benign edge samples are considered
    - Flip only if strongly connected to attack core
    """

    refined = cluster_labels.copy()

    # Step 1: extract attack core
    attack_core = extract_cluster_core(
        X,
        cluster_labels,
        target_label=attack_label,
        core_percentile=core_percentile,
    )

    if verbose:
        print(f"[CCC] Attack core size: {len(attack_core)}")

    if len(attack_core) == 0:
        return refined

    # Step 2: identify benign edge samples
    benign_idx = np.where(cluster_labels == benign_label)[0]
    benign_edge = benign_edge_samples(
        X,
        cluster_labels,
        benign_label=benign_label,
        edge_percentile=edge_percentile,
    )

    if verbose:
        print(f"[CCC] Benign edge candidates: {len(benign_edge)}")

    if len(benign_edge) == 0:
        return refined

    # Step 3: propagation scoring
    if mode == "knn":
        scores = propagation_score_knn(
            X,
            benign_edge,
            attack_core,
            k=k,
        )
        flip_condition = lambda s: s >= propagation_threshold
    else:
        scores = propagation_score_distance(
            X,
            benign_edge,
            attack_core,
        )
        dist_thr = np.percentile(list(scores.values()), 30)
        flip_condition = lambda s: s <= dist_thr

    # Step 4: conservative flip
    flipped = 0
    for i, score in scores.items():
        if refined[i] == benign_label and flip_condition(score):
            refined[i] = attack_label
            flipped += 1

    if verbose:
        before = np.sum(cluster_labels == attack_label)
        after = np.sum(refined == attack_label)
        print(
            f"[CCC] Attack count: {before} → {after} (Δ={after-before}, flipped={flipped})"
        )

    return refined


# 5. Unified entry (pipeline-friendly)

def apply_cluster_core_consistency(
    X,
    cluster_labels,
    enabled=True,
    config=None
):
    """
    Pipeline entry point.
    Safe to call even when disabled.
    """

    if not enabled:
        return cluster_labels

    config = config or {}

    return cluster_core_consistency_refine(
        X,
        cluster_labels,
        attack_label=config.get("attack_label", 1),
        benign_label=config.get("benign_label", 0),
        core_percentile=config.get("core_percentile", 40),
        edge_percentile=config.get("edge_percentile", 70),
        k=config.get("k", 10),
        propagation_threshold=config.get("propagation_threshold", 0.3),
        mode=config.get("mode", "knn"),
        verbose=config.get("verbose", True),
    )
