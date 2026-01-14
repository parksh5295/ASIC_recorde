import numpy as np
from sklearn.metrics import pairwise_distances


# 1. Seed extraction

def extract_attack_seeds(cluster_labels, attack_label=1):
    """
    Attack seeds are samples already classified as attack.
    """
    return np.where(cluster_labels == attack_label)[0]


# 2. Benign core estimation (for safety)

def compute_centroid(X):
    return np.mean(X, axis=0)


def benign_core_distance_threshold(
    X,
    cluster_labels,
    benign_label=0,
    core_percentile=40,
):
    """
    Compute distance threshold defining benign core.
    Used to prevent flipping benign-central samples.
    """

    idx = np.where(cluster_labels == benign_label)[0]
    if len(idx) < 5:
        return None

    Xb = X[idx]
    center = compute_centroid(Xb)
    dists = np.linalg.norm(Xb - center, axis=1)

    return np.percentile(dists, core_percentile)


# 3. Local attack affinity computation

def attack_affinity_knn(
    X,
    benign_indices,
    attack_indices,
    k=10,
):
    """
    For each benign sample, compute the ratio of attack samples
    among its k nearest neighbors.
    """

    if len(attack_indices) == 0 or len(benign_indices) == 0:
        return {}

    X_b = X[benign_indices]
    X_a = X[attack_indices]

    D = pairwise_distances(X_b, X_a)

    scores = {}
    for i, row in enumerate(D):
        nn = np.argsort(row)[:k]
        scores[benign_indices[i]] = len(nn) / k

    return scores


# 4. Attack Affinity Propagation (AAP)

def attack_affinity_propagation(
    X,
    cluster_labels,
    attack_label=1,
    benign_label=0,
    k=10,
    affinity_threshold=0.7,
    benign_core_percentile=40,
    verbose=True,
):
    """
    Label-leakage-free attack propagation.

    Rules:
    - Only 0 → 1 flips allowed
    - Seeds are cluster==1 only
    - Local neighborhood affinity required
    - Benign core samples are protected
    """

    refined = cluster_labels.copy()

    # Step 1: extract attack seeds
    attack_seeds = extract_attack_seeds(cluster_labels, attack_label)

    if verbose:
        print(f"[AAP] Attack seeds: {len(attack_seeds)}")

    if len(attack_seeds) == 0:
        return refined

    # Step 2: benign candidates
    benign_idx = np.where(cluster_labels == benign_label)[0]
    if len(benign_idx) == 0:
        return refined

    # Step 3: benign core protection
    benign_core_thr = benign_core_distance_threshold(
        X,
        cluster_labels,
        benign_label=benign_label,
        core_percentile=benign_core_percentile,
    )

    if benign_core_thr is None:
        return refined

    benign_center = compute_centroid(X[benign_idx])
    benign_dists = {
        i: np.linalg.norm(X[i] - benign_center)
        for i in benign_idx
    }

    # Step 4: affinity scoring
    affinity = attack_affinity_knn(
        X,
        benign_idx,
        attack_seeds,
        k=k,
    )

    flipped = 0

    for i, score in affinity.items():
        if refined[i] != benign_label:
            continue

        # Core protection
        if benign_dists[i] <= benign_core_thr:
            continue

        # Attack affinity condition
        if score >= affinity_threshold:
            refined[i] = attack_label
            flipped += 1

    if verbose:
        before = np.sum(cluster_labels == attack_label)
        after = np.sum(refined == attack_label)
        print(
            f"[AAP] Attack count: {before} → {after} "
            f"(Δ={after-before}, flipped={flipped})"
        )

    return refined


# 5. Pipeline-friendly entry

def apply_attack_affinity_propagation(
    X,
    cluster_labels,
    enabled=True,
    config=None,
):
    """
    Safe pipeline entry point.
    """

    if not enabled:
        return cluster_labels

    config = config or {}

    return attack_affinity_propagation(
        X,
        cluster_labels,
        attack_label=config.get("attack_label", 1),
        benign_label=config.get("benign_label", 0),
        k=config.get("k", 10),
        affinity_threshold=config.get("affinity_threshold", 0.7),
        benign_core_percentile=config.get("benign_core_percentile", 40),
        verbose=config.get("verbose", True),
    )
