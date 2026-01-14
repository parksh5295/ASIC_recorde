import numpy as np
from sklearn.metrics import pairwise_distances


# =========================================================
# 1. Attack core extraction (NO label leakage)
# =========================================================

def extract_attack_core(
    X,
    cluster_labels,
    attack_label=1,
    core_percentile=30,
):
    """
    Extract compact attack core samples
    using only intra-cluster structure.

    NO ground-truth label usage.
    """

    idx = np.where(cluster_labels == attack_label)[0]
    if len(idx) < 5:
        return set()

    Xc = X[idx]
    center = Xc.mean(axis=0)
    dists = np.linalg.norm(Xc - center, axis=1)

    thr = np.percentile(dists, core_percentile)
    return set(idx[dists <= thr])


# =========================================================
# 2. FN-oriented soft attack affinity scoring
# =========================================================

def compute_attack_affinity_scores(
    X,
    cluster_labels,
    attack_label=1,
    benign_label=0,
    k=10,
):
    """
    Compute attack affinity ONLY for benign-labeled samples.

    score(x) = (# of kNN in attack core) / k

    Returned scores are soft evidence, not decisions.
    """

    attack_core = extract_attack_core(
        X,
        cluster_labels,
        attack_label=attack_label,
    )

    if len(attack_core) == 0:
        return {}

    benign_idx = np.where(cluster_labels == benign_label)[0]
    if len(benign_idx) == 0:
        return {}

    X_b = X[benign_idx]
    X_a = X[list(attack_core)]

    D = pairwise_distances(X_b, X_a)

    soft_scores = {}
    for i, row in enumerate(D):
        nn = np.argsort(row)[:k]
        soft_scores[benign_idx[i]] = len(nn) / k

    return soft_scores


# =========================================================
# 3. Optional local density guard (stability check)
# =========================================================

def local_density(
    X,
    idx,
    radius_percentile=10,
):
    """
    Estimate local density via inverse mean distance.
    Lower density = more unstable = safer to flip.
    """

    x = X[idx].reshape(1, -1)
    D = pairwise_distances(x, X)[0]

    thr = np.percentile(D, radius_percentile)
    neighbors = D[D <= thr]

    if len(neighbors) == 0:
        return 0.0

    return 1.0 / (neighbors.mean() + 1e-6)


# =========================================================
# 4. FN-only deferred flip gate
# =========================================================

def fn_only_flip_gate(
    X,
    cluster_labels,
    soft_scores,
    attack_label=1,
    benign_label=0,
    soft_threshold=0.7,
    density_threshold=None,
    max_flips_ratio=0.02,
    verbose=True,
):
    """
    FN-only flip gate.

    Rules:
    - ONLY benign -> attack flips allowed
    - attack -> benign is strictly forbidden
    - extremely conservative by default
    """

    refined = cluster_labels.copy()
    candidates = []

    for idx, score in soft_scores.items():
        if score < soft_threshold:
            continue

        # Optional density guard
        if density_threshold is not None:
            density = local_density(X, idx)
            if density > density_threshold:
                continue

        if refined[idx] == benign_label:
            candidates.append(idx)

    # Absolute safety cap
    max_flips = max(1, int(len(cluster_labels) * max_flips_ratio))
    candidates = candidates[:max_flips]

    for idx in candidates:
        refined[idx] = attack_label

    if verbose:
        before = np.sum(cluster_labels == attack_label)
        after = np.sum(refined == attack_label)
        print(
            f"[FN-AAP] flips={len(candidates)}, "
            f"attack {before} → {after} (Δ={after-before})"
        )

    return refined, candidates


# =========================================================
# 5. Unified pipeline entry (FN recovery ONLY)
# =========================================================

def apply_fn_attack_affinity_recovery(
    X,
    cluster_labels,
    enabled=True,
    config=None,
):
    """
    Pipeline-safe FN recovery module.

    Returns:
    - refined_labels (monotonic: attack count never decreases)
    - soft_scores (sample-level evidence)
    - flipped_indices (FN recovered samples)
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

    refined_labels, flipped = fn_only_flip_gate(
        X,
        cluster_labels,
        soft_scores,
        attack_label=config.get("attack_label", 1),
        benign_label=config.get("benign_label", 0),
        soft_threshold=config.get("soft_threshold", 0.7),
        density_threshold=config.get("density_threshold", None),
        max_flips_ratio=config.get("max_flips_ratio", 0.02),
        verbose=config.get("verbose", True),
    )

    return refined_labels, soft_scores, flipped
