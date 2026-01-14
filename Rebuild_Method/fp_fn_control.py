# fp_fn_control.py

import random
import hashlib


# Utility: stable signature ID (reproducible across runs)
def stable_rule_id(rule: dict) -> str:
    """
    Generate a stable, reproducible rule ID from a rule dictionary.
    """
    return hashlib.md5(
        str(sorted(rule.items())).encode()
    ).hexdigest()


# FP CONTROL
def inject_fp_signatures(
    all_valid_signatures: dict,
    normal_data,
    k: int = 30,
    max_cols: int = 4
):
    """
    Inject fake FP-causing signatures derived from normal data.

    Parameters
    ----------
    all_valid_signatures : dict
        Global signature dictionary (rule_id -> rule)
    normal_data : pd.DataFrame
        Normal-only data for FP construction
    k : int
        Number of FP signatures to inject
    max_cols : int
        Number of columns to use for FP rule construction

    Returns
    -------
    injected_ids : set
        Set of injected FP signature IDs
    """

    if normal_data is None or len(normal_data) == 0:
        return set()

    injected = {}
    sampled = normal_data.sample(min(k, len(normal_data)))

    cols = list(normal_data.columns)[:max_cols]

    for _, row in sampled.iterrows():
        rule = {col: row[col] for col in cols}
        sig_id = stable_rule_id(rule)

        # Avoid overwriting existing signatures
        if sig_id in all_valid_signatures:
            continue

        injected[sig_id] = rule

    all_valid_signatures.update(injected)
    return set(injected.keys())


# FN CONTROL
def remove_fn_signatures(
    all_valid_signatures: dict,
    signature_turn_created: dict,
    strategy: str = "oldest",
    remove_ratio: float = 0.2
):
    """
    Remove important signatures to simulate FN (knowledge loss).

    Parameters
    ----------
    all_valid_signatures : dict
        Global signature dictionary (rule_id -> rule)
    signature_turn_created : dict
        Mapping rule_id -> creation turn
    strategy : str
        Removal strategy: ['oldest', 'newest', 'random']
    remove_ratio : float
        Fraction of signatures to remove (0 < remove_ratio <= 1)

    Returns
    -------
    removed_ids : set
        Set of removed signature IDs
    """

    if not all_valid_signatures:
        return set()

    rule_ids = list(all_valid_signatures.keys())

    # Select removal candidates
    if strategy == "oldest":
        rule_ids.sort(key=lambda rid: signature_turn_created.get(rid, 0))
    elif strategy == "newest":
        rule_ids.sort(
            key=lambda rid: signature_turn_created.get(rid, 0),
            reverse=True
        )
    elif strategy == "random":
        random.shuffle(rule_ids)
    else:
        raise ValueError(f"Unknown FN removal strategy: {strategy}")

    k = max(1, int(len(rule_ids) * remove_ratio))
    removed_ids = set(rule_ids[:k])

    # Remove signatures completely
    for rid in removed_ids:
        all_valid_signatures.pop(rid, None)
        signature_turn_created.pop(rid, None)

    return removed_ids


# FN RECOVERY CHECK
def check_fn_recovery(
    fn_removed_history: dict,
    all_valid_signatures: dict,
    current_turn: int,
    max_lookback: int = None
):
    """
    Check whether FN-removed signatures have been recovered.

    Parameters
    ----------
    fn_removed_history : dict
        Mapping turn -> set(removed_signature_ids)
    all_valid_signatures : dict
        Current global signature dictionary
    current_turn : int
        Current turn number
    max_lookback : int or None
        Maximum number of turns to look back (None = unlimited)

    Returns
    -------
    recovered : dict
        Mapping:
            removed_turn -> {
                'recovered_ids': set,
                'recovery_latency': dict(sig_id -> latency)
            }
    """

    recovered = {}

    for removed_turn, removed_ids in fn_removed_history.items():
        # Skip future or same turn
        if removed_turn >= current_turn:
            continue

        # Lookback window control
        if max_lookback is not None:
            if current_turn - removed_turn > max_lookback:
                continue

        recovered_ids = removed_ids & set(all_valid_signatures.keys())

        if not recovered_ids:
            continue

        recovery_latency = {
            rid: current_turn - removed_turn
            for rid in recovered_ids
        }

        recovered[removed_turn] = {
            "recovered_ids": recovered_ids,
            "recovery_latency": recovery_latency
        }

    return recovered


def summarize_fn_recovery(fn_removed_history, recovered_info):
    total_removed = sum(len(v) for v in fn_removed_history.values())
    total_recovered = sum(len(info["recovered_ids"]) for info in recovered_info.values())

    latencies = []
    for info in recovered_info.values():
        latencies.extend(info["recovery_latency"].values())

    return {
        "fn_total_removed": total_removed,
        "fn_total_recovered": total_recovered,
        "fn_recovery_rate": total_recovered / max(1, total_removed),
        "fn_avg_latency": sum(latencies) / max(1, len(latencies))
    }
