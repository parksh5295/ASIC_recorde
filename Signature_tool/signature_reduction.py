"""
Signature reduction utilities for ISV.
1. Remove signatures that never triggered alerts (inactive signatures)
2. Minimize signatures by removing supersets when subsets exist (reduction)
"""
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def reduce_signatures_by_subsets(signatures: dict):
    """
    Reduce signatures by removing supersets when subsets exist.
    
    Example: If {a:1, b:3, c:6} and {a:1, c:6} both exist,
    keep only {a:1, c:6} and remove {a:1, b:3, c:6}.
    
    Args:
        signatures: Dict of {sig_id: rule_dict}
    
    Returns:
        Dict of reduced signatures (supersets removed)
    """
    if not signatures:
        return {}
    
    # Convert to list of (id, itemset, length) for sorting
    sig_list = []
    for sig_id, rule in signatures.items():
        itemset = frozenset(rule.items())
        sig_list.append({
            'id': sig_id,
            'rule': rule,
            'itemset': itemset,
            'len': len(rule)
        })
    
    # Sort by length (smallest first) - we want to keep smaller rules
    sig_list.sort(key=lambda x: x['len'])
    
    # Build inverted index for fast lookup
    inverted_index = defaultdict(set)
    for sig_data in sig_list:
        for item in sig_data['itemset']:
            inverted_index[item].add(sig_data['id'])
    
    # Find signatures to remove (supersets that have subsets)
    to_remove = set()
    
    logger.info(f"[Reduction] Starting signature reduction on {len(sig_list)} signatures...")
    
    # Check each signature to see if it's a superset of any smaller signature
    for i, sig_data in enumerate(sig_list):
        if sig_data['id'] in to_remove:
            continue
        
        # Find candidate signatures that could be subsets
        # Candidates must contain at least one item from this signature
        candidates = set()
        for item in sig_data['itemset']:
            candidates.update(inverted_index[item])
        
        # Remove self and already-marked-for-removal signatures
        candidates.discard(sig_data['id'])
        candidates -= to_remove
        
        # Check if any candidate is a subset of this signature
        for candidate_id in candidates:
            candidate_data = next((s for s in sig_list if s['id'] == candidate_id), None)
            if candidate_data is None:
                continue
            
            # Skip if candidate is larger (we're looking for subsets, not supersets)
            if candidate_data['len'] >= sig_data['len']:
                continue
            
            # Check if candidate is a subset
            if candidate_data['itemset'].issubset(sig_data['itemset']):
                # Current signature is a superset, mark it for removal
                to_remove.add(sig_data['id'])
                logger.debug(f"[Reduction] Removing superset {sig_data['id']}: {dict(sig_data['rule'])} "
                           f"(subset found: {dict(candidate_data['rule'])})")
                break
    
    # Return reduced signatures
    reduced = {sig_id: rule for sig_id, rule in signatures.items() if sig_id not in to_remove}
    
    removed_count = len(signatures) - len(reduced)
    if removed_count > 0:
        logger.info(f"[Reduction] Reduced {len(signatures)} signatures to {len(reduced)} "
                   f"(removed {removed_count} supersets)")
    else:
        logger.info(f"[Reduction] No reduction needed: {len(signatures)} signatures remain")
    
    return reduced


def identify_inactive_signatures(signatures: dict, alert_results):
    """
    Identify signatures that never triggered any alerts.
    
    Args:
        signatures: Dict of {sig_id: rule_dict}
        alert_results: DataFrame or dict containing alert information with 'signature_id' column
    
    Returns:
        Set of signature IDs that never triggered alerts
    """
    if not signatures or alert_results is None or len(alert_results) == 0:
        # If no alerts, all signatures are inactive
        return set(signatures.keys()) if signatures else set()
    
    # Extract signature IDs that triggered alerts
    if hasattr(alert_results, 'columns') and 'signature_id' in alert_results.columns:
        # DataFrame case
        active_sig_ids = set(alert_results['signature_id'].dropna().unique())
    elif isinstance(alert_results, dict) and 'signature_id' in alert_results:
        # Dict case
        active_sig_ids = set(alert_results['signature_id'])
    elif isinstance(alert_results, (list, tuple)):
        # List of dicts or DataFrames
        active_sig_ids = set()
        for item in alert_results:
            if hasattr(item, 'columns') and 'signature_id' in item.columns:
                active_sig_ids.update(item['signature_id'].dropna().unique())
            elif isinstance(item, dict) and 'signature_id' in item:
                active_sig_ids.add(item['signature_id'])
    else:
        # Unknown format, assume all are active
        logger.warning("[InactiveRemoval] Unknown alert_results format, assuming all signatures are active")
        return set()
    
    # Find inactive signatures (signatures not in active_sig_ids)
    all_sig_ids = set(signatures.keys())
    inactive_sig_ids = all_sig_ids - active_sig_ids
    
    if inactive_sig_ids:
        logger.info(f"[InactiveRemoval] Found {len(inactive_sig_ids)} inactive signatures "
                   f"(out of {len(all_sig_ids)} total)")
    
    return inactive_sig_ids


def remove_inactive_signatures(signatures: dict, alert_results):
    """
    Remove signatures that never triggered any alerts.
    
    Args:
        signatures: Dict of {sig_id: rule_dict}
        alert_results: DataFrame or dict containing alert information with 'signature_id' column
    
    Returns:
        Tuple of (cleaned_signatures_dict, removed_count)
    """
    inactive_ids = identify_inactive_signatures(signatures, alert_results)
    
    if not inactive_ids:
        return signatures.copy(), 0
    
    cleaned = {sig_id: rule for sig_id, rule in signatures.items() if sig_id not in inactive_ids}
    return cleaned, len(inactive_ids)

