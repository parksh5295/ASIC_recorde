import logging
import os
from collections import defaultdict
from tqdm import tqdm
from Signature_tool.performance import get_index_set


logger = logging.getLogger(__name__)

# Cap to prevent huge TP index sets from exhausting memory during subsumption.
# Adjustable via env SUBSUMPTION_TP_CAP (default: 200_000).
TP_CAP = int(os.environ.get("SUBSUMPTION_TP_CAP", "200000"))


def _cap_set(idx_set):
    """Cap the size of an index set to TP_CAP to avoid memory blowups."""
    if idx_set is None:
        return set()
    if len(idx_set) <= TP_CAP:
        return set(idx_set)
    # deterministic slice after sort to keep reproducibility
    return set(sorted(idx_set)[:TP_CAP])

def find_redundant_signatures(signatures: dict, performance_results: dict, coverage_threshold: float = 0.95):
    if not signatures or not performance_results:
        return set()

    # --- FIX-1: Build the index from ALL signatures to ensure superset search is complete. ---
    logger.info("[Subsumption] Building inverted index from all signatures for faster superset search...")
    sig_map = {sig_id: {'items': frozenset(rule.items()), 'len': len(rule)} for sig_id, rule in signatures.items()}
    inverted_index = defaultdict(set)
    for sig_id, sig_data in tqdm(sig_map.items(), desc="[Subsumption] Indexing all signatures"):
        for item in sig_data['items']:
            inverted_index[item].add(sig_id)

    # --- KEEP: Iterate only through signatures that had TPs in the current data chunk. ---
    logger.info("[Subsumption] Pre-processing signatures with TPs for redundancy analysis...")
    processed_sigs = []
    for sig_id, rule in signatures.items():
        perf_entry = performance_results.get(sig_id)
        if not perf_entry:
            continue
        #tp_set = get_index_set(perf_entry, 'tp_indices')
        tp_set = _cap_set(get_index_set(perf_entry, 'tp_indices'))
        if tp_set:
            processed_sigs.append({
                'id': sig_id,
                'items': frozenset(rule.items()),
                'len': len(rule),
                'tp_indices': tp_set
            })

    # Sort by length (general to specific). This is good practice.
    processed_sigs.sort(key=lambda x: x['len'])

    '''
    # --- Build inverted index and sig_map for fast lookups --- # move to FIX-1
    logger.info("[Subsumption] Building inverted index for faster superset search...")
    sig_map = {sig['id']: sig for sig in processed_sigs}
    inverted_index = defaultdict(set)
    for sig in tqdm(processed_sigs, desc="[Subsumption] Indexing signatures"):
        for item in sig['items']:
            inverted_index[item].add(sig['id'])
    '''

    pruned_ids = set()
    logger.info(f"[Subsumption] Starting redundancy analysis on {len(processed_sigs)} signatures...")

    for general_rule in tqdm(processed_sigs, desc="[Subsumption] Finding redundant rules"):
        if general_rule['id'] in pruned_ids:
            continue
            
        general_tps = general_rule['tp_indices']
        if not general_tps:
            continue
            
        general_items = general_rule['items']
        if not general_items:
            continue

        # --- USE THE COMPLETE INDEX FOR THE SEARCH ---
        try:
            min_len_item = min(general_items, key=lambda item: len(inverted_index[item]))
            candidate_ids = set(inverted_index[min_len_item]) # Make a copy
        except (ValueError, KeyError):
            # This can happen if an item in a rule somehow wasn't indexed (e.g., empty general_items)
            continue

        for item in general_items:
            if item == min_len_item:
                continue
            candidate_ids.intersection_update(inverted_index[item])
            
        subsuming_specific_rules = []
        # potential_supersets = 0 # DEBUG
        for specific_id in candidate_ids:
            if specific_id == general_rule['id']:
                continue
            
            specific_rule_data = sig_map[specific_id]
            if specific_rule_data['len'] > general_rule['len']:
                subsuming_specific_rules.append({'id': specific_id, **specific_rule_data})

        if not subsuming_specific_rules:
            continue
            
        combined_tps_from_specific_rules = set()
        for specific_rule in subsuming_specific_rules:
            specific_rule_perf = performance_results.get(specific_rule['id'])
            if specific_rule_perf:
                combined_tps_from_specific_rules.update(
                    #get_index_set(specific_rule_perf, 'tp_indices')
                    _cap_set(get_index_set(specific_rule_perf, 'tp_indices'))
                )
                if len(combined_tps_from_specific_rules) > TP_CAP:
                    # Keep combined set within cap to avoid memory spikes
                    combined_tps_from_specific_rules = _cap_set(combined_tps_from_specific_rules)

        if not combined_tps_from_specific_rules:
            continue
            
        # general_tps = general_rule['tp_indices'] # Moved down to avoid UnboundLocalError if continue happens
        covered_tps = general_tps.intersection(combined_tps_from_specific_rules)
        coverage_ratio = len(covered_tps) / len(general_tps)
        
        if coverage_ratio >= coverage_threshold:
            pruned_ids.add(general_rule['id'])

    logger.info(f"[Subsumption] Redundancy analysis complete. Found {len(pruned_ids)} redundant signatures to prune.")
    return pruned_ids
