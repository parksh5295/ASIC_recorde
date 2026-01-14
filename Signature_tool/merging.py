import logging
from collections import defaultdict
from tqdm import tqdm
from Signature_tool.performance import get_index_set


logger = logging.getLogger(__name__)

def merge_similar_signatures(
    signatures: dict, 
    performance_results: dict, 
    infrequent_threshold: int
):
    """
    Finds and merges similar, infrequent signatures.

    Two rules are considered "similar" if they share all but one feature-value pair.
    They are "infrequent" if their individual True Positive count is at or below the threshold.

    Merged rules will have a frozenset for the value of the differing feature, representing an OR condition.

    Args:
        signatures (dict): The current set of signatures {sig_id: rule_dict}.
        performance_results (dict): A dict mapping sig_id to {'tp_indices': set, ...}.
        infrequent_threshold (int): The TP count at or below which a rule is considered infrequent.

    Returns:
        tuple: (
            newly_merged_rules (dict): {new_sig_id: merged_rule_dict},
            ids_to_remove (set): A set of original signature IDs that were merged.
        )
    """
    if not signatures or infrequent_threshold is None:
        return {}, set()

    logger.info(f"[Merger] Starting signature merging analysis for {len(signatures)} signatures (TP threshold <= {infrequent_threshold}).")

    # 1. Group rules by their "base" (a rule with one item removed)
    # The key is a frozenset of items, the value is a list of what was removed.
    groups = defaultdict(list)
    for sig_id, rule in signatures.items():
        if len(rule) < 2:
            continue  # Rules with only one item cannot be merged in this logic

        rule_items = frozenset(rule.items())
        for item in rule_items:
            base = rule_items - {item}
            diff_key, diff_value = item
            groups[base].append({
                'id': sig_id,
                'diff_key': diff_key,
                'diff_value': diff_value
            })

    newly_merged_rules = {}
    ids_to_remove = set()

    # 2. Iterate through groups to find merge candidates
    # --- NEW: Add tqdm for progress bar ---
    for base, candidates in tqdm(groups.items(), desc="[Merger] Finding merge candidates"):
        if len(candidates) < 2:
            continue

        # Sub-group the candidates by the feature on which they differ
        sub_groups_by_diff_key = defaultdict(list)
        for cand in candidates:
            sub_groups_by_diff_key[cand['diff_key']].append(cand)

        for diff_key, merge_list in sub_groups_by_diff_key.items():
            if len(merge_list) < 2:
                continue
            
            # 3. Check if ALL rules in the potential merge list are infrequent
            are_all_infrequent = True
            for rule_info in merge_list:
                perf = performance_results.get(rule_info['id'])
                if not perf:  # Rule has no performance data (e.g., 0 TP and 0 FP)
                    tp_count = 0
                else:
                    tp_count = len(get_index_set(perf, 'tp_indices'))
                
                if tp_count > infrequent_threshold:
                    are_all_infrequent = False
                    break
            
            if not are_all_infrequent:
                continue

            # 4. If all are infrequent, create the merged rule
            base_dict = dict(base)
            or_values = frozenset({r['diff_value'] for r in merge_list})
            
            merged_rule = base_dict.copy()
            merged_rule[diff_key] = or_values
            
            # A hashable representation of the merged rule for the ID
            hashable_merged_rule = frozenset(merged_rule.items())
            merged_rule_id = hash(hashable_merged_rule)
            
            # Avoid adding a merged rule that already exists or was just created
            if merged_rule_id not in signatures and merged_rule_id not in newly_merged_rules:
                newly_merged_rules[merged_rule_id] = merged_rule
                # logger.debug(f"  - Creating new merged rule for base {dict(base)} on key '{diff_key}'")

            # Mark all original rules that formed this merge for removal
            for rule_info in merge_list:
                ids_to_remove.add(rule_info['id'])

    logger.info(f"[Merger] Merging analysis complete. Created {len(newly_merged_rules)} new rules and identified {len(ids_to_remove)} original rules to remove.")
    return newly_merged_rules, ids_to_remove
