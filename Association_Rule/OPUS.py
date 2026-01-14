# Algorithm: OPUS (Optimal Pattern Discovery)
# Output: Association rules Dictionary List; [{'feature1': value1, 'feature2': value2, ...}, {...}, ...]

from collections import defaultdict
from itertools import combinations
import pandas as pd # For pd.Timestamp
import multiprocessing
import logging

# Module Imports
# --- Import for saving artifacts ---
from Dataset_Choose_Rule.isv_save_log import save_association_artifacts


# Helper function for parallel support calculation in OPUS prune_candidates
# Needs to be defined at the top level or be picklable by multiprocessing
def calculate_support_for_opus_candidate(item_tids, transaction_count, support_cache_for_worker, candidate_itemset):
    # support_cache_for_worker is a read-only copy for this worker, or could be managed if writable
    if not candidate_itemset:
        return 0, candidate_itemset

    items_key = frozenset(candidate_itemset) # Ensure consistent key type
    if items_key in support_cache_for_worker:
        return support_cache_for_worker[items_key], candidate_itemset
    
    if not all(item in item_tids for item in candidate_itemset):
        # print(f"Warning: Item in {candidate_itemset} not found in TIDs for OPUS. Skipping.")
        return 0, candidate_itemset
        
    common_tids = set.intersection(*(item_tids[item] for item in candidate_itemset))
    support = len(common_tids) / transaction_count if transaction_count > 0 else 0
    
    # Note: Worker does not update the shared cache here to avoid complexity.
    # The main process can update its cache if needed based on results, but for pruning, only support value is critical.
    return support, candidate_itemset

# Helper function for parallel rule generation for OPUS (similar to RARM's task)
# Takes a single frequent itemset and checks if any rule derived from it meets min_confidence.
# Returns the original frequent itemset if a strong rule is found, otherwise None.
def generate_opus_rules_for_itemset_task(f_itemset, min_conf, opus_miner_item_tids, opus_miner_transaction_count, opus_miner_get_support_func):
    found_strong_rule = False
    if len(f_itemset) > 1:
        # We need the support of the full itemset (f_itemset) for confidence calculation.
        # Instead of calling calculate_support_for_candidate again, 
        # we should ideally use the support value that made f_itemset frequent.
        # However, opus_miner_get_support_func is passed, which can use caching.
        support_f_itemset = opus_miner_get_support_func(f_itemset)

        if support_f_itemset == 0: # Should not happen for a frequent itemset
            return None

        for i in range(1, len(f_itemset)):
            for antecedent_tuple in combinations(f_itemset, i):
                antecedent = frozenset(antecedent_tuple)
                support_antecedent = opus_miner_get_support_func(antecedent)
                confidence = 0
                if support_antecedent > 0:
                    confidence = support_f_itemset / support_antecedent
                
                if confidence >= min_conf:
                    found_strong_rule = True
                    break # Found one strong rule for this f_itemset
            if found_strong_rule:
                break
    
    return f_itemset if found_strong_rule else None

class OPUSMiner:
    def __init__(self):
        self.transaction_count = 0
        self.item_tids = defaultdict(set)  # Save transaction IDs where each item appears
        self.support_cache = {}  # support value caching
    
    def add_transaction(self, tid, items):
        self.transaction_count += 1
        for item in items:
            self.item_tids[item].add(tid)
    
    def get_support(self, items):
        if not items:
            return 0
            
        # If cached support value exists, return it
        items_key = frozenset(items)
        if items_key in self.support_cache:
            return self.support_cache[items_key]
        
        # Calculate support using TID intersection
        common_tids = set.intersection(*(self.item_tids[item] for item in items))
        support = len(common_tids) / self.transaction_count
        
        # Cache support values for frequently used itemsets
        if len(items) <= 3:  # Only cache small itemsets
            self.support_cache[items_key] = support
            
        return support
    
    def prune_candidates(self, candidates_to_prune, min_support, num_processes_for_prune=None): # Added num_processes
        # Prune candidates using OPUS style, now with parallel support calculation
        if not candidates_to_prune:
            return set()

        if num_processes_for_prune is None:
            num_processes_for_prune = multiprocessing.cpu_count() 

        pruned_set = set()
        # For parallel processing, it's better to convert the set of candidates to a list first.
        candidate_list = list(candidates_to_prune)
        
        # Prepare tasks for parallel support calculation
        # Pass a copy of the current support_cache if it's small, or handle caching strategy carefully.
        # For simplicity, let's assume each worker might re-calculate some supports if cache is not effectively shared or large.
        # A read-only snapshot of the cache can be passed.
        tasks = [(self.item_tids, self.transaction_count, dict(self.support_cache), cand) for cand in candidate_list]

        print(f"      [Debug OPUS Prune] Pruning {len(candidate_list)} candidates using {num_processes_for_prune} processes...")
        if tasks:
            with multiprocessing.Pool(processes=num_processes_for_prune) as pool:
                results = pool.starmap(calculate_support_for_opus_candidate, tasks)
            
            for support, itemset_cand in results:
                if support >= min_support:
                    pruned_set.add(itemset_cand)
                    # Optionally, update the main miner's cache here, but be mindful of potential race conditions
                    # if multiple levels were to somehow write concurrently (not the case here as prune is per level).
                    # For safety, cache update can be done after pool completion based on `results` if needed for future `get_support` calls
                    # outside of parallel pruning.
                    # items_key = frozenset(itemset_cand)
                    # if len(itemset_cand) <= 3 and items_key not in self.support_cache: # Update if small and not present
                    #    self.support_cache[items_key] = support
        return pruned_set


def opus(df, min_support=0.5, min_confidence=0.8, num_processes=None, file_type_for_limit=None, max_level_limit=None, itemset_limit=None, turn_counter=None, params_str=None,
         enable_dynamic_support=False, dynamic_support_threshold=500000, support_increment_factor=1.2, **kwargs):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    print(f"    [Debug OPUS Init] Algorithm: OPUS, Input df shape: {df.shape}, min_support={min_support}, min_confidence={min_confidence}, num_processes={num_processes}, file_type_for_limit='{file_type_for_limit}', max_level_limit={max_level_limit}, itemset_limit={itemset_limit}")
    start_time_total = pd.Timestamp.now()

    # Initialize OPUS miner
    miner = OPUSMiner()
    
    # --- PERFORMANCE OPTIMIZATION: Vectorized Data Transformation ---
    print(f"    [Debug OPUS DataLoad] Initializing OPUSMiner and loading transactions...")
    start_time_dataload = pd.Timestamp.now()
    '''
    for tid, row in enumerate(df.itertuples(index=False, name=None)):
        items = set(f"{col}={val}" for col, val in zip(df.columns, row))
    '''

    cols_df = pd.DataFrame([df.columns.values] * len(df), index=df.index, columns=df.columns)
    data_str_df = df.astype(str)
    item_df = cols_df + '=' + data_str_df
    all_items = item_df.unstack().dropna()
    # --- FIX: Group by level 1 (original row index) instead of level 0 (column name) ---
    transactions = all_items.groupby(level=1).apply(set)
    
    for tid, items in transactions.items():
        miner.add_transaction(tid, items)

    dataload_duration = (pd.Timestamp.now() - start_time_dataload).total_seconds()
    print(f"    [Debug OPUS DataLoad] Transactions loaded. Total transactions: {miner.transaction_count}. Time: {dataload_duration:.2f}s")
    # --- END OPTIMIZATION ---
    
    # --- DYNAMIC SUPPORT: Initialize local_min_support ---
    local_min_support = min_support

    # Find frequent 1-itemsets
    print(f"    [Debug OPUS Freq1] Finding frequent 1-itemsets...")
    frequent_items = {
        item for item in miner.item_tids
        if len(miner.item_tids[item]) / miner.transaction_count >= local_min_support # Use local_min_support
    }
    print(f"    [Debug OPUS Freq1] Found {len(frequent_items)} frequent 1-itemsets.")
    if not frequent_items:
        print("    [Debug OPUS Freq1] No frequent 1-items found. Returning empty list.")
        return [], 0
    
    # Set for rule storage
    rule_set = set()
    
    # Incremental pattern discovery using OPUS style
    current_level_itemsets = {frozenset([item]) for item in frequent_items} # Renamed for clarity
    
    level_count = 1
    while current_level_itemsets:
        '''
        if file_type_for_limit in ['MiraiBotnet', 'NSL-KDD', 'NSL_KDD', 'CICIDS2017', 'CICIDS', 'Kitsune', 'CICModbus23', 'CICModbus', 'IoTID20', 'IoTID', 'netML', 'DARPA98', 'DARPA'] and \
            max_level_limit is not None and \
            level_count > max_level_limit:
            print(f"    [Debug OPUS Loop-{level_count}] Reached max_level_limit ({max_level_limit}) for file_type '{file_type_for_limit}'. Breaking OPUS loop.")
        '''
        # --- OPTIMIZATION: Check level limit at the BEGINNING of the loop ---
        if max_level_limit is not None and level_count > max_level_limit:
            print(f"    [Debug OPUS Loop-{level_count}] Reached max_level_limit ({max_level_limit}). Breaking OPUS loop.")
            break
        # --- END OPTIMIZATION ---

        if not current_level_itemsets:
            print(f"    [Debug OPUS Loop-{level_count}] current_level_itemsets is empty. Breaking.")
            break
        current_itemset_size = len(next(iter(current_level_itemsets)))
        print(f"    [Debug OPUS Loop-{level_count}] Processing itemsets of size: {current_itemset_size}. Num itemsets in current_level: {len(current_level_itemsets)}")
        
        # Rule Generation from current_level_itemsets - NOW PARALLELIZED
        if current_level_itemsets: # Ensure there are itemsets to process for rules
            print(f"      [Debug OPUS Loop-{level_count}] Generating rules from {len(current_level_itemsets)} (size {current_itemset_size}) frequent itemsets using {num_processes} processes...")
            rule_gen_tasks = [
                (itemset, min_confidence, miner.item_tids, miner.transaction_count, miner.get_support) 
                for itemset in current_level_itemsets
            ]
            if rule_gen_tasks:
                with multiprocessing.Pool(processes=num_processes) as pool:
                    results_validated_fitemsets_for_rules = pool.starmap(generate_opus_rules_for_itemset_task, rule_gen_tasks)
                
                # --- NEW: Logic to save rules incrementally ---
                new_rules_this_level = []
                for f_itemset_with_strong_rule in results_validated_fitemsets_for_rules:
                    if f_itemset_with_strong_rule:
                        rule_dict = {}
                        for rule_item_str in f_itemset_with_strong_rule:
                            key, value = rule_item_str.split('=', 1) # Allow '=' in value
                            try:
                                val_float = float(value)
                                rule_dict[key] = int(val_float) if val_float.is_integer() else val_float
                            except ValueError:
                                rule_dict[key] = value
                        rule_tuple = tuple(sorted(rule_dict.items()))
                        if rule_tuple not in rule_set:
                            rule_set.add(rule_tuple)
                            new_rules_this_level.append(rule_dict)
                
                # Save the newly generated rules for this level to the log file
                if turn_counter is not None and params_str is not None and new_rules_this_level:
                    save_association_artifacts(new_rules_this_level, file_type_for_limit, turn_counter, level_count, 'generated_rules.log', params_str)

            print(f"      [Debug OPUS Loop-{level_count}] Rule set size after processing level {current_itemset_size}: {len(rule_set)}")

        # Generate next level candidates (Apriori-gen style before OPUS prune)
        next_level_candidates = set() # Moved this line higher, was after rule gen.
        for itemset_idx, itemset in enumerate(current_level_itemsets):
            if itemset_idx > 0 and itemset_idx % 500 == 0: # Log every 500 itemsets
                print(f"      [Debug OPUS Loop-{level_count}] Processing itemset {itemset_idx}/{len(current_level_itemsets)} for rules: {itemset}")

            '''
            # Efficient subset processing for rule generation
            if len(itemset) > 1:
            for i in range(1, len(itemset)):
                    for antecedent_tuple in combinations(itemset, i):
                        antecedent = frozenset(antecedent_tuple)
                        # consequent = itemset - antecedent # Not directly used here
                    
                    ant_support = miner.get_support(antecedent)
                    if ant_support > 0:
                            itemset_support = miner.get_support(itemset) # Get support of full itemset for confidence
                            confidence = itemset_support / ant_support
                        
                        if confidence >= min_confidence:
                            # Convert rule to sorted tuple
                            rule_dict = {}
                                for rule_item_str in itemset: # OPUS typically considers the frequent itemset itself as a pattern/rule context
                                    key, value = rule_item_str.split('=')
                                    try:
                                        val_float = float(value)
                                        rule_dict[key] = int(val_float) if val_float.is_integer() else val_float
                                    except ValueError:
                                        rule_dict[key] = value # Keep as string
                            
                            rule_tuple = tuple(sorted(rule_dict.items()))
                                if rule_tuple not in rule_set:
                            rule_set.add(rule_tuple)
                                    # if len(rule_set) % 500 == 0:
                                    #     print(f"        [Debug OPUS Loop-{level_count}] Added rule/itemset (total {len(rule_set)}): {rule_tuple}, Conf (of a derived rule): {confidence:.4f}")
            '''
            
            # Generate next level candidates (Apriori-gen style before OPUS prune)
            # Based on the provided code, it seems to use a Apriori-like candidate generation.
            for other_itemset_idx, other_itemset in enumerate(current_level_itemsets):
                 if itemset_idx < other_itemset_idx: # Avoid duplicates and self-comparison for candidate generation
                    if len(itemset.intersection(other_itemset)) == current_itemset_size - 1: # Join if they share k-1 items
                        new_candidate = itemset.union(other_itemset)
                        # Pruning of subsets (Apriori property) is implicitly handled if all subsets were in current_level_itemsets
                        # The explicit check `all(frozenset(subset) in current_level_itemsets ...)` is more robust for Apriori-gen
                        # For OPUS, the pruning is done by miner.prune_candidates later on all generated next_level_candidates
                        # The current code seems to imply adding all k+1 candidates formed this way then pruning. Let's follow that.
                        if len(new_candidate) == current_itemset_size + 1:
                             # Check if all (k)-subsets are in current_level_itemsets (Apriori prune before adding to next_level_candidates)
                            all_subsets_frequent_in_current_level = True
                            if current_itemset_size > 0:
                                for subset_check_tuple in combinations(new_candidate, current_itemset_size):
                                    if frozenset(subset_check_tuple) not in current_level_itemsets:
                                        all_subsets_frequent_in_current_level = False
                                        break
                            if all_subsets_frequent_in_current_level:
                                #if len(next_level_candidates) % 1000 == 0 and len(next_level_candidates) > 0:
                                #    print(f"        [Debug OPUS Loop-{level_count}] Generated candidate for next_level_candidates (count {len(next_level_candidates)}): {new_candidate}")
                                next_level_candidates.add(new_candidate)
        
        print(f"    [Debug OPUS Loop-{level_count}] Generated {len(next_level_candidates)} candidates for next level. Now pruning...")
        
        # --- Optional: Early warning for excessive candidates (before pruning) ---
        if itemset_limit is not None and len(next_level_candidates) > itemset_limit * 10:
            print(f"    [WARN] Large number of candidates ({len(next_level_candidates)}) before pruning. This may indicate low min_support.")
            print(f"    Note: itemset_limit ({itemset_limit}) will be checked after pruning.")
        # Prune candidates using OPUS style to determine next level
        start_prune_time = pd.Timestamp.now()
        # Pass num_processes to the prune_candidates method
        # --- DYNAMIC SUPPORT: Loop for this level's support calculation ---
        while True:
            pruned_candidates = miner.prune_candidates(next_level_candidates, local_min_support, num_processes_for_prune=num_processes) # USE local_min_support

            # --- DYNAMIC SUPPORT: Check and adjust ---
            if enable_dynamic_support and len(pruned_candidates) > dynamic_support_threshold:
                print(f"    [DYNAMIC SUPPORT] At level {level_count}, itemset count ({len(pruned_candidates)}) after pruning exceeded threshold ({dynamic_support_threshold}).")
                local_min_support *= support_increment_factor
                print(f"    [DYNAMIC SUPPORT] Increasing min_support to {local_min_support:.4f} and retrying level {level_count} pruning.")
                # The same candidates will be pruned again with a higher support
                continue
            else:
                # Itemset count is acceptable, proceed
                current_level_itemsets = pruned_candidates
                break
        
        # --- DYNAMIC SUPPORT: End of loop for this level ---

        # --- NEW: Save frequent itemsets ---
        if turn_counter is not None and params_str is not None:
            itemsets_to_save = [dict(zip([f"item_{i+1}" for i in range(len(fs))], fs)) for fs in current_level_itemsets]
            save_association_artifacts(itemsets_to_save, file_type_for_limit, turn_counter, level_count + 1, 'frequent_itemsets.log', params_str)

        prune_duration = (pd.Timestamp.now() - start_prune_time).total_seconds()
        print(f"    [Debug OPUS Loop-{level_count}] Pruning finished. Next level (current_level_itemsets) size: {len(current_level_itemsets)}. Pruning time: {prune_duration:.2f}s. Total rules/itemsets so far: {len(rule_set)}")
        
        # --- ITEMSET_LIMIT CHECK (Safety Valve) ---
        if itemset_limit is not None and len(current_level_itemsets) > itemset_limit:
            print(f"    [WARN] Safety valve triggered: Number of frequent itemsets ({len(current_level_itemsets)}) exceeds the limit ({itemset_limit}).")
            print("    Stopping rule generation for this level to prevent memory overflow.")
            print("    Consider increasing min_support or min_confidence.")
            miner.support_cache.clear()
            break
        
        # Memory management: Clear support cache when no longer needed (though OPUSMiner caches only small itemsets)
        if not current_level_itemsets:
            print(f"    [Debug OPUS Loop-{level_count}] current_level_itemsets is empty after pruning. Clearing support cache. Breaking loop.")
            miner.support_cache.clear()
            break # Exit while loop
        level_count +=1
    
    total_duration = (pd.Timestamp.now() - start_time_total).total_seconds()
    print(f"    [Debug OPUS Finish] OPUS processing finished. Total unique rules/itemsets recorded: {len(rule_set)}. Total time: {total_duration:.2f}s")
    # Convert results
    max_level_reached = level_count
    return [dict(rule) for rule in rule_set], max_level_reached
