# Algorithm: H-Mine (H-Structure Mining)
# Output: Association rules Dictionary List; [{'feature1': value1, 'feature2': value2, ...}, {...}, ...]

from collections import defaultdict
from itertools import combinations
import pandas as pd # For pd.Timestamp
import multiprocessing # Add multiprocessing
import logging # Add logging

# Module Imports
# --- Import for saving artifacts ---
from Dataset_Choose_Rule.isv_save_log import save_association_artifacts


# Helper function for parallel support calculation (similar to RARM's)
# Needs to be defined at the top level or be picklable by multiprocessing
def calculate_support_for_candidate_hmine(item_tids, transaction_count, candidate_itemset):
    if not candidate_itemset:
        return 0, candidate_itemset
    # Ensure all items in candidate_itemset are in item_tids to prevent KeyError
    if not all(item in item_tids for item in candidate_itemset):
        # print(f"Warning: Item in {candidate_itemset} not found in TIDs for HMine. Skipping.")
        return 0, candidate_itemset 
    common_tids = set.intersection(*(item_tids[item] for item in candidate_itemset))
    support = len(common_tids) / transaction_count if transaction_count > 0 else 0
    return support, candidate_itemset

class HStructure:
    def __init__(self):
        self.item_counts = defaultdict(int)
        self.transaction_count = 0
        self.item_tids = defaultdict(set)  # Save transaction IDs where each item appears
    
    def add_transaction(self, tid, items):
        self.transaction_count += 1
        for item in items:
            self.item_counts[item] += 1
            self.item_tids[item].add(tid)
    
    def get_support(self, items):
        if not items:
            return 0
        # Calculate the number of transactions where all items appear simultaneously
        # Ensure all items are in self.item_tids
        if not all(item in self.item_tids for item in items):
            # print(f"Warning (HStructure.get_support): Item in {items} not found. Returning 0 support.")
            return 0
        common_tids = set.intersection(*[self.item_tids[item] for item in items])
        return len(common_tids) / self.transaction_count if self.transaction_count > 0 else 0

# Helper function for parallel rule generation for H-Mine
def generate_hmine_rules_for_itemset_task(f_itemset, min_conf, h_struct_get_support_func):
    found_strong_rule = False
    if len(f_itemset) > 1:
        support_f_itemset = h_struct_get_support_func(f_itemset)
        if support_f_itemset == 0:
            return None

        for i in range(1, len(f_itemset)):
            for antecedent_tuple in combinations(f_itemset, i):
                antecedent = frozenset(antecedent_tuple)
                support_antecedent = h_struct_get_support_func(antecedent)
                confidence = 0
                if support_antecedent > 0:
                    confidence = support_f_itemset / support_antecedent
                
                if confidence >= min_conf:
                    found_strong_rule = True
                    break
            if found_strong_rule:
                break
    
    return f_itemset if found_strong_rule else None

def h_mine(df, min_support=0.5, min_confidence=0.8, num_processes=None, file_type_for_limit=None, max_level_limit=None, itemset_limit=None, turn_counter=None, params_str=None,
           enable_dynamic_support=False, dynamic_support_threshold=500000, support_increment_factor=1.2, **kwargs):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    print(f"    [Debug H-Mine Init] Algorithm: H-Mine, Input df shape: {df.shape}, min_support={min_support}, min_confidence={min_confidence}, num_processes={num_processes}, file_type_for_limit='{file_type_for_limit}', max_level_limit={max_level_limit}, itemset_limit={itemset_limit}")
    start_time_total = pd.Timestamp.now()

    # Initialize H-Structure
    h_struct = HStructure()
    
    # --- PERFORMANCE OPTIMIZATION: Vectorized Data Transformation ---
    print(f"    [Debug H-Mine BuildStruct] Building H-Structure...")
    start_time_build = pd.Timestamp.now()
    '''
    transaction_items = []
    for tid, row in enumerate(df.itertuples(index=False, name=None)):
        items = set(f"{col}={row[idx]}" for idx, col in enumerate(df.columns))
        transaction_items.append(items)  # Keep full transactions for later use
    '''

    cols_df = pd.DataFrame([df.columns.values] * len(df), index=df.index, columns=df.columns)
    data_str_df = df.astype(str)
    item_df = cols_df + '=' + data_str_df
    all_items = item_df.unstack().dropna()
    # Use frozenset as H-mine's add_transaction expects it
    # --- FIX: Group by level 1 (original row index) instead of level 0 (column name) ---
    transactions = all_items.groupby(level=1).apply(frozenset) 
    
    for tid, items in transactions.items():
        h_struct.add_transaction(tid, items)

    build_duration = (pd.Timestamp.now() - start_time_build).total_seconds()
    print(f"    [Debug H-Mine BuildStruct] H-Structure built. Total transactions: {h_struct.transaction_count}. Time: {build_duration:.2f}s")
    # --- END OPTIMIZATION ---
    
    # --- DYNAMIC SUPPORT: Initialize local_min_support ---
    local_min_support = min_support
    
    # Find frequent 1-itemsets
    print(f"    [Debug H-Mine Freq1] Finding frequent 1-itemsets...")
    frequent_items = {
        item for item, count in h_struct.item_counts.items()
        if count / h_struct.transaction_count >= local_min_support # Use local_min_support
    }
    print(f"    [Debug H-Mine Freq1] Found {len(frequent_items)} frequent 1-itemsets.")
    if not frequent_items:
        print("    [Debug H-Mine Freq1] No frequent 1-items found. Returning empty list.")
        return [], 0
    
    # Use set for rule storage (optimized for duplicate removal)
    rule_set = set()
    
    # Generate frequent itemsets and extract rules
    current_level_itemsets = [frozenset([item]) for item in frequent_items] # Changed variable name for clarity
    
    level_count = 1
    max_itemset_size = len(df.columns) if not df.empty else len(frequent_items)

    while current_level_itemsets and len(current_level_itemsets[0]) < max_itemset_size:
        '''
        if file_type_for_limit in ['MiraiBotnet', 'NSL-KDD', 'NSL_KDD', 'CICIDS2017', 'CICIDS', 'Kitsune', 'CICModbus23', 'CICModbus', 'IoTID20', 'IoTID', 'netML', 'DARPA98', 'DARPA'] and \
            max_level_limit is not None and \
            level_count > max_level_limit:
            print(f"    [Debug H-Mine Loop-{level_count}] Reached max_level_limit ({max_level_limit}) for file_type '{file_type_for_limit}'. Breaking H-Mine loop.")
        '''
        # --- OPTIMIZATION: Check level limit at the BEGINNING of the loop ---
        if max_level_limit is not None and level_count > max_level_limit:
            print(f"    [Debug H-Mine Loop-{level_count}] Reached max_level_limit ({max_level_limit}). Breaking H-Mine loop.")
            break
        # --- END OPTIMIZATION ---

        if not current_level_itemsets: 
            print(f"    [Debug H-Mine Loop-{level_count}] current_level_itemsets is empty. Breaking.")
            break
        current_itemset_size = len(current_level_itemsets[0]) 
        print(f"    [Debug H-Mine Loop-{level_count}] Processing itemsets of size: {current_itemset_size}. Num itemsets in current_level: {len(current_level_itemsets)}")
        
        # --- DYNAMIC SUPPORT: Loop for this level's support calculation ---
        while True:
            # Generate all potential next_level candidates first (Apriori-gen)
            potential_next_level_candidates = set()
            # Convert current_level_itemsets to a set for efficient lookup during subset checking if it's not already
            current_level_set_for_lookup = set(current_level_itemsets) 

            # Candidate Generation (Apriori-gen: join + prune)
            # Sort current_level_itemsets to ensure consistent pairing for candidate generation (optional but good practice)
            # For frozensets, direct sorting is not possible, but list of them can be sorted if elements are comparable
            # However, the itemset_idx < other_itemset_idx handles unique pairs.
            for i in range(len(current_level_itemsets)):
                for j in range(i + 1, len(current_level_itemsets)):
                    itemset1 = current_level_itemsets[i]
                    itemset2 = current_level_itemsets[j]
                    
                    # Join step: Check if they share k-1 items
                    # For frozensets, intersection size then union is fine.
                    # Or, convert to list, sort, compare first k-1, then merge.
                    # union_len = len(itemset1.union(itemset2))
                    # if union_len == current_itemset_size + 1: # A simpler check if they differ by one item, implies k-1 shared
                    
                    # More robust join: check if first k-1 items are the same (assuming items within frozenset are somehow ordered or comparable for this)
                    # A common way for Apriori-gen with frozensets:
                    if len(itemset1.intersection(itemset2)) == current_itemset_size - 1:
                        new_candidate = itemset1.union(itemset2)
                        if len(new_candidate) == current_itemset_size + 1:
                            # Pruning step: check if all (k)-subsets are in current_level_itemsets
                            all_subsets_frequent = True
                            if current_itemset_size > 0: # For candidates of size > 1
                                for subset_to_check_tuple in combinations(new_candidate, current_itemset_size):
                                    if frozenset(subset_to_check_tuple) not in current_level_set_for_lookup:
                                        all_subsets_frequent = False
                                        break
                            if all_subsets_frequent:
                                potential_next_level_candidates.add(new_candidate)

            print(f"    [Debug H-Mine Loop-{level_count}] Generated {len(potential_next_level_candidates)} potential candidates for next level.")
            if not potential_next_level_candidates:
                print(f"    [Debug H-Mine Loop-{level_count}] No potential candidates generated. Breaking loop.")
                break

            # Parallel support calculation for candidates
            actual_next_level_itemsets = set()
            tasks = [(h_struct.item_tids, h_struct.transaction_count, cand) for cand in potential_next_level_candidates]

            if tasks:
                print(f"    [Debug H-Mine Loop-{level_count}] Calculating support for {len(tasks)} candidates using {num_processes} processes...")
                with multiprocessing.Pool(processes=num_processes) as pool:
                    results = pool.starmap(calculate_support_for_candidate_hmine, tasks)
                
                for support, itemset_cand in results:
                    if support >= local_min_support: # USE local_min_support
                        actual_next_level_itemsets.add(itemset_cand)

            # --- DYNAMIC SUPPORT: Check and adjust ---
            if enable_dynamic_support and len(actual_next_level_itemsets) > dynamic_support_threshold:
                print(f"    [DYNAMIC SUPPORT] At level {level_count}, itemset count ({len(actual_next_level_itemsets)}) exceeded threshold ({dynamic_support_threshold}).")
                local_min_support *= support_increment_factor
                print(f"    [DYNAMIC SUPPORT] Increasing min_support to {local_min_support:.4f} and retrying level {level_count} support calculation.")
                # We need to re-evaluate frequent_1_itemsets as well, as they form the basis for candidates
                # This makes the retry more complex. A simpler approach is to restart the whole `h_mine` process
                # like Eclat, but let's try an in-loop adjustment first.
                # For now, we will just re-run the support calculation for the same candidates.
                continue
            else:
                break
        
        # --- DYNAMIC SUPPORT: End of loop for this level ---

        # --- ITEMSET_LIMIT CHECK (Safety Valve) ---
        if itemset_limit is not None and len(actual_next_level_itemsets) > itemset_limit:
            print(f"    [WARN] Safety valve triggered: Number of frequent itemsets ({len(actual_next_level_itemsets)}) exceeds the limit ({itemset_limit}).")
            print("    Stopping rule generation for this level to prevent memory overflow.")
            print("    Consider increasing min_support or min_confidence.")
            break

        # --- NEW: Save frequent itemsets ---
        if turn_counter is not None and params_str is not None:
            itemsets_to_save = [dict(zip([f"item_{i+1}" for i in range(len(fs))], fs)) for fs in actual_next_level_itemsets]
            save_association_artifacts(itemsets_to_save, file_type_for_limit, turn_counter, level_count + 1, 'frequent_itemsets.log', params_str)

        # Rule generation from current_level_itemsets - NOW PARALLELIZED
        if current_level_itemsets: # Process rules if current_level is not empty
            print(f"    [Debug H-Mine Loop-{level_count}] Generating rules from {len(current_level_itemsets)} (k={current_itemset_size}) frequent itemsets using {num_processes} processes...")
            rule_gen_tasks_hmine = [
                (item_set, min_confidence, h_struct.get_support) 
                for item_set in current_level_itemsets # Rules from current level's frequent itemsets
            ]
            if rule_gen_tasks_hmine:
                with multiprocessing.Pool(processes=num_processes) as pool:
                    results_validated_fitemsets_for_hmine_rules = pool.starmap(generate_hmine_rules_for_itemset_task, rule_gen_tasks_hmine)
                
                # --- NEW: Logic to save rules incrementally ---
                new_rules_this_level = []
                for f_itemset_with_strong_rule in results_validated_fitemsets_for_hmine_rules:
                    if f_itemset_with_strong_rule:
                        rule_dict = {}
                        for rule_item_str in f_itemset_with_strong_rule:
                            key, value = rule_item_str.split('=', 1)
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

            print(f"    [Debug H-Mine Loop-{level_count}] Rule set size after level {level_count} rule generation: {len(rule_set)}")

        # Update current_level_itemsets for the next iteration
        if not actual_next_level_itemsets:
            print(f"    [Debug H-Mine Loop-{level_count}] Next_level (actual frequent) is empty. Breaking loop.")
            break
        current_level_itemsets = list(actual_next_level_itemsets) # Convert set to list for next iteration's indexed access
        level_count +=1
    
    total_duration = (pd.Timestamp.now() - start_time_total).total_seconds()
    print(f"    [Debug H-Mine Finish] H-Mine processing finished. Total unique rules/itemsets recorded: {len(rule_set)}. Total time: {total_duration:.2f}s")
    # Convert final result to dictionary list
    max_level_reached = level_count
    return [dict(rule) for rule in rule_set], max_level_reached
