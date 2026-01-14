# Algorithm: Eclat (Equivalence Class Clustering and bottom-up Lattice Traversal)
# Using set intersection operations to find infrequent items
# Output: Association rules Dictionary List; [{'feature1': value1, 'feature2': value2, ...}, {...}, ...]

import itertools
from collections import defaultdict
import pandas as pd
import multiprocessing
import logging

# Module Imports
# --- Import for saving artifacts ---
from Dataset_Choose_Rule.isv_save_log import save_association_artifacts


# Calculate Support for how many times a particular itemset appears in the overall data
def get_support(transaction_list, itemset):
    count = sum(1 for transaction in transaction_list if itemset.issubset(transaction))
    return count / len(transaction_list)

def get_confidence(transaction_list, base, full):
    # Confidence = P(Full | Base) = Support(Full) / Support(Base)
    base_support = get_support(transaction_list, base)
    full_support = get_support(transaction_list, full)
    return full_support / base_support if base_support > 0 else 0


def get_support_optimized(tid_map, itemset):
    # Optimize support calculation using tid_map
    if len(itemset) == 1:
        return len(tid_map[next(iter(itemset))]) / tid_map['total']
    
    # Calculate support through intersection calculation
    tids = set.intersection(*[tid_map[item] for item in itemset])
    return len(tids) / tid_map['total']


def get_confidence_optimized(tid_map, base, full):
    base_support = get_support_optimized(tid_map, base)
    full_support = get_support_optimized(tid_map, full)
    return full_support / base_support if base_support > 0 else 0


# Helper function for Eclat parallel subtree processing
def process_eclat_subtree(prefix_itemset, items_to_process_with_tids, min_support, confidence_threshold, tid_map_global, total_transactions_global, file_type_for_limit, max_level_limit, turn_counter=None, params_str=None):
    # This function will perform the Eclat logic for a given starting prefix and its associated items/TID lists.
    # It needs access to global tid_map (or relevant parts) and total_transactions.
    # Note: tid_map here is the full map, get_support_optimized uses it.

    local_frequent_itemsets = set()
    local_rule_set = set()
    
    # Stack for this subtree, initialized with the given items to process for the prefix
    # Each element in items_to_process_with_tids: (item_set_to_add, current_tids_for_item_set_to_add)
    # The initial prefix_itemset is the 'root' for this subtree.
    
    # The stack will store: (current_combined_prefix, list_of_further_items_with_their_tids)
    # Initial items for the stack are derived from items_to_process_with_tids, 
    # where each item is considered as an extension of the initial prefix_itemset.

    # We need to adapt the main Eclat stack logic here for the given prefix.
    # The `items_to_process_with_tids` are candidates to extend the initial `prefix_itemset`.

    # Let's refine the stack for process_eclat_subtree
    # stack stores (current_prefix_being_built, potential_extensions_with_tids)
    # Initial call: prefix_itemset is the first item (e.g., {A}), 
    # items_to_process_with_tids are other 1-items {{B}, tids_B}, {{C}, tids_C} etc. that can extend A

    # Initial items for this subtree processing based on the starting prefix_itemset
    # The stack will manage extensions from this prefix_itemset
    # stack: (current_full_itemset, list_of_remaining_items_that_can_extend_it_with_their_intersected_tids)
    # The items in items_to_process_with_tids are individual 1-itemsets

    # Structure for stack items: (current_itemset, list_of_potential_next_items_with_their_tids)
    # `current_itemset` is the itemset built so far (e.g. prefix_itemset + some extensions)
    # `list_of_potential_next_items_with_their_tids` are items that can extend `current_itemset`
    # Each element in this list: (single_item_frozenset, tids_of_that_single_item)
    
    # Initial stack based on the initial prefix_itemset (which is a 1-itemset)
    # and items_to_process_with_tids (other 1-itemsets with their original TIDs)
    # We need to form the initial combinations for the stack.

    # The main `eclat` function will prepare `tasks` for the pool.
    # Each task is (initial_1_item, list_of_other_1_items_with_tids, ...common_args...)
    # `prefix_itemset` here is that initial_1_item.
    # `items_to_process_with_tids` are the other_1_items_with_tids.

    # The original Eclat stack: (prefix, items) where prefix is the base, items are extensions.
    # Here, the given `prefix_itemset` (e.g. {A}) is the fixed base for this worker.
    # We explore extensions like {A,B}, {A,C}, then {A,B,D} etc.

    # Support for the initial prefix_itemset itself must be >= min_support
    # This should be pre-filtered by the main `eclat` function before creating tasks.
    # Assume prefix_itemset is already frequent.
    local_frequent_itemsets.add(frozenset(prefix_itemset)) # Add the initial prefix (1-itemset)

    # Generate initial items for the stack for this worker
    # These are 2-itemsets starting with prefix_itemset
    initial_extensions = []
    for other_item_fset, other_item_tids in items_to_process_with_tids:
        # other_item_fset is like frozenset(['item_str'])
        # We must ensure other_item is not the same as prefix_itemset's single item
        # and that we maintain an order to avoid duplicate pairs like (A,B) and (B,A) if tasks are (A, others) and (B, others)
        # This ordering is handled by the main `eclat` function when creating tasks.
        
        # TID list for the combined 2-itemset (prefix_itemset + other_item_fset)
        # `prefix_itemset` is a 1-itemset, so `tid_map_global[next(iter(prefix_itemset))]` gets its TIDs
        # `other_item_tids` are already provided.
        combined_tids = tid_map_global[next(iter(prefix_itemset))] & other_item_tids
        
        if len(combined_tids) / total_transactions_global >= min_support:
            # item_to_add_to_current_prefix, its_tid_list_when_combined_with_prefix
            initial_extensions.append((other_item_fset, combined_tids)) 

    # Stack: (current_path_from_initial_prefix, list_of_further_items_to_extend_with_their_intersected_tids)
    # current_path_from_initial_prefix: e.g., if initial prefix is {A}, this could be {B} (meaning current itemset is {A,B})
    # list_of_further_items: candidates to make {A,B,C}, {A,B,D} etc.
    # Each element: (single_item_frozenset_to_add, tids_for_current_path_unioned_with_single_item)

    # Stack: (current_itemset_being_built_on_top_of_prefix, list_of_potential_next_items_with_tids)
    # current_itemset_being_built_on_top_of_prefix is the full itemset, e.g. {A,B}
    # list_of_potential_next_items_with_tids: items like ({C}, tids_of_{A,B,C}), ({D}, tids_of_{A,B,D})
    
    # Start with the initial prefix_itemset. Items in initial_extensions are direct extensions.
    stack = [(prefix_itemset, initial_extensions)]

    while stack:
        current_base_itemset, items_for_extension = stack.pop()
        
        # --- OPTIMIZATION: Check level limit before processing extensions ---
        # The new itemsets would be 1 level deeper than the current base.
        next_level_size = len(current_base_itemset) + 1
        if max_level_limit is not None and next_level_size > max_level_limit:
            # This entire branch of the search tree is beyond the limit.
            continue # Skip processing this stack item
        # --- END OPTIMIZATION ---

        # items_for_extension is a list of (single_item_fset_to_add, tids_if_added_to_current_base_itemset)
        # We need to iterate through these, form new itemsets, check support (already done for these), add to frequent, and generate next level for stack.
        
        idx = -1
        while idx < len(items_for_extension) -1:
            idx += 1 # Process one by one to maintain order for generating next candidates
            item_to_add, item_tids_when_added = items_for_extension[idx] 
            
            # new_itemset_formed is current_base_itemset + item_to_add
            new_itemset_formed = current_base_itemset.union(item_to_add)
            
            # --- NEW: Save frequent itemsets ---
            # Note: We save inside the loop as each new itemset is discovered.
            if turn_counter is not None and params_str is not None:
                itemsets_to_save = [dict(zip([f"item_{i+1}" for i in range(len(new_itemset_formed))], new_itemset_formed))]
                save_association_artifacts(itemsets_to_save, file_type_for_limit, turn_counter, len(new_itemset_formed), 'frequent_itemsets.log', params_str)

            # Support is already pre-calculated for this new_itemset_formed (it's len(item_tids_when_added) / total_transactions_global)
            local_frequent_itemsets.add(frozenset(new_itemset_formed))

            # Rule Generation for new_itemset_formed
            if len(new_itemset_formed) > 1:
                new_rules_this_level = []
                for base_size in range(1, len(new_itemset_formed)):
                    for base_tuple_local in itertools.combinations(new_itemset_formed, base_size):
                        base_set_local = frozenset(base_tuple_local)
                        # Confidence calc uses global tid_map
                        confidence = get_confidence_optimized(tid_map_global, base_set_local, new_itemset_formed)
                        if confidence >= confidence_threshold:
                            rule_dict = {}
                            for rule_item_str in new_itemset_formed: # Store the full frequent itemset as the rule context
                                key, value = rule_item_str.split('=', 1)
                                try:
                                    val_float = float(value)
                                    rule_dict[key] = int(val_float) if val_float.is_integer() else val_float
                                except ValueError:
                                    rule_dict[key] = value
                            rule_tuple = tuple(sorted(rule_dict.items()))
                            if rule_tuple not in local_rule_set:
                                local_rule_set.add(rule_tuple)
                                new_rules_this_level.append(rule_dict)

                # Save the newly generated rules for this level to the log file
                if turn_counter is not None and params_str is not None and new_rules_this_level:
                    save_association_artifacts(new_rules_this_level, file_type_for_limit, turn_counter, len(new_itemset_formed), 'generated_rules.log', params_str)


            # Generate items for the next stack level: extend new_itemset_formed
            potential_next_extensions = []
            # Iterate over items that came *after* item_to_add in items_for_extension list to maintain order and avoid duplicates
            for next_potential_item_idx in range(idx + 1, len(items_for_extension)):
                next_item_to_combine_fset, _ = items_for_extension[next_potential_item_idx] # We need original TIDs of this next_item, not combined ones
                
                # Get original TIDs for next_item_to_combine_fset from global_tid_map
                # This assumes next_item_to_combine_fset is a 1-itemset like frozenset(['X'])
                original_tids_of_next_item = tid_map_global[next(iter(next_item_to_combine_fset))]

                # Intersect TIDs of new_itemset_formed (which are item_tids_when_added) with original TIDs of next_item_to_combine_fset
                new_combined_tids = item_tids_when_added & original_tids_of_next_item

                if len(new_combined_tids) / total_transactions_global >= min_support:
                    potential_next_extensions.append((next_item_to_combine_fset, new_combined_tids))
            
            if potential_next_extensions:
                '''
                # Level limit check before adding to stack
                # The new itemset to be formed would be of size: len(new_itemset_formed) + 1
                next_level_size = len(new_itemset_formed) + 1
                if file_type_for_limit in ['MiraiBotnet', 'NSL-KDD', 'NSL_KDD', 'CICIDS2017', 'CICIDS', 'Kitsune', 'CICModbus23', 'CICModbus', 'IoTID20', 'IoTID', 'netML', 'DARPA98', 'DARPA'] and \
                    max_level_limit is not None and \
                    next_level_size > max_level_limit:
                    # print(f"    [Debug Eclat Subtree] Reached max_level_limit ({max_level_limit}) for file_type '{file_type_for_limit}'. Not adding level {next_level_size} to stack.")
                    pass # Do not add to stack
                else:
                    stack.append((new_itemset_formed, potential_next_extensions))
                '''
                # The level limit check is now done at the start of the while loop,
                # so this check here is redundant and can be removed for cleaner code.
                stack.append((new_itemset_formed, potential_next_extensions))
                
    return local_frequent_itemsets, local_rule_set
            

# Eclat Algorithm: Finding infrequent itemsets using set intersection operations
def eclat(df, min_support=0.5, min_confidence=0.8, num_processes=None, file_type_for_limit=None, max_level_limit=None, itemset_limit=None, turn_counter=None, params_str=None,
          enable_dynamic_support=False, dynamic_support_threshold=500000, support_increment_factor=1.2, **kwargs):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    logger.info(f"    [Debug Eclat Init] Algorithm: Eclat, Input df shape: {df.shape}, min_support={min_support}, confidence_threshold={min_confidence}, num_processes={num_processes}, file_type_for_limit='{file_type_for_limit}', max_level_limit={max_level_limit}, itemset_limit={itemset_limit}, dynamic_support={enable_dynamic_support}")
    
    # --- DYNAMIC SUPPORT: Initialize local_min_support and outer loop ---
    local_min_support = min_support
    while True:
        start_time_total = pd.Timestamp.now()

        # --- PERFORMANCE OPTIMIZATION: Vectorized TID Map Generation ---
        print(f"    [Debug Eclat TIDMap] Creating TID map...")
        start_time_tidmap = pd.Timestamp.now()
        tid_map = defaultdict(set)
        tid_map['total'] = len(df)
        '''
        # Calculate TID list for each item (for memory and time efficiency)
        for tid, row in enumerate(df.itertuples(index=False, name=None)):
            for col_idx, col in enumerate(df.columns):
                item = f"{col}={row[col_idx]}"
                tid_map[item].add(tid)
        '''

        cols_df = pd.DataFrame([df.columns.values] * len(df), index=df.index, columns=df.columns)
        data_str_df = df.astype(str)
        item_df = cols_df + '=' + data_str_df
        all_items = item_df.unstack().dropna()
        
        # --- FIX: Use get_level_values(1) for row index, not 0 for column index ---
        grouped_tids = all_items.groupby(all_items.values).apply(lambda x: set(x.index.get_level_values(1)))
        tid_map.update(grouped_tids.to_dict())

        tidmap_duration = (pd.Timestamp.now() - start_time_tidmap).total_seconds()
        print(f"    [Debug Eclat TIDMap] TID map created. Total items in map (incl. 'total'): {len(tid_map)}. Total transactions: {tid_map['total']}. Time: {tidmap_duration:.2f}s")
        # --- END OPTIMIZATION ---
        
        # Create initial 1-itemsets that are frequent
        frequent_1_itemsets_with_tids = []
        for item_str in tid_map.keys():
            if item_str == 'total':
                continue
            item_fset = frozenset([item_str])
            # Support calculated using original TIDs from tid_map
            support = len(tid_map[item_str]) / tid_map['total'] 
            if support >= local_min_support: # USE local_min_support
                frequent_1_itemsets_with_tids.append((item_fset, tid_map[item_str]))
        
        # Sort frequent_1_itemsets_with_tids by item string to ensure consistent task generation order
        # This helps in the subtree processing to avoid redundant work like (A,B) and (B,A) if not handled.
        frequent_1_itemsets_with_tids.sort(key=lambda x: next(iter(x[0])))

        print(f"    [Debug Eclat Freq1] Found {len(frequent_1_itemsets_with_tids)} frequent 1-itemsets.")
        if not frequent_1_itemsets_with_tids:
            print("    [Debug Eclat Freq1] No frequent 1-itemsets found. Returning empty list.")
            return [], 0

        all_frequent_itemsets = set() 
        all_rules = set()

        # Add all frequent 1-itemsets to all_frequent_itemsets initially
        for item_fset, _ in frequent_1_itemsets_with_tids:
            all_frequent_itemsets.add(item_fset)
            # 1-itemsets don't form rules of the typical A->B type handled by combinations later.
            # If the problem implies single items that are frequent should also be in 'rules', this needs clarification.
            # Based on other algorithms, rule_set usually stores antecedents or full itemsets of size > 1.
            # For Eclat output format, if a frequent itemset itself is a rule context:
            # rule_dict = {key: val for key,val in [s.split('=',1) for s in item_fset]}
            # all_rules.add(tuple(sorted(rule_dict.items()))) # This would add 1-itemsets to rules

        # Prepare tasks for parallel processing
        # Each task will process a subtree starting with a frequent 1-itemset
        tasks = []
        for i in range(len(frequent_1_itemsets_with_tids)):
            prefix_1_itemset, _ = frequent_1_itemsets_with_tids[i] # Original TIDs not needed here for prefix, they are in tid_map_global
            
            # Items that can extend this prefix (items appearing after it in sorted list to avoid duplicates)
            # Each element: (other_1_item_fset, original_tids_of_other_1_item_fset)
            items_for_prefix_extension = [
                (f_item_fset, f_item_tids) 
                for j, (f_item_fset, f_item_tids) in enumerate(frequent_1_itemsets_with_tids) if j > i
            ]
            
            if items_for_prefix_extension: # Only create a task if there's something to extend with
                # Pass tid_map (global) and total_transactions for support/confidence calcs inside worker
                task = (
                    prefix_1_itemset, 
                    items_for_prefix_extension, 
                    local_min_support, 
                    min_confidence, 
                    tid_map, # global tid_map
                    tid_map['total'], # global total_transactions
                    file_type_for_limit,
                    max_level_limit
                )
                # --- NEW: Conditionally add logging params to task ---
                if turn_counter is not None and params_str is not None:
                    task += (turn_counter, params_str)
                else:
                    task += (None, None) # Add placeholders
                tasks.append(task)

        print(f"    [Debug Eclat Parallel] Starting parallel processing for {len(tasks)} subtrees using {num_processes} processes...")
        
        if tasks:
            with multiprocessing.Pool(processes=num_processes) as pool:
                # IMPORTANT: Pass local_min_support down to the worker
                # The task signature needs to be updated to accept this
                # Let's assume process_eclat_subtree is adapted.
                adapted_tasks = [
                    (prefix, items, local_min_support, confidence, tid_map, tid_map['total'], file_type_for_limit, max_level_limit, turn_counter, params_str)
                    for prefix, items, _, confidence, tid_map, total_transactions, file_type_for_limit, max_level_limit, turn_counter, params_str in tasks
                ]
                results = pool.starmap(process_eclat_subtree, adapted_tasks)
        
            # Aggregate results from all subtrees
            for frequent_sets_subtree, rules_subtree in results:
                all_frequent_itemsets.update(frequent_sets_subtree)
                all_rules.update(rules_subtree)
        
        # --- ITEMSET_LIMIT CHECK (Safety Valve) ---
        if itemset_limit is not None and len(all_frequent_itemsets) > itemset_limit:
            logger.warning(f"    [WARN] Safety valve triggered: Number of frequent itemsets ({len(all_frequent_itemsets)}) exceeds the limit ({itemset_limit}).")
            logger.warning("    Stopping rule generation to prevent memory overflow.")
            logger.warning("    Consider increasing min_support or min_confidence.")
            break
        
        # --- DYNAMIC SUPPORT: Check and adjust ---
        if enable_dynamic_support and len(all_frequent_itemsets) > dynamic_support_threshold:
            logger.warning(f"    [DYNAMIC SUPPORT] Total itemset count ({len(all_frequent_itemsets)}) exceeded threshold ({dynamic_support_threshold}).")
            local_min_support *= support_increment_factor
            logger.warning(f"    [DYNAMIC SUPPORT] Increasing min_support to {local_min_support:.4f} and restarting Eclat process.")
            continue # Restart the entire process
        else:
            break # Itemset count is acceptable
    
    total_duration = (pd.Timestamp.now() - start_time_total).total_seconds()
    print(f"    [Debug Eclat Finish] Eclat processing finished. Total frequent itemsets: {len(all_frequent_itemsets)}, Total rules: {len(all_rules)}. Total time: {total_duration:.2f}s")
    # Calculate max level from the maximum itemset size
    max_level_reached = max(len(itemset) for itemset in all_frequent_itemsets) if all_frequent_itemsets else 0
    return [dict(rule) for rule in all_rules], max_level_reached


# Original sequential Eclat code (commented out or for reference)
'''
def eclat_original_sequential(df, min_support=0.5, confidence_threshold=0.8):
    # ... (original sequential implementation as provided in the prompt) ...
    # Create TID mapping
    tid_map = defaultdict(set)
    tid_map['total'] = len(df)
    for tid, row in enumerate(df.itertuples(index=False, name=None)):
        for col_idx, col in enumerate(df.columns):
            item = f"{col}={row[col_idx]}"
            tid_map[item].add(tid)
    
    itemsets = {frozenset([item]) for item in tid_map.keys() if item != 'total'}
    frequent_itemsets_final_set = set()
    rule_set = set()
    stack = [(set(), list(itemsets))] # prefix, items_to_process (list of frozensets)
    
    while stack:
        prefix, items_to_process_list = stack.pop()
        
        # Create a copy to iterate while modifying the original list for recursive calls
        # More Eclat-like: iterate based on an ordering, then recurse with remaining items
        # Correct would be to iterate items_to_process_list with an index
        # and pass remaining part of list for recursion (items_to_process_list[i+1:])
        
        # Simplified from user's code structure
        # Convert items_to_process (which was a list of frozensets) 
        # to a list of (item_frozenset, item_tid_list) for the eclat_tid style recursion, 
        # or adapt the main loop to use get_support_optimized

        # Adopting the iterative stack-based approach from user's latest Eclat snippet:
        idx = -1
        while idx < len(items_to_process_list) -1:
            idx +=1
            item = items_to_process_list[idx] # item is a frozenset like frozenset(['col=val'])
            
            new_prefix = prefix.union(item)
            support = get_support_optimized(tid_map, new_prefix)
            
            if support >= min_support:
                frequent_itemsets_final_set.add(frozenset(new_prefix))

                # Rule generation
                if len(new_prefix) > 1:
                for base_size in range(1, len(new_prefix)):
                        for base_tuple in itertools.combinations(new_prefix, base_size):
                            base_set = frozenset(base_tuple)
                        confidence = get_confidence_optimized(tid_map, base_set, new_prefix)
                        if confidence >= confidence_threshold:
                            rule_dict = {}
                                for pair_str in new_prefix: 
                                    key, value = pair_str.split('=',1)
                                    try:
                                        val_float = float(value)
                                        rule_dict[key] = int(val_float) if val_float.is_integer() else val_float
                                    except ValueError:
                                        rule_dict[key] = value
                                rule_set.add(tuple(sorted(rule_dict.items())))
                
                # Generate items for the next stack level (recursive call in original Eclat)
                # For iterative version: find items that can extend new_prefix
                remaining_items_for_next_level = []
                for next_item_candidate_idx in range(idx + 1, len(items_to_process_list)):
                    next_item_fset = items_to_process_list[next_item_candidate_idx]
                    # In Eclat, we'd typically work with TID intersections for new candidates
                    # The provided sequential code does: combined_candidate = new_prefix.union(next_item_fset)
                    # then get_support_optimized(tid_map, combined_candidate)
                    # This is less efficient than intersecting TID lists directly.
                    # However, to match the structure of user's original loop for parallelization, this is complex.
                    
                    # The user's sequential Eclat loop for `remaining_items_for_next_level` was based on 
                    # `get_support_optimized(tid_map, new_prefix.union(other_item_candidate)) >= min_support` 
                    # where `other_item_candidate` was from the `items_to_process` list (the one popped from stack)
                    # This needs careful adaptation. The key for Eclat is to pass intersected TID lists.
                    
                    # For the sequential version, this was how it built the next stack item:
                    # if get_support_optimized(tid_map, new_prefix.union(items_to_process_list[next_item_candidate_idx])) >= min_support:
                    #    remaining_items_for_next_level.append(items_to_process_list[next_item_candidate_idx])

                # This part is tricky in the iterative version without direct TID list passing in stack items.
                # The original sequential code's stack was: (prefix, list_of_1_item_frozensets_to_try_adding)
                # It re-calculated support for combined. True Eclat passes intersected TIDs.

                # Let's assume the sequential code's structure for `remaining_items_for_next_level` for now:
                # This would require `items_to_process_list` to be available for this `new_prefix`.
                # This structure doesn't lend itself well to the classic Eclat TID intersection passing easily.

                # If following the user's existing iterative Eclat structure more closely:
                # The stack item would be (current_prefix, list_of_remaining_1_item_candidates_to_extend_prefix_with)
                # And we need to ensure the list_of_remaining ... is correctly filtered for the *next* deeper level.
                # The sequential code did: `stack.append((new_prefix, remaining_items_for_next_level))`
                # where `remaining_items_for_next_level` were single items that, when unioned with new_prefix, were frequent.

                # For parallelization, the `process_eclat_subtree` needs to embody the Eclat recursion/iteration starting from its assigned prefix.
                # The `items_for_extension` in `process_eclat_subtree` are (item_fset, combined_tids_with_prefix).
                # So, for the next level, we take `new_itemset_formed` and try to extend it with items from 
                # `items_for_extension` that appeared *after* the `item_to_add`.

                # The current sequential `eclat` in the prompt is a bit of a hybrid.
                # It has a stack like `(prefix, items_to_process_list_of_single_item_frozensets)`.
                # It then iterates `items_to_process_list`, forms `new_prefix = prefix.union(item)`.
                # Checks support of `new_prefix`. If frequent, adds to stack `(new_prefix, filtered_remaining_items_from_original_list)`.
                # This is the structure the parallel worker should mimic for its subtree.
                
                # In the parallel worker `process_eclat_subtree`:
                # `current_base_itemset` is like `new_prefix` from sequential.
                # `items_for_extension` is like `filtered_remaining_items_from_original_list` but with pre-calculated TIDs.

    return [dict(rule) for rule in rule_set]
'''
