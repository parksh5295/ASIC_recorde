# Algorithm: RARM (Rapid Association Rule Mining)
# Improve speed by reducing the search space by eliminating unnecessary candidates, and reduce memory usage by minimizing the set of intermediate candidates
# Output: Association rules Dictionary List; [{'feature1': value1, 'feature2': value2, ...}, {...}, ...]

from collections import defaultdict
from itertools import combinations
import multiprocessing # Add multiprocessing
import pandas as pd # Add pandas for Timestamp if not already there (it seems to be missing in the provided RARM snippet but likely present)
from tqdm import tqdm # ADDED for the progress bar
import logging

# Module Imports
# --- Import for saving artifacts ---
from Dataset_Choose_Rule.isv_save_log import save_association_artifacts



logger = logging.getLogger(__name__)

# Helper function for parallel support calculation
# Needs to be defined at the top level or be picklable by multiprocessing
# MODIFIED: This function now uses global variables set by the initializer.
def calculate_support_for_candidate(candidate_itemset):
    # Use the globally-scoped variables initialized in each worker process
    item_tids = _GLOBAL_ITEM_TIDS_RARM
    transaction_count = _GLOBAL_TOTAL_TX_RARM

    if not candidate_itemset:
        return 0, candidate_itemset # Return itemset for consistency if starmap expects it
    # Ensure all items in 'candidate_itemset' are present in item_tids to avoid KeyError
    # when an item might have been part of a candidate but had 0 support initially (though unlikely for frequent itemsets)
    valid_items_in_candidate = [item for item in candidate_itemset if item in item_tids]
    if len(valid_items_in_candidate) != len(candidate_itemset): # Should not happen if candidate comes from frequent items
        return 0, candidate_itemset


    common_tids = set.intersection(*(item_tids[item] for item in valid_items_in_candidate))
    support = len(common_tids) / transaction_count if transaction_count > 0 else 0
    return support, candidate_itemset

# NEW: Worker initializer and globals for RARM parallel rule generation
_GLOBAL_ITEM_TIDS_RARM = None
_GLOBAL_TOTAL_TX_RARM = 0

def _init_rarm_worker(item_tids, total_tx):
    """Initializes global variables for a RARM worker process."""
    global _GLOBAL_ITEM_TIDS_RARM, _GLOBAL_TOTAL_TX_RARM
    _GLOBAL_ITEM_TIDS_RARM = item_tids
    _GLOBAL_TOTAL_TX_RARM = total_tx

def _support_from_globals_rarm(items):
    """Calculates support using global TID map. For use in RARM worker processes."""
    if not items or not _GLOBAL_ITEM_TIDS_RARM or _GLOBAL_TOTAL_TX_RARM == 0:
        return 0.0
    
    # Safeguard: ensure all items are in the map to prevent KeyErrors
    if not all(item in _GLOBAL_ITEM_TIDS_RARM for item in items):
        return 0.0
        
    common_tids = set.intersection(*(_GLOBAL_ITEM_TIDS_RARM[item] for item in items))
    return len(common_tids) / _GLOBAL_TOTAL_TX_RARM

# NEW: Wrapper function for imap to handle multiple arguments for the rule generation task.
def _rarm_rule_worker_wrapper(args):
    """Helper to unpack arguments for pool.imap_unordered."""
    return generate_rules_for_itemset_task(*args)

# Helper function for parallel rule generation (OPUS/H-Mine/RARM share this)
# MODIFIED: It now uses global variables set by the initializer instead of receiving a function.
def generate_rules_for_itemset_task(f_itemset, min_conf):
    rules = []
    if len(f_itemset) > 1:
        support_f_itemset = _support_from_globals_rarm(f_itemset)
        if support_f_itemset == 0:
            return rules

        for i in range(1, len(f_itemset)):
            for antecedent_tuple in combinations(f_itemset, i):
                antecedent = frozenset(antecedent_tuple)
                support_antecedent = _support_from_globals_rarm(antecedent)
                
                if support_antecedent > 0:
                    confidence = support_f_itemset / support_antecedent
                    if confidence >= min_conf:
                        consequent = f_itemset - antecedent
                        rules.append((antecedent, consequent, confidence, support_f_itemset))
    return rules


class RARMiner:
    def __init__(self):
        self.transaction_count = 0
        self.item_tids = defaultdict(set)  # TID list
        self.item_counts = defaultdict(int)  # Item count
        
    def add_transaction(self, tid, items):
        # Process single transaction
        self.transaction_count += 1
        for item in items:
            self.item_tids[item].add(tid)
            self.item_counts[item] += 1
    
    def get_support_from_tids(self, tids):
        # Calculate support from TID set
        return len(tids) / self.transaction_count
    
    def get_support(self, items):
        # Calculate support for itemset
        if not items:
            return 0
        # Calculate TID intersection
        # Ensure all items in 'items' are present in self.item_tids to avoid KeyError
        common_tids = set.intersection(*(self.item_tids[item] for item in items if item in self.item_tids))
        return self.get_support_from_tids(common_tids)
    
    def get_confidence(self, base_items, full_items):
        # Calculate confidence
        base_support = self.get_support(base_items)
        if base_support == 0:
            return 0
        return self.get_support(full_items) / base_support


def rarm(df, min_support=0.5, min_confidence=0.8, num_processes=None, file_type_for_limit=None, max_level_limit=None, itemset_limit=None, turn_counter=None, params_str=None,
         enable_dynamic_support=False, dynamic_support_threshold=500000, support_increment_factor=1.2):
    # Initialize RARM miner
    miner = RARMiner()
    
    # === MODIFIED: Process count handling ===
    # If num_processes is not provided, use all available cores.
    # Otherwise, use the number of processes passed as an argument.
    # This allows the caller (e.g., Main_Association_Rule) to control the parallelism.
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    # === END MODIFICATION ===

    # Updated debug log to include new parameters
    logger.info(f"    [Debug RARM Init] Initializing RARM. df shape: {df.shape}, min_support={min_support}, min_confidence={min_confidence}, num_processes={num_processes}, file_type_for_limit='{file_type_for_limit}', max_level_limit={max_level_limit}, itemset_limit={itemset_limit}, dynamic_support={enable_dynamic_support}")

    # --- NEW: Conditional logic for CICIoT2023 optimization ---
    if file_type_for_limit == 'CICIoT2023':
        logger.info("    [INFO] CICIoT2023 detected. Using optimized inverted index method for support calculation.")
        
        # --- OPTIMIZED INVERTED INDEX PATH ---
        
        # 1. Build Inverted Index (item_tids)
        logger.info("    [Debug RARM-Opt] Building inverted index from transactions...")
        for tid, row in enumerate(df.itertuples(index=False, name=None)):
            items = set(f"{col}={val}" for col, val in zip(df.columns, row) if pd.notna(val))
            miner.add_transaction(tid, items)
        
        logger.info(f"    [Debug RARM-Opt] Finished loading {miner.transaction_count} transactions into index.")
        
        # 2. Find frequent 1-itemsets
        frequent_items = {
            item for item, count in miner.item_counts.items()
            if count / miner.transaction_count >= min_support
        }
        logger.info(f"    [Debug RARM-Opt] Found {len(frequent_items)} frequent 1-items.")
        if not frequent_items:
            return [], 1

        # 3. Main loop for finding frequent itemsets (level by level)
        rule_set = set()
        current_level = {frozenset([item]) for item in frequent_items}
        level_count = 1
        final_max_level = 1
        max_itemset_size = len(df.columns) if not df.empty else len(frequent_items)

        while current_level and len(next(iter(current_level))) < max_itemset_size:
            if max_level_limit is not None and level_count > max_level_limit:
                logger.info(f"    [Debug RARM-Opt Loop-{level_count}] Reached max_level_limit ({max_level_limit}). Breaking loop.")
                break
            
            final_max_level = level_count
            current_itemset_size = len(next(iter(current_level)))
            logger.info(f"    [Debug RARM-Opt Loop-{level_count}] Processing itemsets of size: {current_itemset_size}. Num itemsets: {len(current_level)}")

            # Candidate Generation
            potential_next_level_candidates = set()
            for itemset in current_level:
                for item in frequent_items - itemset:
                    candidate = itemset | {item}
                    if len(candidate) == current_itemset_size + 1:
                        # Pruning step
                        is_valid_candidate = all(frozenset(subset) in current_level for subset in combinations(candidate, current_itemset_size))
                        if is_valid_candidate:
                            potential_next_level_candidates.add(candidate)
            
            logger.info(f"    [Debug RARM-Opt Loop-{level_count}] Generated {len(potential_next_level_candidates)} potential candidates.")
            if not potential_next_level_candidates:
                break
                
            # Parallel support calculation using the pre-built inverted index
            next_level_frequent_itemsets = set()
            logger.info(f"    [Debug RARM-Opt Loop-{level_count}] Calculating support for {len(potential_next_level_candidates)} candidates using {num_processes} processes...")
            
            with multiprocessing.Pool(processes=num_processes, initializer=_init_rarm_worker, initargs=(miner.item_tids, miner.transaction_count)) as pool:
                pbar_itemsets = tqdm(
                    pool.imap_unordered(calculate_support_for_candidate, potential_next_level_candidates, chunksize=1000),
                    total=len(potential_next_level_candidates),
                    desc=f"    [RARM-Opt Lvl-{level_count} Itemsets]"
                )
                for support, itemset_cand in pbar_itemsets:
                    if support >= min_support:
                        next_level_frequent_itemsets.add(itemset_cand)

            logger.info(f"    [Debug RARM-Opt Loop-{level_count}] Found {len(next_level_frequent_itemsets)} frequent itemsets.")

            if itemset_limit is not None and len(next_level_frequent_itemsets) > itemset_limit:
                logger.warning(f"    [WARN] Safety valve triggered. Exiting.")
                break

            # Rule Generation (remains the same)
            if next_level_frequent_itemsets:
                logger.info(f"    [Debug RARM-Opt Loop-{level_count}] Generating rules from {len(next_level_frequent_itemsets)} frequent itemsets...")
                rule_gen_tasks = [(itemset, min_confidence) for itemset in next_level_frequent_itemsets]
                with multiprocessing.Pool(processes=num_processes, initializer=_init_rarm_worker, initargs=(miner.item_tids, miner.transaction_count)) as pool:
                    pbar_rules = tqdm(pool.imap_unordered(_rarm_rule_worker_wrapper, rule_gen_tasks, chunksize=1000), total=len(rule_gen_tasks), desc=f"    [RARM-Opt Lvl-{level_count} Rules]")
                    for rules_from_one_itemset in pbar_rules:
                        for antecedent, consequent, confidence, support in rules_from_one_itemset:
                            rule_dict = {}
                            full_itemset_for_dict = antecedent.union(consequent)
                            for item_str in full_itemset_for_dict:
                                key, value_str = item_str.split('=', 1)
                                try:
                                    val_float = float(value_str)
                                    rule_dict[key] = int(val_float) if val_float.is_integer() else val_float
                                except ValueError:
                                    rule_dict[key] = value_str
                            rule_tuple = tuple(sorted(rule_dict.items()))
                            rule_set.add(rule_tuple)
            
            # --- NEW: Save generated rules ---
            if turn_counter is not None and params_str is not None and next_level_frequent_itemsets:
                # In append mode, we only want to save the rules generated at THIS level.
                # The rule_set contains all rules so far, so we need to find the new ones.
                new_rules_this_level = []
                for antecedent, consequent, confidence, support in rules_from_one_itemset:
                    rule_dict = {}
                    full_itemset_for_dict = antecedent.union(consequent)
                    for item_str in full_itemset_for_dict:
                        key, value_str = item_str.split('=', 1)
                        try:
                            val_float = float(value_str)
                            rule_dict[key] = int(val_float) if val_float.is_integer() else val_float
                        except ValueError:
                            rule_dict[key] = value_str
                    new_rules_this_level.append(rule_dict)

                if new_rules_this_level:
                    save_association_artifacts(new_rules_this_level, file_type_for_limit, turn_counter, level_count, 'generated_rules.log', params_str)

            if not next_level_frequent_itemsets:
                break
            current_level = next_level_frequent_itemsets
            level_count += 1
        
        logger.info(f"    [Debug RARM-Opt Finish] Processing finished. Total rules: {len(rule_set)}. Max level: {final_max_level}")
        return [dict(rule) for rule in rule_set], final_max_level

    else:
        # --- ORIGINAL RARM LOGIC FOR ALL OTHER DATASETS ---
        logger.info("    [INFO] Using original RARM logic for this dataset.")
        
        # --- Re-enabled original streaming approach for performance comparison ---
        # Convert data and build initial structure (streaming approach)
        for tid, row in enumerate(df.itertuples(index=False, name=None)):
            items = set(f"{col}={val}" for col, val in zip(df.columns, row) if pd.notna(val))
            miner.add_transaction(tid, items)
        
        '''        
        # --- PERFORMANCE OPTIMIZATION: Vectorized Data Transformation ---
        # The original row-by-row iteration with f-string formatting is a major bottleneck
        # for large dataframes. This new approach vectorizes the transformation.
        
        # 1. Create a dataframe of column names, repeated for each row.
        #    This avoids doing slow string operations inside a Python loop.
        cols_df = pd.DataFrame([df.columns.values] * len(df), index=df.index, columns=df.columns)
        
        # 2. Convert the entire data dataframe to string type at once.
        #    This is significantly faster than converting cell by cell.
        data_str_df = df.astype(str)
        
        # 3. Concatenate the column names with the string data, using '=' as a separator.
        #    This is a fast, vectorized string operation.
        item_df = cols_df + '=' + data_str_df
        
        # 4. Use a fast method to convert each row into a set of items, ignoring NaNs.
        #    The 'unstack' method is a highly efficient way to get a series of all items.
        #    We create a MultiIndex to track (row_index, column_name) and then filter NaNs.
        #    The original `if pd.notna(val)` is handled by `dropna()`.
        all_items = item_df.unstack().dropna()
        # --- FIX: Group by level 1 (original row index) instead of level 0 (column name) ---
        transactions = all_items.groupby(level=1).apply(set)
        
        # 5. Load the processed transactions into the miner.
        for tid, items in transactions.items():
            miner.add_transaction(tid, items)
        '''

        # --- END OPTIMIZATION ---

        logger.info(f"    [Debug RARM DataLoad] Finished loading {miner.transaction_count} transactions.")

        # --- DYNAMIC SUPPORT: Initialize local_min_support ---
        local_min_support = min_support
        
        # --- FIX: Use correct support calculation method ---
        # The denominator for support must be the total number of rows in the original dataframe.
        total_transactions = len(df)

        # Find frequent 1-itemset (items with minimum support)
        # The numerator for support is the number of transactions an item appears in,
        # which is the length of its TID set in the miner.
        frequent_items = {
            item for item, tids in miner.item_tids.items()
            if len(tids) / total_transactions >= local_min_support
        }
        # --- END FIX ---

        logger.info(f"    [Debug RARM Freq1] Found {len(frequent_items)} frequent 1-items.")
        if not frequent_items:
            logger.info("    [Debug RARM Freq1] No frequent 1-items found. Returning empty list.")
            return [], 1 # Return empty list and 1 for final_max_level

        # Set for rule storage
        rule_set = set()
        
        # Process level by level (memory efficient)
        current_level = {frozenset([item]) for item in frequent_items}
        
        level_count = 1
        final_max_level = 1
        # Limit max itemset size to avoid excessively long runs if frequent_items is huge.
        # This limit can be df.shape[1] (number of columns) or a practical limit.
        max_itemset_size = len(df.columns) if not df.empty else len(frequent_items)

        while current_level and len(next(iter(current_level))) < max_itemset_size:
            # --- OPTIMIZATION: Check level limit at the BEGINNING of the loop ---
            # This prevents doing all the hard work for a level that will be discarded.
            if max_level_limit is not None and level_count > max_level_limit:
                logger.info(f"    [Debug RARM Loop-{level_count}] Reached max_level_limit ({max_level_limit}). Breaking RARM loop.")
                break
            # --- END OPTIMIZATION ---

            final_max_level = level_count
            
            current_itemset_size = len(next(iter(current_level)))
            logger.info(f"    [Debug RARM Loop-{level_count}] Processing itemsets of size: {current_itemset_size}. Num itemsets in current_level: {len(current_level)}")
            
            # Generate all potential next_level candidates first
            potential_next_level_candidates = set()
            if not frequent_items or not current_level: # Safety check
                break

            # Candidate Generation (Apriori-gen like from RARM's perspective)
            # itemset is k, item is 1, candidate is k+1
            for itemset in current_level:
                # Only try to extend with items that are 'larger' than any item in itemset (lexicographical or other consistent order)
                # to avoid duplicate candidates like {A,B} and {B,A}.
                # Or, more simply, ensure `item` is not already in `itemset`.
                # The original `frequent_items - itemset` handles this.
                for item in frequent_items - itemset: 
                    candidate = itemset | {item}
                    if len(candidate) == current_itemset_size + 1:
                        subsets_are_frequent = True
                        if current_itemset_size > 0: # Check subsets only if k > 0 (i.e., candidate size > 1)
                            for subset_to_check in combinations(candidate, current_itemset_size):
                                if frozenset(subset_to_check) not in current_level:
                                    subsets_are_frequent = False
                                    break
                        if subsets_are_frequent:
                            potential_next_level_candidates.add(candidate)
            
            logger.info(f"    [Debug RARM Loop-{level_count}] Generated {len(potential_next_level_candidates)} potential candidates for next level.")
            if not potential_next_level_candidates:
                logger.info(f"    [Debug RARM Loop-{level_count}] No potential candidates generated. Breaking loop.")
                break

            # --- DYNAMIC SUPPORT: Loop for this level ---
            while True:
                # Parallel support calculation for candidates
                next_level_frequent_itemsets = set()
                
                if potential_next_level_candidates:
                    logger.info(f"    [Debug RARM Loop-{level_count}] Calculating support for {len(potential_next_level_candidates)} candidates using {num_processes} processes (support={local_min_support:.4f})...")
                    with multiprocessing.Pool(
                        processes=num_processes,
                        initializer=_init_rarm_worker,
                        initargs=(miner.item_tids, total_transactions)
                    ) as pool:
                        pbar_itemsets = tqdm(
                            pool.imap_unordered(calculate_support_for_candidate, potential_next_level_candidates, chunksize=1000),
                            total=len(potential_next_level_candidates),
                            desc=f"    [RARM Lvl-{level_count} Itemsets]"
                        )
                        
                        for support, itemset_cand in pbar_itemsets:
                            if support >= local_min_support:
                                next_level_frequent_itemsets.add(itemset_cand)

                logger.info(f"    [Debug RARM Loop-{level_count}] Found {len(next_level_frequent_itemsets)} frequent itemsets for the next level.")
                
                # --- DYNAMIC SUPPORT: Check and adjust ---
                if enable_dynamic_support and len(next_level_frequent_itemsets) > dynamic_support_threshold:
                    logger.warning(f"    [DYNAMIC SUPPORT] At level {level_count}, itemset count ({len(next_level_frequent_itemsets)}) exceeded threshold ({dynamic_support_threshold}).")
                    local_min_support *= support_increment_factor
                    logger.warning(f"    [DYNAMIC SUPPORT] Increasing min_support to {local_min_support:.4f} and retrying level {level_count} support calculation.")
                    
                    # Clear frequent items from this attempt and retry the support calculation
                    next_level_frequent_itemsets.clear()
                    continue # This will re-run the support calculation with the new local_min_support
                else:
                    # The number of itemsets is acceptable, so we can break out of the dynamic support loop for this level.
                    break

            # --- DYNAMIC SUPPORT: End of loop for this level ---
            
            # --- Save frequent itemsets ---
            if turn_counter is not None and params_str is not None:
                itemsets_to_save = [dict(zip([f"item_{i+1}" for i in range(len(fs))], fs)) for fs in next_level_frequent_itemsets]
                save_association_artifacts(itemsets_to_save, file_type_for_limit, turn_counter, level_count + 1, 'frequent_itemsets.log', params_str)

            # --- RESTORED: Safety Valve Check ---
            if itemset_limit is not None and len(next_level_frequent_itemsets) > itemset_limit:
                logger.warning(f"    [WARN] Safety valve triggered: Number of frequent itemsets ({len(next_level_frequent_itemsets)}) exceeds the limit ({itemset_limit}).")
                logger.warning("    Stopping rule generation for this level to prevent memory overflow.")
                logger.warning("    Consider increasing min_support or min_confidence.")
                break 

            # Rule generation from newly found frequent itemsets (next_level_frequent_itemsets)
            if next_level_frequent_itemsets:
                logger.info(f"    [Debug RARM Loop-{level_count}] Generating rules from {len(next_level_frequent_itemsets)} frequent itemsets using {num_processes} processes for rule generation...")
                #'''
                # nds to the '[RARM Lvl-X Rules]' progress bar.
                rule_gen_processes = num_processes
                if file_type_for_limit in ['CICIoT2023', 'CICIoT'] and level_count >= 2:
                    rule_gen_processes = min(32, num_processes)
                    logger.info(f"    [Debug RARM Loop-{level_count}] [MEMORY OPTIMIZATION] Limiting rule generation processes to {rule_gen_processes} for '{file_type_for_limit}'.")
                else:
                    logger.info(f"    [Debug RARM Loop-{level_count}] Generating rules from {len(next_level_frequent_itemsets)} frequent itemsets using {rule_gen_processes} processes for rule generation...")
                #'''
                
                # MODIFIED: Task list no longer includes the bound method.
                rule_gen_tasks = [
                    (itemset, min_confidence) 
                    for itemset in next_level_frequent_itemsets
                ]
                if rule_gen_tasks:
                    # MODIFIED: Pool is created with an initializer to safely share read-only data with workers.
                    with multiprocessing.Pool(
                        processes=num_processes,
                        initializer=_init_rarm_worker,
                        initargs=(miner.item_tids, total_transactions)
                    ) as pool:
                        # MODIFIED: Consume the iterator *inside* the 'with' block to prevent deadlocks.
                        # ADDED: Progress bar for rule generation
                        pbar_rules = tqdm(
                            pool.imap_unordered(
                                _rarm_rule_worker_wrapper, 
                                rule_gen_tasks,
                                chunksize=1000
                            ),
                            total=len(rule_gen_tasks),
                            desc=f"    [RARM Lvl-{level_count} Rules]"
                        )

                        # for rules_from_one_itemset in results_iterator:
                        for rules_from_one_itemset in pbar_rules:
                            # --- MODIFIED: Logic to save rules incrementally ---
                            new_rules_this_level = []
                            for antecedent, consequent, confidence, support in rules_from_one_itemset:
                                # Convert to the required dictionary format for the final output
                                rule_dict = {}
                                full_itemset_for_dict = antecedent.union(consequent)
                                for item_str in full_itemset_for_dict:
                                    key, value_str = item_str.split('=', 1)
                                    try:
                                        val_float = float(value_str)
                                        rule_dict[key] = int(val_float) if val_float.is_integer() else val_float
                                    except ValueError:
                                        rule_dict[key] = value_str
                                
                                rule_tuple = tuple(sorted(rule_dict.items()))
                                # Add to the main rule set and also to the list for this level's log
                                if rule_tuple not in rule_set:
                                    rule_set.add(rule_tuple)
                                    new_rules_this_level.append(rule_dict)

                            # Save the newly generated rules for this itemset to the log file
                            if turn_counter is not None and params_str is not None and new_rules_this_level:
                                save_association_artifacts(new_rules_this_level, file_type_for_limit, turn_counter, level_count, 'generated_rules.log', params_str)

                logger.info(f"    [Debug RARM Loop-{level_count}] Rule set size after processing level {level_count}: {len(rule_set)}")

            # Prepare for the next level
            if not next_level_frequent_itemsets:
                logger.info(f"    [Debug RARM Loop-{level_count}] No more frequent itemsets found. Breaking loop.")
                break
            current_level = next_level_frequent_itemsets # Move to next level
            level_count += 1
        
        logger.info(f"    [Debug RARM Finish] RARM processing finished. Total rules found: {len(rule_set)}. Max level reached: {final_max_level}")
        # Convert results
        return [dict(rule) for rule in rule_set], final_max_level
