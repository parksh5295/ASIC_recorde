# Algorithm: SaM (Split and Merge)
# Output: Association rules Dictionary List; [{'feature1': value1, 'feature2': value2, ...}, {...}, ...]

from collections import defaultdict
from itertools import combinations
import pandas as pd # For pd.Timestamp
import multiprocessing  
from tqdm import tqdm

# Module Imports
# --- Import for saving artifacts ---
from Dataset_Choose_Rule.isv_save_log import save_association_artifacts


class ChunkProcessor:
    def __init__(self, min_support):
        self.transaction_count = 0
        self.item_counts = defaultdict(int)
        self.min_support = min_support
        
    def process_transaction(self, transaction):
        # Process single transaction
        self.transaction_count += 1
        for item in transaction:
            self.item_counts[item] += 1
    
    def get_frequent_items(self):
        # Return items with minimum support
        return {item for item, count in self.item_counts.items() 
               if self.transaction_count > 0 and count / self.transaction_count >= self.min_support} # Add check for self.transaction_count > 0


class SaMiner:
    def __init__(self, min_support, chunk_size=1000):
        self.min_support = min_support
        self.chunk_size = chunk_size
        self.transaction_count = 0
        self.item_tids = defaultdict(set)
    
    def get_support(self, items):
        # Calculate support using TID intersection
        if not items:
            return 0
        common_tids = set.intersection(*(self.item_tids[item] for item in items))
        return len(common_tids) / self.transaction_count
    
    def process_chunk(self, transactions):
        # Process chunk
        processor = ChunkProcessor(self.min_support)
        
        # Process transactions in chunk
        for transaction in transactions:
            processor.process_transaction(transaction)
        
        return processor.get_frequent_items()


# Helper function for parallel chunk processing in SaM
def process_single_chunk_sam(chunk_df_as_list_of_transactions, min_support_for_chunk):
    # This function processes a single chunk and returns its local frequent items.
    # It encapsulates the logic of SaMiner.process_chunk or ChunkProcessor directly.
    if not chunk_df_as_list_of_transactions:
        return set()
    
    # Using ChunkProcessor directly for simplicity in parallel worker
    chunk_processor = ChunkProcessor(min_support_for_chunk)
    for transaction in chunk_df_as_list_of_transactions:
        chunk_processor.process_transaction(transaction)
    return chunk_processor.get_frequent_items()

# Helper function for parallel global support calculation in SaM Merge phase
# MODIFIED: This function now uses global variables for TID map and transaction count,
# set by the pool initializer, to avoid passing large objects for every task.
def calculate_global_support_sam(item_set):
    if not item_set:
        return 0, item_set
    
    # Use the globally-scoped variables initialized in each worker process
    item_tids_global = _GLOBAL_ITEM_TIDS
    total_transactions_global = _GLOBAL_TOTAL_TX
    
    # Ensure all items in item_set are in item_tids_global to prevent KeyError
    if not all(item in item_tids_global for item in item_set):
        # This case should ideally not happen if item_set comes from all_frequent_items_from_chunks
        # which are composed of valid items from the dataset.
        # However, as a safeguard:
        # print(f"Warning: Item in {item_set} not found in global TIDs. Skipping support calc.")
        return 0, item_set 
        
    common_tids = set.intersection(*(item_tids_global[item] for item in item_set))
    support = len(common_tids) / total_transactions_global if total_transactions_global > 0 else 0
    return support, item_set

# NEW: Worker initializer and globals for parallel rule generation
_GLOBAL_ITEM_TIDS = None
_GLOBAL_TOTAL_TX = 0

def _init_sam_worker(item_tids, total_tx):
    """Initializes global variables for a worker process."""
    global _GLOBAL_ITEM_TIDS, _GLOBAL_TOTAL_TX
    _GLOBAL_ITEM_TIDS = item_tids
    _GLOBAL_TOTAL_TX = total_tx

def _support_from_globals(items):
    """Calculates support using global TID map. For use in worker processes."""
    if not items or not _GLOBAL_ITEM_TIDS or _GLOBAL_TOTAL_TX == 0:
        return 0.0
    
    # Safeguard: ensure all items are in the map to prevent KeyErrors
    if not all(item in _GLOBAL_ITEM_TIDS for item in items):
        return 0.0
        
    common_tids = set.intersection(*(_GLOBAL_ITEM_TIDS[item] for item in items))
    return len(common_tids) / _GLOBAL_TOTAL_TX

# NEW: Wrapper function for imap to handle multiple arguments for the rule generation task.
def _sam_rule_worker_wrapper(args):
    """Helper to unpack arguments for pool.imap_unordered."""
    return generate_sam_rules_for_itemset_task(*args)


# MODIFIED: Helper function for parallel rule generation for SaM.
# It now uses global variables set by the initializer instead of receiving a function.
def generate_sam_rules_for_itemset_task(f_itemset, min_conf):
    found_strong_rule = False
    if len(f_itemset) > 1:
        # We need the support of the full itemset (f_itemset) for confidence calculation.
        support_f_itemset = _support_from_globals(f_itemset)

        if support_f_itemset == 0: # Should not happen for a frequent itemset found globally
            return None

        for i in range(1, len(f_itemset)):
            for antecedent_tuple in combinations(f_itemset, i):
                antecedent = frozenset( antecedent_tuple)
                support_antecedent = _support_from_globals(antecedent)
                confidence = 0
                if support_antecedent > 0:
                    confidence = support_f_itemset / support_antecedent
                
                if confidence >= min_conf:
                    found_strong_rule = True
                    break # Found one strong rule for this f_itemset
            if found_strong_rule:
                break
    
    return f_itemset if found_strong_rule else None

def sam(df, min_support=0.5, min_confidence=0.8, num_processes=None, file_type_for_limit=None, max_level_limit=None, itemset_limit=None, turn_counter=None, params_str=None,
        enable_dynamic_support=False, dynamic_support_threshold=400000, support_increment_factor=1.2, **kwargs):
    # === MODIFIED: Process count handling ===
    # If num_processes is not provided, use all available cores.
    # Otherwise, use the number of processes passed as an argument.
    # This allows the caller (e.g., Main_Association_Rule) to control the parallelism.
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    # === END MODIFICATION ===
    
    print(f"    [Debug SaM Init] Algorithm: SaM (Split and Merge), Input df shape: {df.shape}, min_support={min_support}, min_confidence={min_confidence}, num_processes={num_processes}, itemset_limit={itemset_limit}, file_type='{file_type_for_limit}'")
    start_time_total = pd.Timestamp.now()

    # --- NEW: Conditional logic for CICIoT2023 optimization ---
    if file_type_for_limit == 'CICIoT2023':
        print("    [INFO] CICIoT2023 detected. Using optimized inverted index method for SaM.")
        miner = SaMiner(min_support)

        # 1. Build Inverted Index (item_tids) - This is the core of the optimization
        print("    [Debug SaM-Opt] Building inverted index from transactions...")
        for tid, row in enumerate(df.itertuples(index=False, name=None)):
            transaction = set(f"{col}={val}" for col, val in zip(df.columns, row) if pd.notna(val))
            miner.transaction_count += 1
            for item in transaction:
                miner.item_tids[item].add(tid)
        print(f"    [Debug SaM-Opt] Finished loading {miner.transaction_count} transactions into index.")

        # 2. Find frequent 1-itemsets directly from the full index
        frequent_1_items_support = {}
        with multiprocessing.Pool(processes=num_processes, initializer=_init_sam_worker, initargs=(miner.item_tids, miner.transaction_count)) as pool:
            tasks = [frozenset([item]) for item in miner.item_tids.keys()]
            pbar_l1 = tqdm(pool.imap_unordered(calculate_global_support_sam, tasks, chunksize=1000), total=len(tasks), desc="    [SaM-Opt L1 Support]")
            for support, item_fset in pbar_l1:
                if support >= min_support:
                    frequent_1_items_support[list(item_fset)[0]] = support
        
        globally_frequent_1_itemsets_tuples = {frozenset([item]) for item in frequent_1_items_support.keys()}
        print(f"    [Debug SaM-Opt] Found {len(globally_frequent_1_itemsets_tuples)} globally frequent 1-itemsets.")

        if not globally_frequent_1_itemsets_tuples:
            return [], 1

        # 3. Main loop for finding frequent itemsets (level by level)
        rule_set = set()
        current_level_itemsets = globally_frequent_1_itemsets_tuples
        level_count = 1
        final_max_level = 1
        
        while current_level_itemsets:
            if max_level_limit is not None and level_count > max_level_limit:
                print(f"    [Debug SaM-Opt Loop-{level_count}] Reached max_level_limit. Breaking loop.")
                break
            
            final_max_level = level_count
            current_itemset_size = len(next(iter(current_level_itemsets)))
            print(f"    [Debug SaM-Opt Loop-{level_count}] Processing {len(current_level_itemsets)} itemsets of size: {current_itemset_size}.")

            # Rule Generation from current level (parallelized)
            print(f"    [Debug SaM-Opt Loop-{level_count}] Generating rules...")
            rule_gen_tasks = [(itemset, min_confidence) for itemset in current_level_itemsets]
            with multiprocessing.Pool(processes=num_processes, initializer=_init_sam_worker, initargs=(miner.item_tids, miner.transaction_count)) as pool:
                pbar_rules = tqdm(pool.imap_unordered(_sam_rule_worker_wrapper, rule_gen_tasks, chunksize=1000), total=len(rule_gen_tasks), desc=f"    [SaM-Opt Lvl-{level_count} Rules]")
                
                # --- NEW: Logic to save rules incrementally ---
                new_rules_this_level = []
                for f_itemset_with_strong_rule in pbar_rules:
                    if f_itemset_with_strong_rule:
                        rule_dict = {}
                        for item_str in f_itemset_with_strong_rule:
                            key, value_str = item_str.split('=', 1)
                            try:
                                val_float = float(value_str)
                                rule_dict[key] = int(val_float) if val_float.is_integer() else val_float
                            except ValueError:
                                rule_dict[key] = value_str
                        rule_tuple = tuple(sorted(rule_dict.items()))
                        if rule_tuple not in rule_set:
                            rule_set.add(rule_tuple)
                            new_rules_this_level.append(rule_dict)

                # Save the newly generated rules for this level to the log file
                if turn_counter is not None and params_str is not None and new_rules_this_level:
                    save_association_artifacts(new_rules_this_level, file_type_for_limit, turn_counter, level_count, 'generated_rules.log', params_str)


            # Candidate Generation for the next level
            potential_next_level_candidates = set()
            for itemset1 in current_level_itemsets:
                for itemset2 in current_level_itemsets:
                    if itemset1 != itemset2:
                        candidate = itemset1.union(itemset2)
                        if len(candidate) == current_itemset_size + 1:
                            # Pruning
                            if all(frozenset(subset) in current_level_itemsets for subset in combinations(candidate, current_itemset_size)):
                                potential_next_level_candidates.add(candidate)
            
            print(f"    [Debug SaM-Opt Loop-{level_count}] Generated {len(potential_next_level_candidates)} potential candidates for next level.")
            if not potential_next_level_candidates:
                break
            
            if itemset_limit is not None and len(potential_next_level_candidates) > itemset_limit:
                print(f"    [WARN] Safety valve triggered. Exiting.")
                break

            # Parallel support calculation for candidates
            next_level_frequent_itemsets = set()
            with multiprocessing.Pool(processes=num_processes, initializer=_init_sam_worker, initargs=(miner.item_tids, miner.transaction_count)) as pool:
                pbar_candidates = tqdm(pool.imap_unordered(calculate_global_support_sam, potential_next_level_candidates, chunksize=1000), total=len(potential_next_level_candidates), desc=f"    [SaM-Opt Lvl-{level_count+1} Candidates]")
                for support, item_fset_cand in pbar_candidates:
                    if support >= min_support:
                        next_level_frequent_itemsets.add(item_fset_cand)

            if not next_level_frequent_itemsets:
                break
            
            # --- NEW: Save frequent itemsets ---
            if turn_counter is not None and params_str is not None:
                itemsets_to_save = [dict(zip([f"item_{i+1}" for i in range(len(fs))], fs)) for fs in next_level_frequent_itemsets]
                save_association_artifacts(itemsets_to_save, file_type_for_limit, turn_counter, level_count + 1, 'frequent_itemsets.log', params_str)

            current_level_itemsets = next_level_frequent_itemsets
            level_count += 1
            
        total_duration = (pd.Timestamp.now() - start_time_total).total_seconds()
        print(f"    [Debug SaM-Opt Finish] Processing finished. Total rules: {len(rule_set)}. Max level: {final_max_level}. Total time: {total_duration:.2f}s")
        return [dict(rule) for rule in rule_set], final_max_level

    else:
        # --- ORIGINAL SAM LOGIC FOR ALL OTHER DATASETS ---
        print("    [INFO] Using original SaM logic for this dataset.")
        # Initialize
        miner = SaMiner(min_support)
        #chunk_size = max(1, len(df) // 4)  # Control memory usage - User may set specific chunk_size via SaMiner if extended
        # For now, using a fixed or dynamic chunk_size as per original SaM logic if it had one, or a reasonable default.
        # The original code seems to calculate it: len(df) // 4. Let's assume this is intended unless SaMiner allows external chunk_size.
        # If SaMiner is only initialized with min_support, chunk_size might be defined in SaMiner or fixed here.
        # The provided SaMiner class __init__ takes min_support and chunk_size, defaulting to 1000.
        # However, the sam function re-calculates chunk_size. Let's use the one in sam function for logging.
        calculated_chunk_size = max(1, len(df) // 4) if len(df) > 0 else 1
        # If miner.chunk_size is used by process_chunk, we should ensure it's set or use calculated_chunk_size.
        # For logging, we use calculated_chunk_size used for splitting df.
        print(f"    [Debug SaM Init] Calculated chunk_size for splitting: {calculated_chunk_size}")

        
        # Build TID mapping (streaming approach)
        print(f"    [Debug SaM TIDMap] Building TID map...")
        start_time_tidmap = pd.Timestamp.now()
        for tid, row in enumerate(df.itertuples(index=False, name=None)):
            transaction = set(f"{col}={val}" for col, val in zip(df.columns, row))
            miner.transaction_count += 1
            for item in transaction:
                miner.item_tids[item].add(tid)
        tidmap_duration = (pd.Timestamp.now() - start_time_tidmap).total_seconds()
        print(f"    [Debug SaM TIDMap] TID map built. Total transactions: {miner.transaction_count}. Time: {tidmap_duration:.2f}s")
        
        # Split step: Process chunks
        print(f"    [Debug SaM SplitPhase] Starting Split Phase: Processing chunks...")
        all_frequent_items_from_chunks = set() # Renamed for clarity
        # Use calculated_chunk_size for splitting df here
        chunks = [df.iloc[i:i + calculated_chunk_size] for i in range(0, len(df), calculated_chunk_size)]
        print(f"    [Debug SaM SplitPhase] Data split into {len(chunks)} chunks.")

        # Parallelize the chunk processing part of the Split Phase
        split_phase_tasks = []
        for chunk_df in chunks:
            if not chunk_df.empty:
                transactions_in_chunk = [
                    set(f"{col}={val}" for col, val in zip(chunk_df.columns, row))
                    for row in chunk_df.itertuples(index=False, name=None)
                ]
                # Pass min_support (which is global min_support for SaM's local frequent item finding)
                split_phase_tasks.append((transactions_in_chunk, miner.min_support)) 
        
        if split_phase_tasks:
            print(f"    [Debug SaM SplitPhase] Processing {len(split_phase_tasks)} chunks in parallel using {num_processes} processes...")
            with multiprocessing.Pool(processes=num_processes) as pool:
                # Results will be a list of sets, where each set contains frequent items from one chunk
                chunk_results = pool.starmap(process_single_chunk_sam, split_phase_tasks)
            
                for frequent_items_in_single_chunk in chunk_results:
                    all_frequent_items_from_chunks.update(frequent_items_in_single_chunk)
            print(f"    [Debug SaM SplitPhase] Finished parallel chunk processing. Total unique frequent items from all chunks: {len(all_frequent_items_from_chunks)}")
        else:
            print("    [Debug SaM SplitPhase] No non-empty chunks to process.")

        print(f"    [Debug SaM SplitPhase] Split Phase finished. Total unique frequent items found across all chunks: {len(all_frequent_items_from_chunks)}.")
        if not all_frequent_items_from_chunks:
            print("    [Debug SaM MergePhase] No frequent items found after Split Phase. Returning empty list.")
            return [], 1 # Return empty list and 1 for final_max_level
        
        # Merge step: Create global frequent itemset
        print(f"    [Debug SaM MergePhase] Starting Merge Phase: Generating globally frequent 1-itemsets from chunk results...")
        rule_set = set()
        
        # Parallelize the global support check for 1-itemsets from chunks
        globally_frequent_1_itemsets_candidates = list(all_frequent_items_from_chunks) # Convert to list for tasks
        globally_frequent_1_itemsets_tuples = set() # Store as frozenset([item])
        
        # MODIFIED: Task list no longer includes the large item_tids and transaction_count objects.
        merge_phase_L1_tasks = [
            (frozenset([item]),) 
            for item in globally_frequent_1_itemsets_candidates
        ]

        if merge_phase_L1_tasks:
            print(f"    [Debug SaM MergePhase] Checking global support for {len(merge_phase_L1_tasks)} L1 candidates using {num_processes} processes...")
            # MODIFIED: Pool is created with an initializer to safely share read-only data with workers.
            with multiprocessing.Pool(
                processes=num_processes,
                initializer=_init_sam_worker,
                initargs=(miner.item_tids, miner.transaction_count)
            ) as pool:
                # MODIFIED: Use imap_unordered and consume the iterator *inside* the 'with' block to prevent deadlocks.
                # Wrapper is not needed here since calculate_global_support_sam takes a single iterable argument.
                # The task list needs to be just the itemsets, not tuples.
                tasks = [frozenset([item]) for item in globally_frequent_1_itemsets_candidates]
                results_iterator = pool.imap_unordered(calculate_global_support_sam, tasks, chunksize=1000)

                for support, item_fset in results_iterator:
                    if support >= min_support:
                        globally_frequent_1_itemsets_tuples.add(item_fset)
        
        # Original sequential L1 filtering (for reference)
        # globally_frequent_1_itemsets = {frozenset([item]) for item in all_frequent_items_from_chunks \
        #                                   if miner.get_support({item}) >= min_support}
        current_level_itemsets = globally_frequent_1_itemsets_tuples # Use the parallel result

        print(f"    [Debug SaM MergePhase] Found {len(current_level_itemsets)} globally frequent 1-itemsets to start Merge Phase.")

        if not current_level_itemsets:
            print("    [Debug SaM MergePhase] No globally frequent 1-itemsets found. Returning empty list.")
            return [], 1 # Return empty list and 1 for final_max_level

        level_count = 1
        final_max_level = 1
        while current_level_itemsets:
            # --- OPTIMIZATION: Check level limit at the BEGINNING of the loop ---
            if max_level_limit is not None and level_count > max_level_limit:
                print(f"    [Debug SaM MergeLoop-{level_count}] Reached max_level_limit ({max_level_limit}). Breaking SaM loop.")
                break
            # --- END OPTIMIZATION ---

            final_max_level = level_count
            
            if not current_level_itemsets:
                print(f"    [Debug SaM MergeLoop-{level_count}] current_level_itemsets is empty. Breaking.")
                break
            current_itemset_size = len(next(iter(current_level_itemsets)))
            print(f"    [Debug SaM MergeLoop-{level_count}] Processing itemsets of size: {current_itemset_size}. Num itemsets in current_level: {len(current_level_itemsets)}")
            
            # Rule Generation from current_level_itemsets - NOW PARALLELIZED
            if current_level_itemsets: # Process rules if current_level is not empty
                print(f"    [Debug SaM MergeLoop-{level_count}] Generating rules from {len(current_level_itemsets)} (size {current_itemset_size}) globally frequent itemsets using {num_processes} processes...")
                # MODIFIED: Task list no longer includes the bound method.
                rule_gen_tasks_sam = [
                    (itemset, min_confidence) 
                    for itemset in current_level_itemsets
                ]
                if rule_gen_tasks_sam:
                    # MODIFIED: Pool is created with an initializer to set up worker-local global variables.
                    # This avoids pickling bound methods or large data structures for every task.
                    with multiprocessing.Pool(
                        processes=num_processes,
                        initializer=_init_sam_worker,
                        initargs=(miner.item_tids, miner.transaction_count)
                    ) as pool:
                        # MODIFIED: Use imap_unordered with a wrapper and consume inside the loop to prevent deadlocks.
                        results_iterator = pool.imap_unordered(
                            _sam_rule_worker_wrapper, 
                            rule_gen_tasks_sam,
                            chunksize=1000
                        )
                    
                        # ADDED: Progress bar for rule generation
                        pbar_rules = tqdm(
                            results_iterator,
                            total=len(rule_gen_tasks_sam),
                            desc=f"    [SaM Lvl-{level_count} Rules]"
                        )

                        # --- NEW: Logic to save rules incrementally ---
                        new_rules_this_level = []
                        for f_itemset_with_strong_rule in pbar_rules:
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
                                # Add to the main rule set and also to the list for this level's log
                                if rule_tuple not in rule_set:
                                    rule_set.add(rule_tuple)
                                    new_rules_this_level.append(rule_dict)

                        # Save the newly generated rules for this level to the log file
                        if turn_counter is not None and params_str is not None and new_rules_this_level:
                            save_association_artifacts(new_rules_this_level, file_type_for_limit, turn_counter, level_count, 'generated_rules.log', params_str)

                print(f"    [Debug SaM MergeLoop-{level_count}] Rule set size after processing level {current_itemset_size}: {len(rule_set)}")

            # Original sequential rule generation loop (now replaced by parallel version above)
            # print(f"    [Debug SaM MergeLoop-{level_count}] Generating rules from {len(current_level_itemsets)} (size {current_itemset_size}) globally frequent itemsets...")
            # for itemset_in_current_level in current_level_itemsets: ...
            #     if len(itemset_in_current_level) > 1: ...
            #         for i in range(1, len(itemset_in_current_level)): ...
            #             for antecedent_tuple in combinations(itemset_in_current_level, i): ...
            #                 if confidence >= min_confidence: ...
            #                     rule_set.add(rule_tuple) ...

            # Candidate generation part (Apriori-like) for the NEXT level
            next_level_candidates = set() # Moved this here, was conflictingly named with the output of parallel support calc before.
            
            # Candidate generation part (Apriori-like)
            # This part itself is complex to parallelize efficiently due to inter-dependencies (checking subsets in current_level_itemsets)
            # We will parallelize the support counting for the generated candidates.
            potential_merge_candidates = set()
            if current_itemset_size > 0:
                for itemset in current_level_itemsets: # itemset is k-itemset
                    # Extend with globally frequent 1-items not already in itemset
                    for single_item_fset_global in globally_frequent_1_itemsets_tuples: # These are frozenset([item_str])
                        if not single_item_fset_global.issubset(itemset): # Ensure item is not already in itemset
                            candidate = itemset.union(single_item_fset_global)
                            if len(candidate) == current_itemset_size + 1: # Formed a (k+1) candidate
                                all_subsets_frequent = True
                                for subset_tuple in combinations(candidate, current_itemset_size):
                                    if frozenset(subset_tuple) not in current_level_itemsets:
                                        all_subsets_frequent = False
                                        break
                                if all_subsets_frequent:
                                    potential_merge_candidates.add(candidate)
            
            # --- NEW: Safety Valve Check ---
            if itemset_limit is not None and len(potential_merge_candidates) > itemset_limit:
                print(f"    [WARN] Safety valve triggered: Number of merge candidates ({len(potential_merge_candidates)}) exceeds the limit ({itemset_limit}).")
                print("    Stopping candidate processing for this level to prevent memory overflow.")
                print("    Consider increasing min_support or min_confidence.")
                break

            # Parallel support calculation for these potential_merge_candidates
            next_level_itemsets_from_merge = set()
            # MODIFIED: Task list no longer includes the large item_tids and transaction_count objects.
            merge_phase_Lk_tasks = [
                cand for cand in potential_merge_candidates
            ]

            if merge_phase_Lk_tasks:
                print(f"    [Debug SaM MergeLoop-{level_count}] Checking global support for {len(merge_phase_Lk_tasks)} L{level_count+1} candidates using {num_processes} processes...")
                # MODIFIED: Pool is created with an initializer to safely share read-only data with workers.
                with multiprocessing.Pool(
                    processes=num_processes,
                    initializer=_init_sam_worker,
                    initargs=(miner.item_tids, miner.transaction_count)
                ) as pool:
                    # ADDED: Progress bar for candidate support calculation
                    pbar_candidates = tqdm(
                        pool.imap_unordered(calculate_global_support_sam, merge_phase_Lk_tasks, chunksize=1000),
                        total=len(merge_phase_Lk_tasks),
                        desc=f"    [SaM Lvl-{level_count} Candidates]"
                    )
                
                    for support, item_fset_cand in pbar_candidates:
                        if support >= min_support:
                            next_level_itemsets_from_merge.add(item_fset_cand)
                            # This verbose logging can be commented out to keep the progress bar clean
                            # if len(next_level_itemsets_from_merge) % 1000 == 0 and len(next_level_itemsets_from_merge) > 0:
                            #     print(f"        [Debug SaM MergeLoop-{level_count}] Found candidate for next level (count {len(next_level_itemsets_from_merge)}): {item_fset_cand}, Support: {support:.4f}")
        
            next_level_candidates = next_level_itemsets_from_merge # Assign to the variable used later

            # --- NEW: Save frequent itemsets ---
            if turn_counter is not None and params_str is not None:
                itemsets_to_save = [dict(zip([f"item_{i+1}" for i in range(len(fs))], fs)) for fs in next_level_candidates]
                save_association_artifacts(itemsets_to_save, file_type_for_limit, turn_counter, level_count + 1, 'frequent_itemsets.log', params_str)

            '''
            # Generate rules from the current frequent 'itemset' (from current_level_itemsets)
            # This rule generation can also be parallelized by chunking current_level_itemsets
            if len(itemset) > 1: # itemset here refers to the one from `for itemset_idx, itemset in enumerate(current_level_itemsets):`
                                 # This loop needs to be outside the candidate generation or refactored if rules are from next_level_candidates
                                 # The original code generated rules from `itemset` in `current_level_itemsets`.
                                 # Let's assume we generate rules from `current_level_itemsets` before moving to `next_level_candidates`.
                                 # This part is outside the candidate generation for loop, so `itemset` is from `current_level_itemsets`.
                # This rule generation loop for current_level_itemsets should be processed for *all* itemsets in current_level_itemsets.
                # It can be parallelized by splitting current_level_itemsets into chunks.
                # For now, keeping this part sequential after parallel candidate finding for simplicity of this edit step.
                # To parallelize rule generation: create tasks from current_level_itemsets and run a helper in parallel.
                for i in range(1, len(itemset)):
                    for antecedent_tuple in combinations(itemset, i):
                        antecedent = frozenset(antecedent_tuple)
                        # consequent = itemset - antecedent # Not directly used for rule_dict key
                                    
                                    ant_support = miner.get_support(antecedent)
                                    if ant_support > 0:
                        itemset_support = miner.get_support(itemset) # Get support of full itemset for confidence
                        confidence = itemset_support / ant_support
                                    
                                    if confidence >= min_confidence:
                                        rule_dict = {}
                        # The original code uses antecedent for rule_dict, this seems to be the standard.
                        for rule_item_str in antecedent:
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
                                #     print(f"          [Debug SaM MergeLoop-{level_count}] Added rule (total {len(rule_set)}): {rule_tuple}, Conf: {confidence:.4f}")
            '''
            
            print(f"    [Debug SaM MergeLoop-{level_count}] Finished processing level. Next_level candidates size: {len(next_level_candidates)}. Total rules so far: {len(rule_set)}")
            if not next_level_candidates:
                print(f"    [Debug SaM MergeLoop-{level_count}] Next_level_candidates is empty. Breaking loop.")
                break
            current_level_itemsets = next_level_candidates
            level_count +=1
        
        total_duration = (pd.Timestamp.now() - start_time_total).total_seconds()
        print(f"    [Debug SaM Finish] SaM processing finished. Total rules: {len(rule_set)}. Max level reached: {final_max_level}. Total time: {total_duration:.2f}s")
        
        return [dict(rule) for rule in rule_set], final_max_level
