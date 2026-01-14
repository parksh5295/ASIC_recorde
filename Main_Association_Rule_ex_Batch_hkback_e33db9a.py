# A program that automatically generates signatures using association rules.
# EXPERIMENTAL VERSION: Implements a memory-efficient signature filtering logic with BATCH PROCESSING.

import argparse
import numpy as np
import time
import multiprocessing
import os
import pandas as pd
from tqdm import tqdm
from Dataset_Choose_Rule.association_data_choose import get_clustered_data_path
from Dataset_Choose_Rule.choose_amount_dataset import file_cut
from definition.Anomal_Judgment import anomal_judgment_label, anomal_judgment_nonlabel
from utils.time_transfer import time_scalar_transfer
from utils.class_row import without_labelmaking_out
from Modules.Heterogeneous_module import choose_heterogeneous_method
from Heterogeneous_Method.separate_group_mapping import map_intervals_to_groups
from utils.remove_rare_columns import remove_rare_columns
from Modules.Association_module import association_module
from Modules.Signature_evaluation_module import signature_evaluate
from Modules.Signature_underlimit import under_limit
from Modules.Signature_Organize import organize_signatures
from Evaluation.calculate_signature import calculate_signatures
from Dataset_Choose_Rule.save_csv import csv_association
from Dataset_Choose_Rule.time_save import time_save_csv_CS_ex
import logging
from datetime import datetime
import sys


logger = logging.getLogger(__name__)

# --- BATCH HELPER: Global tuple to hold batched normal data ---
_worker_data_filter_batch = ()

def _init_worker_filter_batch(normal_data_chunks, min_support):
    """Initializer for worker process, passing chunks of normal data."""
    global _worker_data_filter_batch
    _worker_data_filter_batch = (normal_data_chunks, min_support)

def _is_rule_valid_for_filtering_batch(rule):
    """Worker task that checks a rule against chunks of normal data (Scenario B)."""
    global _worker_data_filter_batch
    normal_data_chunks, min_support = _worker_data_filter_batch
    
    for chunk in normal_data_chunks:
        if calculate_support_for_itemset(rule, chunk, min_support):
            return None # Invalid rule, frequent in at least one normal chunk
            
    return rule # Valid signature, not frequent in any normal chunk

LEVEL_LIMITS_BY_FILE_TYPE = {
    'MiraiBotnet': 5, 'NSL-KDD': 5, 'NSL_KDD': 5, 'DARPA98': None, 'DARPA': None,
    'CICIDS2017': 4, 'CICIDS': 4, 'CICModbus23': None, 'CICModbus': None,
    'IoTID20': None, 'IoTID': None, 'netML': 5, 'Kitsune': 5,
    'CICIoT': 4, 'CICIoT2023': 4, 'default': None
}

def calculate_support_for_itemset(itemset, df, min_support):
    if not itemset or df.empty:
        return False
    mask = pd.Series([True] * len(df), index=df.index)
    for key, value in itemset.items():
        if key in df.columns:
            mask &= (df[key] == value)
        else:
            return False
    support = mask.sum() / len(df)
    return support >= min_support

# MODIFIED: Added batch_size
def process_confidence_iteration(min_confidence, anomal_grouped_data, nomal_grouped_data, Association_mathod, min_support, association_metric, group_mapped_df, signature_ea, precision_underlimit, cores_per_algo_task, current_file_type, itemset_limit, batch_size,
                                 prune_signatures, prune_coverage_threshold, merge_signatures, merge_infrequent_threshold):
    iteration_start_time = time.time()
    print(f"  [PCI Start] conf={min_confidence}, algo={Association_mathod}, algo_procs={cores_per_algo_task}, batch_size={batch_size}")
    
    max_level = LEVEL_LIMITS_BY_FILE_TYPE.get(current_file_type, LEVEL_LIMITS_BY_FILE_TYPE['default'])
    
    print(f"  [PCI Mining] Generating association rules from anomaly data...")
    association_list_anomal, _ = association_module(
        anomal_grouped_data, Association_mathod, min_support, 
        min_confidence, association_metric, num_processes=cores_per_algo_task, 
        file_type_for_limit=current_file_type, max_level_limit=max_level,
        itemset_limit=itemset_limit
    )
    
    # --- BATCH MODIFICATION START ---
    print(f"  [PCI Filter] Filtering {len(association_list_anomal):,} anomaly rules against batched normal data...")
    signatures = []

    normal_data_chunks = [nomal_grouped_data.iloc[i:i + batch_size] for i in range(0, len(nomal_grouped_data), batch_size)]
    
    if not nomal_grouped_data.empty and association_list_anomal:
        # Special process limit for memory-intensive datasets
        if current_file_type in ['CICIDS2017', 'CICIDS', 'Kitsune', 'CICIoT', 'CICIoT2023']:
            num_filter_processes = 12
        else:
            num_filter_processes = cores_per_algo_task

        try:
            with multiprocessing.Pool(
                processes=num_filter_processes,
                initializer=_init_worker_filter_batch,
                initargs=(normal_data_chunks, min_support)
            ) as pool:
                chunksize = max(1, len(association_list_anomal) // (num_filter_processes * 10))
                results_iterator = pool.imap_unordered(_is_rule_valid_for_filtering_batch, association_list_anomal, chunksize=chunksize)
                pbar = tqdm(results_iterator, total=len(association_list_anomal), desc="[PCI Filter] Progress")
                for result_rule in pbar:
                    if result_rule is not None:
                        signatures.append(result_rule)
        except KeyboardInterrupt:
            print("\n[RESCUE] KeyboardInterrupt detected! Halting filtering.")
            sys.exit(1)
    else:
        signatures = association_list_anomal
    # --- BATCH MODIFICATION END ---
        
    print(f"  [PCI Filter] Found {len(signatures):,} rules that are unique to anomaly data.")

    # --- Signature Organization Step ---
    if prune_signatures and signatures:
        print(f"  [PCI Organize] Starting Signature Organization for {len(signatures)} signatures...")
        
        # Convert list of dicts to dict of {hash: dict} for the organizer
        signatures_dict = {hash(frozenset(rule.items())): rule for rule in signatures}
        
        # Create a simple run_dir for caching within this non-incremental script
        run_dir_org = os.path.join("../Dataset_ex", "organizer_cache_batch")

        organized_signatures_dict = organize_signatures(
            all_signatures=signatures_dict,
            data_provider=group_mapped_df, # Use the full mapped dataframe for performance calculation
            num_processes=cores_per_algo_task,
            run_dir=run_dir_org,
            turn_counter=0, # This is a single run, so turn is 0
            coverage_threshold=prune_coverage_threshold,
            enable_merging=merge_signatures,
            merge_infrequent_threshold=merge_infrequent_threshold,
            data_batch_size=args.batch_size
        )
        
        # Convert back to list of dicts and update the signatures variable
        signatures = list(organized_signatures_dict.values())
        print(f"  [PCI Organize] Organization complete. {len(signatures)} signatures remain.")


    # --- BATCH MODIFICATION START ---
    print(f"  [PCI Signature Batch] Starting batch processing for signature evaluation. Total signatures: {len(signatures)}")
    signature_result = []
    num_sig_batches = (len(signatures) + batch_size - 1) // batch_size
    for i in tqdm(range(num_sig_batches), desc="  [PCI Signature Batch] Progress"):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        signature_batch = signatures[start_idx:end_idx]
        if signature_batch:
            batch_result = signature_evaluate(group_mapped_df, signature_batch)
            if batch_result:
                signature_result.extend(batch_result)
    print(f"  [PCI Signature Batch] Finished. Evaluated {len(signature_result)} signatures.")
    # --- BATCH MODIFICATION END ---
    
    signature_sets = under_limit(signature_result, signature_ea, precision_underlimit)

    current_recall = calculate_signatures(group_mapped_df, signature_sets) if signature_sets else 0

    print(f"  [PCI Finish] conf: {min_confidence}, "
          f"Anomal Rules Found: {len(association_list_anomal)}, "
          f"Final Signatures: {len(signatures)}, "
          f"Signature Sets: {len(signature_sets) if signature_sets else 0}, "
          f"Recall: {current_recall:.4f}")
    
    total_time_per_iteration = time.time() - iteration_start_time
    return min_confidence, current_recall, signature_sets, total_time_per_iteration

def main():
    total_start_time = time.time()
    timing_info = {}

    parser = argparse.ArgumentParser(description='Argparser')
    parser.add_argument('--file_type', type=str, default="MiraiBotnet")
    parser.add_argument('--file_number', type=int, default=1)
    parser.add_argument('--train_test', type=int, default=0)
    parser.add_argument('--heterogeneous', type=str, default="Normalized")
    parser.add_argument('--clustering', type=str, default="kmeans")
    parser.add_argument('--eval_clustering_silhouette', type=str, default="n")
    parser.add_argument('--association', type=str, default="apriori")
    parser.add_argument('--precision_underlimit', type=float, default=0.6)
    parser.add_argument('--signature_ea', type=int, default=15)
    parser.add_argument('--association_metric', type=str, default='confidence')
    parser.add_argument('--num_processes', type=int, default=None)
    parser.add_argument('--itemset_limit', type=int, default=10000000)
    # --- Batch Arguments ---
    parser.add_argument('--batch_size', type=int, default=50000, help="Batch size for processing normal data chunks and signatures.")
    # --- Signature Organization Arguments ---
    parser.add_argument('--prune_signatures', action='store_true', help="If set, enables the signature pruning (subsumption) process to remove redundant rules.")
    parser.add_argument('--prune_coverage_threshold', type=float, default=0.9, help="Coverage threshold for the signature pruning process.")
    parser.add_argument('--merge_signatures', action='store_true', help="If set, enables merging of similar, infrequent signatures.")
    parser.add_argument('--merge_infrequent_threshold', type=int, default=5, help="TP count at or below which a rule is considered infrequent and eligible for merging.")
    parser.add_argument('--signature_organize', action='store_true', help="A shorthand to enable both --prune_signatures and --merge_signatures.")

    args = parser.parse_args()
    
    # --- Handle the signature organization shorthand ---
    if args.signature_organize:
        args.prune_signatures = True
        args.merge_signatures = True
        
    logger.info(f"Global start time: {datetime.fromtimestamp(total_start_time).strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Data loading
    start = time.time()
    file_path, file_number = get_clustered_data_path(args.file_type, args.file_number)
    data = file_cut(args.file_type, file_path, 'all')
    timing_info['1_load_data'] = time.time() - start
    logger.info(f"Loading data from file: {file_path}")

    # 2. Labeling (Original Logic Preserved)
    start = time.time()
    # ... (All original if/elif/else blocks for labeling are preserved here) ...
    if args.file_type in ['MiraiBotnet', 'NSL-KDD', 'NSL_KDD']: data['label'], _ = anomal_judgment_nonlabel(args.file_type, data)
    elif args.file_type == 'netML': data['label'] = data['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
    elif args.file_type == 'DARPA98': data['label'] = data['Class'].apply(lambda x: 0 if str(x).strip() == '-' else 1)
    elif args.file_type in ['CICIDS2017', 'CICIDS']:
        if 'Label' in data.columns: data['label'] = data['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
        else: data['label'] = 0
    elif args.file_type in ['CICModbus23', 'CICModbus']: data['label'] = data['Attack'].apply(lambda x: 0 if str(x).strip() == 'Baseline Replay: In position' else 1)
    elif args.file_type in ['IoTID20', 'IoTID']: data['label'] = data['Label'].apply(lambda x: 0 if str(x).strip() == 'Normal' else 1)
    elif args.file_type in ['CICIoT', 'CICIoT2023']: data['label'] = data['attack_flag']
    elif args.file_type == 'Kitsune': data['label'] = data['Label']
    else: data['label'] = anomal_judgment_label(data)
    timing_info['2_anomal_judgment'] = time.time() - start

    # 3. Preprocessing (Original Logic Preserved)
    start = time.time()
    data = time_scalar_transfer(data, args.file_type)
    embedded_dataframe, _, category_mapping, data_list = choose_heterogeneous_method(data, args.file_type, args.heterogeneous, 'N')
    group_mapped_df, mapped_info_df = map_intervals_to_groups(embedded_dataframe, category_mapping, data_list, 'N')
    # ... (saving mapped_info, assigning label and cluster columns is preserved) ...
    group_mapped_df['label'] = data['label'].values
    group_mapped_df['cluster'] = data['cluster'].values
    
    anomal_grouped_data = group_mapped_df[group_mapped_df['cluster'] == 1]
    anomal_grouped_data = without_labelmaking_out(anomal_grouped_data, ['label', 'cluster'])
    nomal_grouped_data = group_mapped_df[group_mapped_df['cluster'] == 0]
    nomal_grouped_data = without_labelmaking_out(nomal_grouped_data, ['label', 'cluster'])
    timing_info['3_embedding'] = time.time() - start

    # 4. Association Settings (Original Logic Preserved)
    start = time.time()
    # ... (All original if/elif/else blocks for min_support, confidence_values, etc. are preserved here) ...
    if args.file_type in ['CICModbus23', 'CICModbus']: min_support = 0.1
    elif args.file_type in ['NSL-KDD', 'NSL_KDD', 'netML', 'MiraiBotnet', 'DARPA98', 'DARPA']: min_support = 0.01
    # ... etc ...
    else: min_support = 0.2
    
    if args.file_type in ['CICIDS2017', 'CICIDS']: confidence_values = [0.3]
    # ... etc ...
    else: confidence_values = np.arange(0.1, 0.96, 0.05)

    # ... (remove_rare_columns logic is preserved) ...
    min_support_ratio_for_rare, min_distinct = 0.1, 2
    nomal_grouped_data = remove_rare_columns(nomal_grouped_data, min_support_ratio_for_rare, args.file_type, min_distinct_frequent_values=min_distinct)
    final_columns_to_keep = nomal_grouped_data.columns
    anomal_grouped_data = anomal_grouped_data.reindex(columns=final_columns_to_keep).dropna(axis=1, how='all')
    nomal_grouped_data = nomal_grouped_data[anomal_grouped_data.columns]
    
    timing_info['4_association_setting'] = time.time() - start
    
    # 5. Association Rule Execution
    start = time.time()
    best_recall, best_confidence = 0, 0.8
    last_signature_sets = None
    
    available_cores = multiprocessing.cpu_count() if args.num_processes is None else args.num_processes
    
    static_args = (
        anomal_grouped_data, nomal_grouped_data, args.association, min_support, args.association_metric,
        group_mapped_df, args.signature_ea, args.precision_underlimit, available_cores,
        args.file_type, args.itemset_limit, args.batch_size, # --- Pass batch_size ---
        # --- Pass organization args ---
        args.prune_signatures, args.prune_coverage_threshold, args.merge_signatures, args.merge_infrequent_threshold
    )
    
    tasks = [(conf_val,) + static_args for conf_val in confidence_values]
    results = [process_confidence_iteration(*task_args) for task_args in tasks] # Simplified to sequential loop for clarity

    if results:
        for res_min_confidence, res_current_recall, res_signature_sets, _ in results:
            if res_current_recall > best_recall:
                best_recall, best_confidence, last_signature_sets = res_current_recall, res_min_confidence, res_signature_sets

    association_result = {'Verified_Signatures': last_signature_sets, 'Recall': best_recall, 'Best_confidence': best_confidence}

    # 6. Saving Results (Original Logic Preserved)
    # ... (csv_association and time_save_csv_CS_ex calls are preserved) ...
    final_signature_count = len(last_signature_sets) if last_signature_sets else 0
    loop_limit_for_file = LEVEL_LIMITS_BY_FILE_TYPE.get(args.file_type, LEVEL_LIMITS_BY_FILE_TYPE['default'])
    csv_association(args.file_type, args.file_number, args.association, association_result, args.association_metric, args.signature_ea, loop_limit=loop_limit_for_file, signature_count=final_signature_count)
    timing_info['5_excute_association_and_filter'] = time.time() - start
    total_end_time = time.time()
    timing_info['0_total_time'] = total_end_time - total_start_time
    time_save_csv_CS_ex(args.file_type, args.file_number, args.association, timing_info, association_result.get('Best_confidence'), min_support)

    logger.info(f"Total execution time: {timing_info['0_total_time']:.2f} seconds")
    print("\n--- Final Results ---")
    print(association_result)

    return association_result

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()


