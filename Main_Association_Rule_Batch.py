# A program that automatically generates signatures using association rules.

import argparse
import numpy as np
import time
import multiprocessing # Added for parallel processing
import functools # Added for functools.partial if needed, though starmap is used here
import os
from Dataset_Choose_Rule.association_data_choose import file_path_line_association
from Dataset_Choose_Rule.choose_amount_dataset import file_cut
from definition.Anomal_Judgment import anomal_judgment_label, anomal_judgment_nonlabel
from utils.time_transfer import time_scalar_transfer
from utils.class_row import anomal_class_data, without_labelmaking_out, nomal_class_data, without_label
from Modules.Heterogeneous_module import choose_heterogeneous_method
from Heterogeneous_Method.separate_group_mapping import map_intervals_to_groups
from utils.remove_rare_columns import remove_rare_columns
from Modules.Association_module import association_module
from Modules.Signature_evaluation_module import signature_evaluate
from Modules.Signature_underlimit import under_limit
from Evaluation.calculate_signature import calculate_signatures
from Modules.Difference_sets import dict_list_difference
from Dataset_Choose_Rule.save_csv import csv_association
from Dataset_Choose_Rule.time_save import time_save_csv_CS
import logging
from datetime import datetime
from tqdm import tqdm # --- NEW: Import tqdm for progress bars ---


# Define the logger object at the module level so it can be used throughout the script.
# The actual configuration of the logger (level, format, handlers) will be done
# in the if __name__ == '__main__': block or in the main() function.
logger = logging.getLogger(__name__)

LEVEL_LIMITS_BY_FILE_TYPE = {
    'MiraiBotnet': 5,
    'NSL-KDD': 5,
    'NSL_KDD': 5,
    'DARPA98': None,
    'DARPA': None,
    'CICIDS2017': 4,
    'CICIDS': 4,
    'CICModbus23': None,
    'CICModbus': None,
    'IoTID20': None,
    'IoTID': None,
    'netML': 5,
    'Kitsune': 5,
    'default': None
}

# Helper function for parallel processing
# MODIFIED: Added batch_size as an argument
def process_confidence_iteration(min_confidence, anomal_grouped_data, nomal_grouped_data, Association_mathod, min_support, association_metric, group_mapped_df, signature_ea, precision_underlimit, cores_per_algo_task, current_file_type, batch_size):
    """Processes a single iteration of the confidence loop with batching."""
    iteration_start_time = time.time()
    print(f"  [PCI Start] conf={min_confidence}, algo={Association_mathod}, algo_procs={cores_per_algo_task}, batch_size={batch_size}")

    max_level = LEVEL_LIMITS_BY_FILE_TYPE.get(current_file_type, LEVEL_LIMITS_BY_FILE_TYPE['default'])
    
    # --- 1. Generate Anomaly Rules (as before) ---
    association_list_anomal, _ = association_module(anomal_grouped_data, Association_mathod, min_support, min_confidence, association_metric, num_processes=cores_per_algo_task, file_type_for_limit=current_file_type, max_level_limit=max_level)
    
    # --- 2. Normal Data Batch Processing (Scenario B) ---
    print(f"  [PCI Normal Batch] Starting batch processing for normal data. Total size: {len(nomal_grouped_data)}")
    signatures = association_list_anomal  # Start with all anomaly rules
    
    num_batches = (len(nomal_grouped_data) + batch_size - 1) // batch_size
    for i in tqdm(range(num_batches), desc="  [PCI Normal Batch] Progress"):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        normal_chunk = nomal_grouped_data.iloc[start_idx:end_idx]
        
        if normal_chunk.empty:
            continue
            
        association_list_normal_chunk, _ = association_module(normal_chunk, Association_mathod, min_support, min_confidence, association_metric, num_processes=cores_per_algo_task, file_type_for_limit=current_file_type, max_level_limit=max_level)
        
        # Sequentially subtract the rules found in the normal chunk
        signatures = dict_list_difference(signatures, association_list_normal_chunk)
        print(f"    Batch {i+1}/{num_batches}: Found {len(association_list_normal_chunk)} rules in normal chunk. Remaining candidate signatures: {len(signatures)}")

    print(f"  [PCI Normal Batch] Finished. Found {len(signatures)} raw signatures after filtering.")

    # --- 3. Signature Evaluation Batch Processing ---
    print(f"  [PCI Signature Batch] Starting batch processing for signature evaluation. Total signatures: {len(signatures)}")
    signature_result = []
    num_sig_batches = (len(signatures) + batch_size - 1) // batch_size
    for i in tqdm(range(num_sig_batches), desc="  [PCI Signature Batch] Progress"):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        signature_batch = signatures[start_idx:end_idx]

        if not signature_batch:
            continue
            
        batch_result = signature_evaluate(group_mapped_df, signature_batch)
        if batch_result:
            signature_result.extend(batch_result)
    
    print(f"  [PCI Signature Batch] Finished. Evaluated {len(signature_result)} signatures.")

    # --- 4. Final Steps (as before, no batching needed here) ---
    signature_sets = under_limit(signature_result, signature_ea, precision_underlimit)

    if not signature_sets:
        current_recall = 0
    else:
        current_recall = calculate_signatures(group_mapped_df, signature_sets)

    print(f"  [PCI Finish] conf: {min_confidence}, "
          f"Anomal Rules: {len(association_list_anomal)}, "
          f"Signatures (after filter): {len(signatures)}, "
          f"Signature Sets (after eval): {len(signature_sets) if signature_sets else 0}, "
          f"Recall: {current_recall:.4f}")
    
    total_time_per_iteration = time.time() - iteration_start_time
    return min_confidence, current_recall, signature_sets, total_time_per_iteration

def main():
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
    parser.add_argument('--num_processes', type=int, default=None, help="Number of processes to use for the main pool (outer loop) and internal algorithms.")
    parser.add_argument('--batch_size', type=int, default=50000, help="Batch size for processing normal data and signatures.") # --- NEW ---

    args = parser.parse_args()

    file_type = args.file_type
    file_number = args.file_number
    train_tset = args.train_test
    heterogeneous_method = args.heterogeneous
    clustering_algorithm = args.clustering
    eval_clustering_silhouette = args.eval_clustering_silhouette
    Association_mathod = args.association
    precision_underlimit = args.precision_underlimit
    signature_ea = args.signature_ea
    association_metric = args.association_metric
    num_processes = args.num_processes

    total_start_time = time.time()
    timing_info = {}

    logger.info(f"Global start time: {datetime.fromtimestamp(total_start_time).strftime('%Y-%m-%d %H:%M:%S')}")

    # ... (Data loading and preprocessing remains the same) ...
    start = time.time()
    file_path, file_number = file_path_line_association(file_type, file_number)
    cut_type = 'all'
    data = file_cut(file_type, file_path, cut_type)
    timing_info['1_load_data'] = time.time() - start
    logger.info(f"Loading data from file: {file_path}")
    
    start = time.time()
    if file_type in ['MiraiBotnet', 'NSL-KDD', 'NSL_KDD']:
        data['label'], _ = anomal_judgment_nonlabel(file_type, data)
    elif file_type == 'netML':
        data['label'] = data['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
    elif file_type == 'DARPA98':
        data['label'] = data['Class'].apply(lambda x: 0 if str(x).strip() == '-' else 1)
    elif file_type in ['CICIDS2017', 'CICIDS']:
        if 'Label' in data.columns:
            data['label'] = data['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
        else:
            data['label'] = 0
    elif file_type in ['CICModbus23', 'CICModbus']:
        data['label'] = data['Attack'].apply(lambda x: 0 if str(x).strip() == 'Baseline Replay: In position' else 1)
    elif file_type in ['IoTID20', 'IoTID']:
        data['label'] = data['Label'].apply(lambda x: 0 if str(x).strip() == 'Normal' else 1)
    elif file_type in ['CICIoT', 'CICIoT2023']:
        data['label'] = data['attack_flag']
    elif file_type == 'Kitsune':
        data['label'] = data['Label']
    else:
        data['label'] = anomal_judgment_label(data)
    timing_info['2_anomal_judgment'] = time.time() - start
    if 'label' in data.columns:
        nomal_count = len(data[data['label'] == 0])
        anomal_count = len(data) - nomal_count
        logger.info(f"Label generation complete. Total: {len(data)}, Nomal transactions: {nomal_count}, Anomal transactions: {anomal_count}")

    start = time.time()
    data = time_scalar_transfer(data, file_type)
    regul = 'N'
    embedded_dataframe, feature_list, category_mapping, data_list = choose_heterogeneous_method(data, file_type, heterogeneous_method, regul)
    group_mapped_df, mapped_info_df = map_intervals_to_groups(embedded_dataframe, category_mapping, data_list, regul)
    mapped_info_save_path_dir = f"../Dataset_Paral/signature/{file_type}/"
    if not os.path.exists(mapped_info_save_path_dir):
        os.makedirs(mapped_info_save_path_dir)
    mapped_info_save_path = f"{mapped_info_save_path_dir}{file_type}_{file_number}_mapped_info.csv"
    mapped_info_df.to_csv(mapped_info_save_path, index=False)
    if len(group_mapped_df) == len(data) and file_type == 'netML':
        group_mapped_df['label'] = data['label'].values
    elif file_type != 'netML':
        group_mapped_df['label'] = data['label']
    anomal_grouped_data = anomal_class_data(group_mapped_df)
    anomal_grouped_data = without_label(anomal_grouped_data)
    nomal_grouped_data = nomal_class_data(group_mapped_df)
    nomal_grouped_data = without_label(nomal_grouped_data)
    timing_info['3_embedding'] = time.time() - start

    start = time.time()
    if file_type in ['CICModbus23', 'CICModbus']:
        min_support = 0.1
    elif file_type in ['NSL-KDD', 'NSL_KDD', 'netML', 'MiraiBotnet', 'DARPA98', 'DARPA']:
        min_support = 0.01
    elif file_type in ['Kitsune']:
        min_support = 0.05
    elif file_type in ['CICIDS2017', 'CICIDS']:
        min_support = 0.04
    elif file_type in ['CICIoT', 'CICIoT2023']:
        min_support = 0.03
    else:
        min_support = 0.2
    min_support_ratio_for_rare = 0.1
    min_distinct = 2
    best_confidence = 0.8
    if file_type in ['CICIDS2017', 'CICIDS', 'Kitsune']:
        confidence_values = [0.1]
    elif file_type in ['CICModbus23', 'CICModbus']:
        confidence_values = np.arange(0.1, 0.96, 0.05)
    elif file_type in ['MiraiBotnet', 'NSL-KDD', 'NSL_KDD']:
        confidence_values = np.arange(0.1, 0.14, 0.05)
    else:
        confidence_values = np.arange(0.1, 0.96, 0.05)
    best_recall = 0
    anomal_grouped_data = remove_rare_columns(anomal_grouped_data, min_support_ratio_for_rare, file_type, min_distinct_frequent_values=min_distinct)
    nomal_grouped_data = remove_rare_columns(nomal_grouped_data, min_support_ratio_for_rare, file_type, min_distinct_frequent_values=min_distinct)
    timing_info['4_association_setting'] = time.time() - start

    start = time.time()
    last_signature_sets = None
    if num_processes is not None:
        available_cores = num_processes
    else:
        available_cores = multiprocessing.cpu_count()
    main_pool_procs = 1
    cores_per_algo_task = available_cores
    
    static_args = (
        anomal_grouped_data,
        nomal_grouped_data,
        Association_mathod,
        min_support,
        association_metric,
        group_mapped_df,
        signature_ea,
        precision_underlimit,
        cores_per_algo_task,
        file_type,
        args.batch_size # --- NEW: Pass batch_size ---
    )

    tasks = [(conf_val,) + static_args for conf_val in confidence_values]
    results = [process_confidence_iteration(*task_args) for task_args in tasks]

    if results:
        for res_min_confidence, res_current_recall, res_signature_sets, res_total_time in results:
            if res_current_recall > best_recall:
                best_recall = res_current_recall
                best_confidence = res_min_confidence
                last_signature_sets = res_signature_sets
                total_time_per_iteration = res_total_time

    association_result = {
        'Verified_Signatures': last_signature_sets,
        'Recall': best_recall,
        'Best_confidence': best_confidence
    }
    final_signature_count = len(last_signature_sets) if last_signature_sets else 0
    loop_limit_for_file = LEVEL_LIMITS_BY_FILE_TYPE.get(file_type, LEVEL_LIMITS_BY_FILE_TYPE['default'])
    csv_association(
        file_type, file_number, Association_mathod, association_result, 
        association_metric, signature_ea, loop_limit=loop_limit_for_file, signature_count=final_signature_count
    )
    timing_info['5_excute_association'] = time.time() - start

    total_end_time = time.time()
    timing_info['0_total_time'] = total_end_time - total_start_time
    time_save_csv_CS(file_type, file_number, Association_mathod, timing_info, best_confidence, min_support)

    logger.info(f"Total execution time for all algorithms: {total_end_time - total_start_time:.2f} seconds")
    logger.info(f"Global end time: {datetime.fromtimestamp(total_end_time).strftime('%Y-%m-%d %H:%M:%S')}")

    return association_result


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    main()
