import pandas as pd
import argparse
import time
import os
import sys
import multiprocessing
import logging
from datetime import datetime
# import gc

LEVEL_LIMITS_BY_FILE_TYPE = {
    'MiraiBotnet': 3,
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
    'CICIoT': 5,
    'CICIoT2023': 5,
    'netML': 5,
    'Kitsune': 5,
    'default': None
}

# === START: Project Root Path Correction ===
# Ensure the project root directory (the parent of the script's directory) is in the Python path.
# This allows imports like 'from Modules.some_module import ...' to work correctly
# regardless of where the script is executed from (e.g., from 'ASIC/' or its parent).
try:
    # Get the absolute path of the directory containing this script (e.g., '.../ASIC')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the path of the parent directory (e.g., '...')
    project_root = os.path.dirname(script_dir)
    # Add the project root to the system's path list for module searching
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    # Fallback for environments where __file__ might not be defined (e.g., interactive notebooks)
    # Assumes the script is being run from the 'ASIC' directory.
    if '..' not in sys.path:
        sys.path.insert(0, '..')
# === END: Project Root Path Correction ===


# Now, the following imports should work without triggering the dummy functions.
from Dataset_Choose_Rule.dtype_optimize import load_csv_safely

# Assuming necessary modules from your project are available in the path
# We will try to import them, but might need adjustments based on the exact project structure.
try:
    from Dataset_Choose_Rule.association_data_choose import file_path_line_association
    from Dataset_Choose_Rule.choose_amount_dataset import file_cut
    from definition.Anomal_Judgment import anomal_judgment_label, anomal_judgment_nonlabel
    from utils.time_transfer import time_scalar_transfer
    from Modules.Heterogeneous_module import choose_heterogeneous_method
    from Heterogeneous_Method.separate_group_mapping import map_intervals_to_groups
    from Modules.Association_module import association_module
    from Modules.Difference_sets import dict_list_difference
    from Rebuild_Method.FalsePositive_Check import apply_signatures_to_dataset
except ImportError as e:
    print(f"Warning: Could not import all project modules: {e}. Some functionalities might be limited.")
    # Provide dummy functions if modules are not found, to allow the script to be created.
    def association_module(df, *args, **kwargs):
        print("WARNING: Using dummy 'association_module'.")
        # Return a dummy signature based on the first row
        if not df.empty:
            rule = {col: val for col, val in df.iloc[0].items() if pd.notna(val)}
            return [rule] if rule else []
        return []
    def apply_signatures_to_dataset(df, sigs):
        print("WARNING: Using dummy 'apply_signatures_to_dataset'.")
        return pd.DataFrame()
    def load_csv_safely(file_type, path):
        print("WARNING: Using dummy 'load_csv_safely'. Attempting pd.read_csv directly.")
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            return None

# NEW: Import for plotting
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib is not installed. Plotting functionality will be disabled.")
    plt = None


# --- Logger Setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def preprocess_and_map_chunk(chunk_df, file_type, category_mapping, data_list):
    """
    Applies the necessary preprocessing and mapping to a data chunk.
    This function encapsulates the steps from data loading to group mapping.
    """
    # 1. Labeling
    if file_type in ['MiraiBotnet', 'NSL-KDD', 'NSL_KDD']:
        chunk_df['label'], _ = anomal_judgment_nonlabel(file_type, chunk_df)
    elif file_type == 'netML':
        chunk_df['label'] = chunk_df['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
    elif file_type == 'DARPA98':
        chunk_df['label'] = chunk_df['Class'].apply(lambda x: 0 if str(x).strip() == '-' else 1)
    elif file_type in ['CICIDS2017', 'CICIDS']:
        if 'Label' in chunk_df.columns:
            chunk_df['label'] = chunk_df['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
        else:
            logger.warning(f"Chunk for {file_type} is missing 'Label' column. Defaulting label to 0.")
            chunk_df['label'] = 0 # Default
    elif file_type in ['CICModbus23', 'CICModbus']:
        chunk_df['label'] = chunk_df['Attack'].apply(lambda x: 0 if str(x).strip() == 'Baseline Replay: In position' else 1)
    elif file_type in ['IoTID20', 'IoTID']:
        chunk_df['label'] = chunk_df['Label'].apply(lambda x: 0 if str(x).strip() == 'Normal' else 1)
    elif file_type in ['CICIoT', 'CICIoT2023']:
        chunk_df['label'] = chunk_df['attack_flag']
    elif file_type == 'Kitsune':
        chunk_df['label'] = chunk_df['Label']
    else:
        logger.warning(f"Using generic anomal_judgment_label for {file_type}.")
        chunk_df['label'] = anomal_judgment_label(chunk_df)

    # 2. Time transfer and other preprocessing might be needed here if not done prior
    # chunk_df = time_scalar_transfer(chunk_df, file_type) # This might be complex for chunking

    # 3. Map intervals to groups
    # Note: `data_list` and `regul` might need careful handling in a chunk-based approach.
    # We pass them as is for now.
    group_mapped_chunk, _ = map_intervals_to_groups(chunk_df, category_mapping, data_list, regul='N')
    
    # Re-attach the label column
    if 'label' in chunk_df.columns:
        group_mapped_chunk['label'] = chunk_df['label']
        
    return group_mapped_chunk


def main(args):
    """
    Main function to run the incremental signature generation and validation process.
    """
    start_time = time.time()
    
    # --- MODIFIED: Initial Setup to load category_mapping from file ---
    logger.info("--- Initial Setup: Loading Category Mapping Information ---")
    
    # Construct the path to the pre-generated mapped_info.csv file
    mapped_info_path = f"../Dataset_Paral/signature/{args.file_type}/{args.file_type}_{args.file_number}_mapped_info.csv"
    
    try:
        logger.info(f"Loading mapping information from: {mapped_info_path}")
        mapped_info_df = load_csv_safely(args.file_type, mapped_info_path)
        if mapped_info_df is None or mapped_info_df.empty:
            raise FileNotFoundError
            
        # Reconstruct the category_mapping dictionary from the dataframe
        # This logic is based on how it's constructed in `Validation_backup.py`
        category_mapping = { 'interval': {}, 'categorical': pd.DataFrame(), 'binary': pd.DataFrame() }
        
        for column in mapped_info_df.columns:
            column_mappings = []
            for value in mapped_info_df[column].dropna():
                if isinstance(value, str) and '=' in value:
                    column_mappings.append(value)
            if column_mappings:
                category_mapping['interval'][column] = pd.Series(column_mappings)
        
        category_mapping['interval'] = pd.DataFrame(category_mapping['interval'])
        logger.info(f"Successfully loaded and reconstructed category_mapping. Interval features found: {list(category_mapping['interval'].columns)}")

    except FileNotFoundError:
        logger.error(f"FATAL: `mapped_info.csv` not found at '{mapped_info_path}'.")
        logger.error("Please run `Main_Association_Rule.py` first to generate the necessary mapping file.")
        return
    except Exception as e:
        logger.error(f"FATAL: An error occurred while loading or parsing `mapped_info.csv`: {e}")
        return

    # This is likely not needed if the mapping is handled correctly, but kept for compatibility.
    data_list = [pd.DataFrame(), pd.DataFrame()]
    # --- END MODIFIED SECTION ---

    # --- Data Streaming and Processing ---
    file_path, _ = file_path_line_association(args.file_type, args.file_number)
    chunk_size = args.chunk_size
    
    # Using python's csv reader for robust chunking
    try:
        data_iterator = pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)
    except FileNotFoundError:
        logger.error(f"Data file not found at: {file_path}")
        return

    all_valid_signatures = {} # Using a dict to store signatures by a unique ID
    signatures_to_remove = set()
    turn_counter = 0
    
    # NEW: For metrics and plotting
    history = []
    processed_data_so_far = pd.DataFrame()


    for chunk in data_iterator:
        turn_counter += 1
        logger.info(f"--- Processing Turn {turn_counter} (Rows {turn_counter*chunk_size - (chunk_size-1)} - {turn_counter*chunk_size}) ---")
        
        # Preprocess and map the current chunk of data
        try:
            mapped_chunk = preprocess_and_map_chunk(chunk, args.file_type, category_mapping, data_list)
            # Accumulate processed data for total recall/precision calculation
            processed_data_so_far = pd.concat([processed_data_so_far, mapped_chunk], ignore_index=True)
        except Exception as e:
            logger.error(f"Failed to preprocess chunk {turn_counter}: {e}")
            continue

        normal_data_in_chunk = mapped_chunk[mapped_chunk['label'] == 0].copy()
        anomalous_data_in_chunk = mapped_chunk[mapped_chunk['label'] == 1].copy()

        # --- 1. Validation Step ---
        newly_removed_count = 0
        if all_valid_signatures and not normal_data_in_chunk.empty:
            logger.info(f"Validating {len(all_valid_signatures) - len(signatures_to_remove)} existing signatures against {len(normal_data_in_chunk)} normal data rows...")
            
            # Format signatures for the apply function
            signatures_to_test = [
                {'id': sig_id, 'name': f'Sig_{sig_id}', 'rule_dict': rule}
                for sig_id, rule in all_valid_signatures.items() if sig_id not in signatures_to_remove
            ]

            if signatures_to_test:
                # This function returns a DataFrame with alerts
                fp_alerts = apply_signatures_to_dataset(normal_data_in_chunk, signatures_to_test)
                
                if not fp_alerts.empty:
                    flagged_for_removal = set(fp_alerts['signature_id'].unique())
                    newly_flagged = flagged_for_removal - signatures_to_remove
                    newly_removed_count = len(newly_flagged)
                    if newly_removed_count > 0:
                        logger.warning(f"Found {newly_removed_count} new signatures causing False Positives in this turn. Flagging for removal.")
                        signatures_to_remove.update(newly_flagged)

        # --- 2. Generation Step ---
        new_signatures_found = 0
        if not anomalous_data_in_chunk.empty:
            logger.info(f"Generating new signatures from {len(anomalous_data_in_chunk)} anomalous data rows...")
            
            max_level = LEVEL_LIMITS_BY_FILE_TYPE.get(args.file_type, LEVEL_LIMITS_BY_FILE_TYPE['default'])

            anomalous_rules = association_module(
                anomalous_data_in_chunk.drop(columns=['label']),
                association_rule_choose=args.association_method,
                min_support=args.min_support,
                min_confidence=args.min_confidence,
                association_metric='confidence',
                num_processes=args.num_processes,
                file_type_for_limit=args.file_type,
                max_level_limit=max_level
            )
            
            # Add new, unique signatures to the main pool
            for rule in anomalous_rules:
                rule_id = hash(frozenset(rule.items()))
                if rule_id not in all_valid_signatures and rule_id not in signatures_to_remove:
                    all_valid_signatures[rule_id] = rule
                    new_signatures_found += 1
            
            logger.info(f"Generated {new_signatures_found} new unique candidate signatures.")
        
        # --- 3. Performance Evaluation for the current turn ---
        current_valid_signatures = {sig_id: rule for sig_id, rule in all_valid_signatures.items() if sig_id not in signatures_to_remove}
        total_recall = 0
        total_precision = 0
        
        if current_valid_signatures and not processed_data_so_far.empty:
            formatted_current_sigs = [{'id': sid, 'name': f'Sig_{sid}', 'rule_dict': r} for sid, r in current_valid_signatures.items()]
            
            alerts = apply_signatures_to_dataset(processed_data_so_far, formatted_current_sigs)
            
            true_positives = 0
            false_positives = 0
            
            if not alerts.empty:
                alerted_indices = alerts['alert_index'].unique()
                actual_positives_indices = processed_data_so_far[processed_data_so_far['label'] == 1].index
                actual_negatives_indices = processed_data_so_far[processed_data_so_far['label'] == 0].index
                
                true_positives = len(set(alerted_indices).intersection(set(actual_positives_indices)))
                false_positives = len(set(alerted_indices).intersection(set(actual_negatives_indices)))

            total_anomalies_so_far = (processed_data_so_far['label'] == 1).sum()
            total_alerts = len(alerts['alert_index'].unique()) if not alerts.empty else 0
            
            total_recall = true_positives / total_anomalies_so_far if total_anomalies_so_far > 0 else 0
            total_precision = true_positives / total_alerts if total_alerts > 0 else 0

        logger.info(f"End of Turn {turn_counter}. Total Signatures: {len(current_valid_signatures)}. Recall: {total_recall:.4f}. Precision: {total_precision:.4f}")

        # Append metrics to history
        history.append({
            'turn': turn_counter,
            'generated': new_signatures_found,
            'removed': newly_removed_count,
            'recall': total_recall,
            'precision': total_precision
        })
        
        # Explicitly run garbage collection to free up memory at the end of a turn
        # logger.info(f"End of Turn {turn_counter}. Running garbage collection...")
        # collected_count = gc.collect()
        # logger.info(f"Garbage collection complete. Freed {collected_count} objects.")


    # --- Finalization ---
    final_signatures = {sig_id: rule for sig_id, rule in all_valid_signatures.items() if sig_id not in signatures_to_remove}
    
    logger.info("--- Process Complete ---")
    logger.info(f"Initial unique signatures generated: {len(all_valid_signatures)}")
    logger.info(f"Signatures removed due to FPs: {len(signatures_to_remove)}")
    logger.info(f"Final count of validated signatures: {len(final_signatures)}")

    # Save the final signatures to a CSV file
    final_signatures_df = pd.DataFrame([{'signature_rule': str(rule)} for rule in final_signatures.values()])
    
    # MODIFIED: Path adjusted for root directory
    output_dir = f"../Dataset_Paral/validation/{args.file_type}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_filename = f"{args.file_type}_{args.file_number}_{args.association_method}_incremental_signatures.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    final_signatures_df.to_csv(output_path, index=False)
    logger.info(f"Final signatures saved to: {output_path}")

    # --- PLOTTING ---
    if plt and history:
        logger.info("Generating performance graph...")
        history_df = pd.DataFrame(history)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f'Incremental Signature Performance for {args.file_type}\n(support={args.min_support}, confidence={args.min_confidence})', fontsize=16)

        # Subplot 1: Signature Counts
        ax1.plot(history_df['turn'], history_df['generated'], 'o-', label='Generated Signatures in Turn', color='green')
        ax1.plot(history_df['turn'], history_df['removed'], 'x-', label='Removed Signatures in Turn', color='red')
        ax1.set_ylabel('Count')
        ax1.set_title('Signature Generation and Removal per Turn')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Subplot 2: Recall and Precision
        ax2.plot(history_df['turn'], history_df['recall'], 'o-', label='Total Recall', color='blue')
        ax2.plot(history_df['turn'], history_df['precision'], 'x-', label='Total Precision', color='purple')
        ax2.set_xlabel('Turn (50-row chunks)')
        ax2.set_ylabel('Metric Value')
        ax2.set_title('Overall Recall and Precision Over Time')
        ax2.set_ylim(0, 1.05)
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Define output path for the graph
        graph_dir = "../isv_graph/"
        if not os.path.exists(graph_dir):
            try:
                os.makedirs(graph_dir)
            except OSError as e:
                logger.error(f"Could not create graph directory {graph_dir}: {e}")
                graph_dir = "." # Fallback to current directory
        
        graph_filename = f"{args.file_type}_{args.file_number}_s{args.min_support}_c{args.min_confidence}_metrics.jpg"
        graph_path = os.path.join(graph_dir, graph_filename)
        
        try:
            plt.savefig(graph_path, format='jpg', dpi=150)
            logger.info(f"Performance graph saved to: {graph_path}")
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")

    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Incrementally generate and validate signatures from a dataset.")
    parser.add_argument('--file_type', type=str, default="MiraiBotnet", help="Type of the dataset file.")
    parser.add_argument('--file_number', type=int, default=1, help="Number of the dataset file.")
    parser.add_argument('--association_method', type=str, default='rarm', help="Association rule algorithm to use.")
    parser.add_argument('--min_support', type=float, default=0.3, help="Minimum support for association rule mining.")
    parser.add_argument('--min_confidence', type=float, default=0.8, help="Minimum confidence for association rule mining.")
    parser.add_argument('--num_processes', type=int, default=4, help="Number of processes to use for parallel tasks. Defaults to 4.")
    parser.add_argument('--chunk_size', type=int, default=500, help="Number of rows to process in each incremental turn.")
    
    cli_args = parser.parse_args()
    main(cli_args) 