import argparse
import csv
import ast
import os
import pandas as pd
import multiprocessing # For parallel processing
import numpy as np # For array_split if used
from utils.class_row import anomal_class_data, nomal_class_data, without_label
from utils.remove_rare_columns import remove_rare_columns
from Dataset_Choose_Rule.association_data_choose import file_path_line_association
from definition.Anomal_Judgment import anomal_judgment_label, anomal_judgment_nonlabel
from Heterogeneous_Method.separate_group_mapping import map_intervals_to_groups


# Helper function to check if a row matches a signature's conditions
def row_matches_signature(row, signature_conditions, attack_type_column_name_in_signature_key, signature_id_for_debug='UnknownSig'):
    # signature_conditions is the dict like {'Protocol': 'TCP', 'SrcPort': '80', ...}
    # print(f"[DEBUG row_matches_signature] SigID: {signature_id_for_debug} - Checking row: {row.name if isinstance(row, pd.Series) else 'N/A'}") # Can be too verbose
    for cond_key, cond_val_sig in signature_conditions.items():
        if cond_key == attack_type_column_name_in_signature_key or cond_key == 'Signature_ID' or cond_key == 'signature_name': 
            continue

        row_val = row.get(cond_key) 
        
        # Prepare for comparison
        str_row_val = str(row_val) if row_val is not None else None
        str_cond_val_sig = str(cond_val_sig) if cond_val_sig is not None else None

        # Detailed print for mismatch, or for all comparisons if highly verbose debugging is needed
        # if str_row_val != str_cond_val_sig:
        #     print(f"  [DEBUG row_matches_signature] SigID: {signature_id_for_debug}, RowIdx: {row.name if isinstance(row, pd.Series) else 'N/A'} - Mismatch on Key '{cond_key}': RowVal='{str_row_val}' (Type: {type(row_val)}), SigVal='{str_cond_val_sig}' (Type: {type(cond_val_sig)})")
        
        if row_val is None: # If the data row doesn't have this feature, it cannot match.
            # print(f"  [DEBUG row_matches_signature] SigID: {signature_id_for_debug} - Key '{cond_key}' not in row or row_val is None. No match.")
            return False
        
        if str_row_val != str_cond_val_sig:
            return False
            
    # If loop completes, all conditions matched
    # print(f"  [DEBUG row_matches_signature] SigID: {signature_id_for_debug} - MATCHED RowIdx: {row.name if isinstance(row, pd.Series) else 'N/A'}")
    return True

# Worker function for parallel processing of data chunks
def _process_evaluation_chunk(data_chunk, categories_to_evaluate, signatures_grouped_by_intended_category, attack_type_column_name_in_data_and_signature):
    chunk_metrics = {
        category: {'TP': 0, 'FP': 0, 'FN': 0}
        for category in categories_to_evaluate
    }
    # print(f"[DEBUG _process_evaluation_chunk] Processing chunk of size {len(data_chunk)} with {len(signatures_grouped_by_intended_category)} sig groups & {len(categories_to_evaluate)} eval categories")

    for row_idx, row in data_chunk.iterrows():
        actual_row_category = row[attack_type_column_name_in_data_and_signature]
        is_actually_an_attack_row_of_any_type = (row['label'] == 1)

        # Debug print for a specific row if needed
        # if row_idx < 5: # Example: Print details for first 5 rows of each chunk
        #     print(f"  [DEBUG _process_evaluation_chunk] Row {row_idx}: Label={row['label']}, TrueCategory='{actual_row_category}'")

        for category_C_under_evaluation in categories_to_evaluate:
            system_predicts_row_as_category_C = False
            if category_C_under_evaluation in signatures_grouped_by_intended_category:
                for sig_idx, sig_cond_for_C in enumerate(signatures_grouped_by_intended_category[category_C_under_evaluation]):
                    # Try to get a unique ID for the signature for better debug messages
                    sig_id_debug = sig_cond_for_C.get('Signature_ID', f'UnnamedSig_CatC_{sig_idx}') 
                    if row_matches_signature(row, sig_cond_for_C, attack_type_column_name_in_data_and_signature, signature_id_for_debug=sig_id_debug):
                        system_predicts_row_as_category_C = True
                        # print(f"    [DEBUG _process_evaluation_chunk] Row {row_idx} MATCHED by SigID: {sig_id_debug} for Category_C: {category_C_under_evaluation}")
                        break
            
            is_row_actually_category_C = (is_actually_an_attack_row_of_any_type and actual_row_category == category_C_under_evaluation)

            if is_row_actually_category_C:
                if system_predicts_row_as_category_C:
                    chunk_metrics[category_C_under_evaluation]['TP'] += 1
                else:
                    chunk_metrics[category_C_under_evaluation]['FN'] += 1
                    # if is_actually_an_attack_row_of_any_type: # Log all FNs for actual attacks
                    #     print(f"    [DEBUG FN] Row {row_idx} (True: {actual_row_category}, Label: {row['label']}) was FN for Category_C: {category_C_under_evaluation}")
            else:
                if system_predicts_row_as_category_C:
                    chunk_metrics[category_C_under_evaluation]['FP'] += 1
                    # print(f"    [DEBUG FP] Row {row_idx} (True: {actual_row_category}, Label: {row['label']}) was FP for Category_C: {category_C_under_evaluation} (System predicted it as C)")
    
    # After processing all rows in the chunk, print a summary if any TPs were found in this chunk
    # for cat, met in chunk_metrics.items():
    #     if met['TP'] > 0 or met['FP'] > 0 or met['FN'] > 0:
    #          print(f"  [DEBUG _process_evaluation_chunk] Chunk Summary for {cat}: TP={met['TP']}, FP={met['FP']}, FN={met['FN']}")
    return chunk_metrics

def evaluate_signatures(signature_data_list, data_df, attack_type_column_name_in_data_and_signature):
    """
    Evaluates signatures against data to calculate TP, FP, FN for each attack category.

    Args:
        signature_data_list (list): List of signatures. Each signature is a dict, 
                                    e.g., {'signature_name': {'Signature_dict': conditions, ...}}
        data_df (pd.DataFrame): Preprocessed data with 'label' and true attack types.
        attack_type_column_name_in_data_and_signature (str): Column name in data_df for true attack type,
                                                              AND key in signature's Signature_dict for its intended attack type.
    Returns:
        dict: {category: {'TP': count, 'FP': count, 'FN': count}}
    """

    # Determine categories to evaluate based on true attacks in data and intended types in signatures
    true_attack_categories_in_data = set(
        data_df[data_df['label'] == 1][attack_type_column_name_in_data_and_signature].unique()
    )
    # Refine: remove non-attack placeholder values if any
    true_attack_categories_in_data = {cat for cat in true_attack_categories_in_data if cat and cat not in ['Unknown', 'Benign', 'Normal', '-']}

    signatures_grouped_by_intended_category = {}
    all_intended_sig_categories = set()

    for sig_data in signature_data_list:
        sig_conditions = sig_data.get('signature_name', {}).get('Signature_dict', {})
        if not sig_conditions: continue
        
        intended_category = sig_conditions.get(attack_type_column_name_in_data_and_signature)
        if intended_category and intended_category not in ['Unknown', 'Benign', 'Normal', '-']: # Focus on actual attack types
            all_intended_sig_categories.add(intended_category)
            if intended_category not in signatures_grouped_by_intended_category:
                signatures_grouped_by_intended_category[intended_category] = []
            signatures_grouped_by_intended_category[intended_category].append(sig_conditions)
            
    categories_to_evaluate = true_attack_categories_in_data.union(all_intended_sig_categories)
    
    final_evaluation_metrics = {
        category: {'TP': 0, 'FP': 0, 'FN': 0}
        for category in categories_to_evaluate
    }

    if data_df.empty:
        return final_evaluation_metrics

    num_processes = multiprocessing.cpu_count()
    # Ensure num_processes is not more than the number of rows if dataframe is small
    num_processes = min(num_processes, len(data_df))
    if num_processes == 0: num_processes = 1 # Ensure at least one process for very small data_df
        
    # Split dataframe into chunks for parallel processing
    # Using np.array_split can handle cases where len(data_df) is not perfectly divisible by num_processes
    df_chunks = np.array_split(data_df, num_processes)
    
    # Filter out empty chunks that might be produced by array_split if num_processes > len(data_df)
    df_chunks = [chunk for chunk in df_chunks if not chunk.empty]

    if not df_chunks: # If all chunks ended up empty (e.g. original df was empty)
        return final_evaluation_metrics

    # Update num_processes if the number of non-empty chunks is less
    num_processes = len(df_chunks)
    if num_processes == 0: return final_evaluation_metrics # Should be caught by earlier checks

    pool_args = [(chunk, categories_to_evaluate, signatures_grouped_by_intended_category, attack_type_column_name_in_data_and_signature) for chunk in df_chunks]

    with multiprocessing.Pool(processes=num_processes) as pool:
        results_from_chunks = pool.starmap(_process_evaluation_chunk, pool_args)

    # Aggregate results from all chunks
    for chunk_result in results_from_chunks:
        for category, metrics in chunk_result.items():
            if category in final_evaluation_metrics:
                final_evaluation_metrics[category]['TP'] += metrics['TP']
                final_evaluation_metrics[category]['FP'] += metrics['FP']
                final_evaluation_metrics[category]['FN'] += metrics['FN']
            # else: This category wasn't in the initial set, which shouldn't happen if categories_to_evaluate is passed correctly.

    # print(f"[DEBUG evaluate_signatures] Results from chunks before aggregation: {results_from_chunks}")
    return final_evaluation_metrics


def write_results_to_csv(evaluation_results, output_file_path):
    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Headers might need adjustment if average P/R are to be different or omitted
        writer.writerow(['Attack Type', 'Total Precision', 'Total Recall'])
        
        all_precisions = []
        all_recalls = []

        for attack_type, metrics in evaluation_results.items():
            tp = metrics['TP']
            fp = metrics['FP']
            fn = metrics['FN']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            all_precisions.append(precision)
            all_recalls.append(recall)
            
            writer.writerow([attack_type, precision, recall])

        # Calculate and write Macro Average Precision and Recall
        if all_precisions: # Avoid division by zero if no categories
            macro_avg_precision = sum(all_precisions) / len(all_precisions)
        else:
            macro_avg_precision = 0
        
        if all_recalls:
            macro_avg_recall = sum(all_recalls) / len(all_recalls)
        else:
            macro_avg_recall = 0
            
        writer.writerow([]) # Add an empty row for separation
        writer.writerow(['Macro Average', macro_avg_precision, macro_avg_recall])
        print(f"Macro Average Precision: {macro_avg_precision:.4f}")
        print(f"Macro Average Recall: {macro_avg_recall:.4f}")

# write_detailed_results_to_csv is not directly compatible with the new evaluate_signatures output.
# It would require evaluate_signatures to return per-row, per-signature match details if that level of logging is needed.
# For now, its call in main should be commented out or adapted.

def main():
    parser = argparse.ArgumentParser(description='Evaluate attack identification using signatures.')
    parser.add_argument('--file_type', type=str, required=True, help='Type of the dataset file')
    parser.add_argument('--file_number', type=int, required=True, help='File number for dataset')
    parser.add_argument('--association', type=str, required=True, help='Association method to use')
    args = parser.parse_args()

    file_type = args.file_type
    file_number = args.file_number
    association_method = args.association

    data_csv_path, _ = file_path_line_association(file_type, file_number)
    # Construct absolute path for signature_csv_path
    base_signature_dir = os.path.expanduser(f"~/asic/Dataset_Paral/signature/{file_type}")
    signature_csv_path = os.path.join(base_signature_dir, f'{file_type}_{association_method}_{file_number}_confidence_signature_train_ea15.csv')
    print(f"Debug: Looking for signature file at: {signature_csv_path}") # Added for debugging

    output_csv_path = os.path.join(base_signature_dir, f'{file_type}_{association_method}_attack_identification_results.csv') # Also make output path absolute
    # detailed_output_csv_path = os.path.join(base_signature_dir, f'{file_type}_detailed_signature_results.csv') # Commented out

    with open(data_csv_path, mode='r', newline='') as data_file:
        data_reader = csv.DictReader(data_file)
        data_list_of_dicts = [row for row in data_reader]

    if not data_list_of_dicts:
        print(f"No data found in {data_csv_path}")
        return
        
    data_df = pd.DataFrame(data_list_of_dicts)

    # Assign labels and true attack type column
    # The column name used for true attack type (e.g., 'Class', 'Attack', 'AttackType')
    # This name must also be used inside signatures if they self-identify their target type.
    attack_type_column_name_in_data_and_signature = '' 

    if file_type in ['MiraiBotnet', 'NSL-KDD', 'NSL_KDD']:
        data_df['label'], _ = anomal_judgment_nonlabel(file_type, data_df) # Assuming this handles attack_type_column too or it needs to be set
        # For these, `attack_type_column_name_in_data_and_signature` needs to be defined if not 'AttackType' by default
        if file_type == 'MiraiBotnet' and 'AttackType' not in data_df.columns: # Example for Mirai
             # If anomal_judgment_nonlabel doesn't create it, or if it's named differently
             # data_df['AttackType'] = data_df[SOME_OTHER_MIRAI_SPECIFIC_COLUMN]
             pass # Placeholder: Ensure MiraiBotnet data has its specific attack type column
        attack_type_column_name_in_data_and_signature = 'AttackType' # Default, adjust if needed

    elif file_type == 'netML':
        data_df['label'] = data_df['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
        data_df['AttackType'] = data_df['Label'] # Assuming 'Label' column contains specific attack types or 'BENIGN'
        attack_type_column_name_in_data_and_signature = 'AttackType' 

    elif file_type == 'DARPA98':
        data_df['label'] = data_df['Class'].apply(lambda x: 0 if str(x).strip() == '-' else 1)
        # 'Class' column itself is the attack type identifier
        attack_type_column_name_in_data_and_signature = 'Class' 

    elif file_type in ['CICIDS2017', 'CICIDS']:
        if 'Label' in data_df.columns:
            data_df['label'] = data_df['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
            data_df['AttackType'] = data_df['Label'] # Label contains specific attacks or 'BENIGN'
        else: # Fallback if 'Label' is missing
            data_df['label'] = 0 
            data_df['AttackType'] = 'Unknown'
        attack_type_column_name_in_data_and_signature = 'AttackType'

    elif file_type in ['CICModbus23', 'CICModbus']:
        data_df['label'] = data_df['Attack'].apply(lambda x: 0 if str(x).strip() == 'Baseline Replay: In position' else 1)
        # 'Attack' column itself is the attack type identifier
        attack_type_column_name_in_data_and_signature = 'Attack' 

    elif file_type in ['IoTID20', 'IoTID']:
        data_df['label'] = data_df['Label'].apply(lambda x: 0 if str(x).strip() == 'Normal' else 1)
        data_df['AttackType'] = data_df['Label'] # Label contains specific attacks or 'Normal'
        attack_type_column_name_in_data_and_signature = 'AttackType'
    else:
        # Fallback for other/unknown file_types
        data_df['label'] = anomal_judgment_label(data_df) # Ensure this returns a Series
        if 'AttackType' not in data_df.columns: # If not set by judgment func
            print(f"Warning: 'AttackType' column not automatically set for file_type '{file_type}'. Defaulting to 'Unknown'.")
            data_df['AttackType'] = 'Unknown'
        attack_type_column_name_in_data_and_signature = 'AttackType'

    # Ensure 'label' is numeric (0 or 1)
    data_df['label'] = pd.to_numeric(data_df['label'], errors='coerce').fillna(0).astype(int)

    # Replace benign-like names in the true attack type column with a consistent 'Benign' identifier
    # This helps in distinguishing from actual attack categories during evaluation.
    benign_markers = ['BENIGN', 'Normal', '-', 'Baseline Replay: In position'] # Add others as necessary
    for marker in benign_markers:
        if attack_type_column_name_in_data_and_signature in data_df.columns:
            data_df[attack_type_column_name_in_data_and_signature] = data_df[attack_type_column_name_in_data_and_signature].replace(marker, 'Benign')

    # Load mapping information
    mapping_file_path = os.path.expanduser(f"~/asic/Dataset_Paral/signature/{file_type}/{file_type}_{file_number}_mapped_info.csv")
    if not os.path.exists(mapping_file_path):
        raise FileNotFoundError(f"Mapping file not found at: {mapping_file_path}")
    mapping_info_df = pd.read_csv(mapping_file_path)

    category_mapping = {'interval': {}, 'categorical': pd.DataFrame(), 'binary': pd.DataFrame()}
    for column in mapping_info_df.columns:
        column_mappings = [val for val in mapping_info_df[column].dropna() if isinstance(val, str) and '=' in val]
        if column_mappings:
            category_mapping['interval'][column] = pd.Series(column_mappings)
    category_mapping['interval'] = pd.DataFrame(category_mapping['interval'])

    data_for_mapping = data_df.copy()
    simple_data_list_for_mapping = [pd.DataFrame(index=data_df.index), pd.DataFrame(index=data_df.index)]
    group_mapped_interval_cols_df, _ = map_intervals_to_groups(data_for_mapping, category_mapping, simple_data_list_for_mapping, regul='N')
    
    final_evaluation_df = data_df.copy()
    for col in group_mapped_interval_cols_df.columns:
        final_evaluation_df[col] = group_mapped_interval_cols_df[col]

    # Convert mapped columns to numeric for reliable comparison if signature values are numeric
    for col in category_mapping['interval'].columns:
        if col in final_evaluation_df.columns:
            final_evaluation_df[col] = pd.to_numeric(final_evaluation_df[col], errors='coerce')

    print(f"Data prepared for evaluation. Sample:\n{final_evaluation_df.head()}")
    print(f"Using '{attack_type_column_name_in_data_and_signature}' as the key for attack types in data and signatures.")

    # Load signatures
    all_signatures_from_file = []
    if not os.path.exists(signature_csv_path):
        print(f"Signature file not found: {signature_csv_path}")
        return
        
    with open(signature_csv_path, mode='r', newline='') as sig_file:
        sig_reader = csv.DictReader(sig_file)
        if 'Verified_Signatures' not in sig_reader.fieldnames:
            raise ValueError(f"'Verified_Signatures' column not found in {signature_csv_path}")
        
        for sig_row_idx, sig_row in enumerate(sig_reader):
            verified_sigs_str = sig_row.get('Verified_Signatures')
            if verified_sigs_str:
                try:
                    current_signature_batch = ast.literal_eval(verified_sigs_str)
                    if isinstance(current_signature_batch, list):
                        all_signatures_from_file.extend(current_signature_batch)
                    else:
                        print(f"Warning: Row {sig_row_idx+1} 'Verified_Signatures' in {signature_csv_path} is not a list (type: {type(current_signature_batch)}). Skipping.")
                except (ValueError, SyntaxError) as e:
                    print(f"Error parsing 'Verified_Signatures' in row {sig_row_idx+1} of {signature_csv_path}: {e}. Content preview: '{verified_sigs_str[:100]}...'")
            # else: Row might be empty, which is fine.

    if not all_signatures_from_file:
        print(f"No valid signatures loaded from {signature_csv_path}. Cannot perform evaluation.")
        return

    print(f"Loaded {len(all_signatures_from_file)} signatures for evaluation.")

    # print(f"[DEBUG main] Sample of final_evaluation_df before evaluate_signatures (first 3 rows):\n{final_evaluation_df.head(3)}")
    # print(f"[DEBUG main] Info of final_evaluation_df:\n{final_evaluation_df.info()}")
    # print(f"[DEBUG main] Attack types in data to be evaluated: {final_evaluation_df[attack_type_column_name_in_data_and_signature].unique()}")
    # print(f"[DEBUG main] Loaded {len(all_signatures_from_file)} signatures. Sample signature 0 conditions: {all_signatures_from_file[0].get('signature_name',{}).get('Signature_dict',{}) if all_signatures_from_file else 'No sigs'}")
    # print(f"[DEBUG main] Signatures grouped by intended category keys: {signatures_grouped_by_intended_category.keys() if 'signatures_grouped_by_intended_category' in locals() else 'Not computed yet'}") #This var is local to evaluate_signatures

    aggregated_metrics = evaluate_signatures(all_signatures_from_file, final_evaluation_df, attack_type_column_name_in_data_and_signature)
    
    write_results_to_csv(aggregated_metrics, output_csv_path)
    print(f"Aggregated evaluation results saved to {output_csv_path}")
    # print(f"Detailed results CSV generation (currently commented out) would be at {detailed_output_csv_path}")

if __name__ == '__main__':
    main()


