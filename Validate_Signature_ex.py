import pandas as pd
import numpy as np
import json
import argparse
import logging
import os
import sys
import ast
from datetime import datetime
import io

# Set the project root path based on the path of the current script (adjust as needed)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add the project root to the front of sys.path (avoid duplicate additions if it already exists)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Import project-specific modules --- 
# Now all local modules are imported relative to the project root.
from utils.time_transfer import time_scalar_transfer, convert_cic_time_to_numeric_scalars
from utils.save_data_io import save_to_json, load_from_json

from Rebuild_Method.FalsePositive_Check import (
    apply_signatures_to_dataset,
    calculate_fp_scores,
    evaluate_false_positives,
    summarize_fp_results
)

from Validation.generation_fp import generate_fake_fp_signatures
from Validation.Validation_util import map_data_using_category_mapping
    
from Dataset_Choose_Rule.choose_amount_dataset import file_cut_GEN
from Dataset_Choose_Rule.Raw_Dataset_infos import Dataset_infos
from Dataset_Choose_Rule.association_data_choose import file_path_line_association
from Dataset_Choose_Rule.dtype_optimize import _post_process_specific_datasets


# Logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set default path (may need to be adjusted for your environment)
# PROJECT_ROOT is ~/asic/ASIC_0605/ (example)
# We need to go one level up from PROJECT_ROOT to get to ~/asic/
GRANDPARENT_DIR = os.path.dirname(PROJECT_ROOT) # This should give ~/asic/
BASE_SIGNATURE_PATH = os.path.join(GRANDPARENT_DIR, "Dataset_Paral", "signature")
BASE_MAPPING_PATH = os.path.join(GRANDPARENT_DIR, "Dataset_Paral", "signature") # Corrected path
BASE_DATA_PATH = os.path.join(GRANDPARENT_DIR, "Dataset", "load_dataset") # Corrected path for datasets


def calculate_performance_metrics(alerts_df, ground_truth_df):
    """
    Calculates TP, FP, FN, Precision, and Recall based on alerts and ground truth.
    Assumes ground_truth_df has a 'label' column where 1=attack, 0=normal.
    """
    if 'label' not in ground_truth_df.columns:
        logger.error("Ground truth 'label' column not found for performance calculation.")
        return {
            "true_positives": 0, "false_positives": 0, "false_negatives": 0,
            "precision": 0.0, "recall": 0.0, "total_alerts": len(alerts_df)
        }

    # Indices of actual attacks from the ground truth
    actual_positives_indices = set(ground_truth_df[ground_truth_df['label'] == 1].index)
    
    # Handle case where no alerts were generated
    if alerts_df.empty:
        return {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": len(actual_positives_indices),
            "precision": 0.0,
            "recall": 0.0,
            "total_alerts": 0
        }

    # Indices of data points that triggered an alert
    alerted_indices = set(alerts_df['alert_index'])
    
    # TP: Alerted indices that are actual positives
    tp = len(alerted_indices.intersection(actual_positives_indices))
    
    # FP: Alerted indices that are NOT actual positives
    fp = len(alerted_indices) - tp
    
    # FN: Actual positive indices that were NOT alerted
    fn = len(actual_positives_indices) - tp
    
    # Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "total_alerts": len(alerts_df)
    }


def load_signatures(file_type, config_name_prefix):
    sig_file_name_json = f"{file_type}_{config_name_prefix}.json"
    sig_file_path_json = os.path.join(BASE_SIGNATURE_PATH, file_type, sig_file_name_json)
    
    signatures = None
    
    logger.info(f"Attempting to load signatures from JSON: {sig_file_path_json}")
    if os.path.exists(sig_file_path_json):
        signatures = load_from_json(sig_file_path_json)
        if signatures is not None:
            logger.info(f"Successfully loaded signatures from JSON: {sig_file_path_json}")
        else:
            logger.warning(f"Failed to load signatures from existing JSON file: {sig_file_path_json}")
    else:
        logger.info(f"JSON signature file not found: {sig_file_path_json}")

    if signatures is None:
        sig_file_name_csv = f"{file_type}_{config_name_prefix}.csv"
        sig_file_path_csv = os.path.join(BASE_SIGNATURE_PATH, file_type, sig_file_name_csv)
        logger.info(f"Attempting to load signatures from CSV: {sig_file_path_csv}")
        if os.path.exists(sig_file_path_csv):
            try:
                df = pd.read_csv(sig_file_path_csv)
                if not df.empty and 'Verified_Signatures' in df.columns:
                    # Assuming the relevant data is in the first row
                    verified_signatures_str = df['Verified_Signatures'].iloc[0]
                    if pd.isna(verified_signatures_str):
                        logger.error(f"'Verified_Signatures' content is NaN in CSV: {sig_file_path_csv}")
                    else:
                        try:
                            parsed_list = ast.literal_eval(verified_signatures_str)
                            if isinstance(parsed_list, list):
                                extracted_signatures = []
                                for idx, item in enumerate(parsed_list):
                                    if isinstance(item, dict) and \
                                       'signature_name' in item and \
                                       isinstance(item['signature_name'], dict) and \
                                       'Signature_dict' in item['signature_name']:
                                        
                                        rule_dict = item['signature_name']['Signature_dict']
                                        # Use a generated ID for now, or look for a specific field if available
                                        # For example, if item['signature_name'] has an 'id' or 'name' field
                                        sig_id = item['signature_name'].get('name', f"csv_sig_{idx}") 
                                        extracted_signatures.append({'id': sig_id, 'rule_dict': rule_dict})
                                    else:
                                        logger.warning(f"Skipping item from CSV due to unexpected structure: {item}")
                                signatures = extracted_signatures
                                logger.info(f"Successfully extracted {len(signatures)} signatures from CSV: {sig_file_path_csv}")
                            else:
                                logger.error(f"Parsed 'Verified_Signatures' from CSV is not a list: {sig_file_path_csv}")
                        except (ValueError, SyntaxError) as e:
                            logger.error(f"Error parsing 'Verified_Signatures' string from CSV {sig_file_path_csv}: {e}")
                else:
                    logger.error(f"CSV file {sig_file_path_csv} is empty or 'Verified_Signatures' column is missing.")
            except Exception as e:
                logger.error(f"Error reading CSV file {sig_file_path_csv}: {e}")
        else:
            logger.info(f"CSV signature file not found: {sig_file_path_csv}")

    if signatures is None:
        logger.error(f"Failed to load signatures from both JSON and CSV for prefix: {config_name_prefix}")
        return None

    # Common processing for signatures loaded from either JSON or CSV
    if not isinstance(signatures, list):
        if isinstance(signatures, dict) and all(isinstance(s_val, dict) for s_val in signatures.values()):
            logger.info("Attempting to convert signatures from dict of dicts to list of dicts...")
            signatures_list = []
            for s_id, s_content in signatures.items():
                s_content['id'] = s_id # Assign or overwrite 'id'
                # Ensure 'rule_dict' exists, if s_content is supposed to be the rule itself
                if 'rule_dict' not in s_content: 
                     # This case implies the dict itself is the rule_dict
                     # This might happen if the JSON format was { "id1": {"rule_key": "val"}, ...}
                     # And we expect {"id": "id1", "rule_dict": {"rule_key": "val"}}
                     # However, the original code appends s_content directly, assuming it's already structured
                     # correctly or that rule_dict is a primary key within it.
                     # For safety, if 'rule_dict' is not in s_content, we might need to decide if s_content *is* the rule_dict
                     # This part of logic is from the original code for JSON dict-of-dicts.
                     # If CSV always produces list of {'id': X, 'rule_dict': Y}, this dict-of-dicts branch is only for JSON.
                     pass # Assuming s_content structure is what's expected or 'rule_dict' is already a key within.
                signatures_list.append(s_content)
            signatures = signatures_list
            logger.info(f"Successfully converted signatures to list format. Count: {len(signatures)}")
        else:
            logger.error("Signatures are not in the expected list format and could not be converted.")
            return None
            
    # Final validation
    if not signatures: # Check if signatures list is empty after all processing
        logger.error("No signatures were successfully loaded or extracted.")
        return None

    if not all(isinstance(s, dict) and 'id' in s and 'rule_dict' in s for s in signatures):
        logger.error("Signatures are not in the expected final format (list of dicts with 'id' and 'rule_dict').")
        logger.debug(f"Problematic signatures data: {signatures[:5]}") # Log first few items for debugging
        return None
        
    logger.info(f"Final loaded signature count: {len(signatures)}")
    return signatures


def load_category_mapping(file_type, file_number):
    """
    Loads and correctly parses the category mapping CSV file, distinguishing
    between interval and categorical/binary features based on robust heuristics.
    """
    map_file_name = f"{file_type}_{file_number}_mapped_info.csv"
    map_file_path = os.path.join(BASE_MAPPING_PATH, file_type, map_file_name)
    
    logger.info(f"Attempting to load and parse category mapping from CSV: {map_file_path}")

    if not os.path.exists(map_file_path):
        logger.error(f"CSV Mapping file not found: {map_file_path}")
        return None

    try:
        df_mapping = pd.read_csv(map_file_path, dtype=str).fillna('')
        if df_mapping.empty:
            logger.error(f"Mapping CSV file is empty: {map_file_path}")
            return None
        
        category_mapping = {
            'interval': pd.DataFrame(),
            'categorical': {},
            'binary': {}
        }

        def is_interval_col(series):
            # A robust heuristic: a column is interval if its first valid entry
            # looks like a mathematical interval, e.g., "(0.5, 1.5]=1"
            first_valid_entry = series[series != ''].iloc[0]
            return ('(' in first_valid_entry or '[' in first_valid_entry) and \
                   (')' in first_valid_entry or ']' in first_valid_entry) and \
                   ',' in first_valid_entry

        def reconstruct_categorical_map(series):
            # Reconstructs a mapping dict from "value=group" strings
            mapping_dict = {}
            for item in series[series != '']:
                parts = str(item).split('=', 1)
                if len(parts) == 2:
                    value, group_str = parts
                    try:
                        mapping_dict[value.strip()] = int(group_str.strip())
                    except ValueError:
                        logger.warning(f"Could not parse group ID as integer in '{item}'. Skipping.")
            return mapping_dict

        interval_cols = []
        categorical_cols = []
        for col in df_mapping.columns:
            if df_mapping[col][df_mapping[col] != ''].empty:
                continue
            if is_interval_col(df_mapping[col]):
                interval_cols.append(col)
        else:
                categorical_cols.append(col)

        if interval_cols:
            logger.info(f"Identified Interval columns: {interval_cols}")
            category_mapping['interval'] = df_mapping[interval_cols].copy()
        
        if categorical_cols:
            logger.info(f"Identified Categorical/Binary columns: {categorical_cols}")
            for col in categorical_cols:
                mapping_dict = reconstruct_categorical_map(df_mapping[col])
                if not mapping_dict:
                    logger.warning(f"Could not reconstruct map for categorical column '{col}'.")
                    continue
                # Simple heuristic: if only 2 unique values, consider it binary
                if len(mapping_dict) <= 2:
                    category_mapping['binary'][col] = mapping_dict
        else:
                    category_mapping['categorical'][col] = mapping_dict

        logger.info(f"Successfully loaded and processed category mapping from {map_file_path}")
        return category_mapping

    except Exception as e:
        logger.error(f"Error processing mapping CSV file {map_file_path}: {e}", exc_info=True)
        return None


def load_dataset(file_type):
    # Get the base dataset path from the imported function
    # file_path_line_association returns (file_path, file_number), we only need file_path
    base_dataset_path, _ = file_path_line_association(file_type)

    # The path returned by file_path_line_association is relative to its own location.
    # We need to make it absolute or relative to the current PROJECT_ROOT or GRANDPARENT_DIR.
    # Assuming paths in association_data_choose.py like "../Dataset/load_dataset/..." 
    # are intended to be relative from a script in a subfolder of GRANDPARENT_DIR (e.g. PROJECT_ROOT)
    # So, os.path.join(GRANDPARENT_DIR, path_from_assoc_choose.lstrip('../')) might work.
    # Let's check how GRANDPARENT_DIR and BASE_DATA_PATH are defined again.
    # GRANDPARENT_DIR = os.path.dirname(PROJECT_ROOT) # e.g., /home/work/asic
    # BASE_DATA_PATH = os.path.join(GRANDPARENT_DIR, "Dataset", "load_dataset") # e.g., /home/work/asic/Dataset/load_dataset
    
    # file_path_line_association returns paths like "../Dataset/load_dataset/CICModbus23/CICModbus23_total.csv"
    # If Validate_Signature_ex.py is at /home/work/asic/ASIC_0605/Validate_Signature_ex.py
    # Then os.path.abspath(os.path.join(os.path.dirname(__file__), base_dataset_path)) should resolve correctly.
    # Or, more simply, ensure base_dataset_path is correctly relative to GRANDPARENT_DIR if it starts with "../"
    
    if base_dataset_path.startswith("../"):
        # Resolve path relative to the directory of Validate_Signature_ex.py (PROJECT_ROOT)
        # then go up one level as implied by ".." in the path string from association_data_choose
        # and then follow the rest of the path. GRANDPARENT_DIR is effectively PROJECT_ROOT/..
        dataset_path = os.path.abspath(os.path.join(PROJECT_ROOT, base_dataset_path))
    elif base_dataset_path.startswith("~"):
        dataset_path = os.path.expanduser(base_dataset_path)
    else: 
        # If it's already an absolute path or a path type not starting with ../ or ~
        # This case might need more specific handling if other path types occur
        # If it's already an absolute path or a path type not starting with ../ or ~
        # This case might need more specific handling if other path types occur
        dataset_path = base_dataset_path

    # Original filename logic for logging, though the actual file loaded is dataset_path
    # This might be confusing if dataset_name_suffix was important for distinguishing versions/types.
    # For now, we use the direct path. The concept of dataset_name_suffix might need re-evaluation
    # if different versions of the *same base file* are needed.
    dataset_filename_for_log = os.path.basename(dataset_path) 

    logger.info(f"Loading dataset from: {dataset_path} (determined by association_data_choose.py)")
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found: {dataset_path}")
        return None
    try:
        dataset_info = Dataset_infos.get(file_type, {})
        header_row = 0 if dataset_info.get('has_header', True) else None
        df = file_cut_GEN(file_type, dataset_path, 'all', header=header_row)
        # --- DEBUG: Print columns to identify the ground truth column ---
        if file_type in ['DARPA', 'DARPA98']:
            logger.info(f"[DEBUG] Columns for DARPA98 dataset: {df.columns.tolist()}")
        # --- END DEBUG ---
    except Exception as e:
        logger.error(f"Error loading data using file_cut_GEN for {dataset_path}: {e}")
        try:
            # Fallback to simple pd.read_csv if file_cut_GEN fails
            df = pd.read_csv(dataset_path, header=header_row) # Pass header_row here too
            logger.info(f"Successfully loaded {dataset_path} using pd.read_csv fallback.")
            # IMPORTANT: Apply post-processing in the fallback as well
            df = _post_process_specific_datasets(df, file_type)
            logger.info("Applied post-processing to the fallback-loaded data.")
        except Exception as e_pd:
            logger.error(f"Error loading data using pd.read_csv for {dataset_path}: {e_pd}")
            return None
    logger.info(f"Loaded dataset {dataset_filename_for_log}, shape: {df.shape if df is not None else 'Error'}")
    return df

def main(args):
    """Main execution function."""
    logger.info(f"--- Starting Validation Script with args: {args} ---")
    
    # --- Step 1: Load all necessary data ---
    original_signatures = load_signatures(args.file_type, args.config_name_prefix)
    if original_signatures is None:
        logger.error("Stopping due to failure in loading signatures.")
        return
    
    category_mapping = load_category_mapping(args.file_type, args.file_number)
    if category_mapping is None:
        logger.error("Stopping due to failure in loading category mapping.")
        return

    data_df = load_dataset(args.file_type)
    if data_df is None or data_df.empty:
        logger.error("Stopping due to failure in loading or processing dataset.")
        return
    
    # --- Start: Specific and Centralized Labeling Logic ---
    # This section ensures consistent and accurate labeling across different main scripts.
    # It is placed here because the logic in dtype_optimize.py was too general.
    logger.info(f"Applying specific labeling for file_type: {args.file_type}")
    if args.file_type in ['MiraiBotnet', 'NSL-KDD', 'NSL_KDD']:
        logger.warning(f"Labeling for {args.file_type} might need a specific function ('anomal_judgment_nonlabel'). Using placeholder.")
        if 'Label' in data_df.columns:
             data_df['label'] = data_df['Label']
        else:
             data_df['label'] = 0
    elif args.file_type == 'netML':
        data_df['label'] = data_df['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
    elif args.file_type == 'DARPA98':
        if 'Class' in data_df.columns:
            data_df['label'] = data_df['Class'].apply(lambda x: 0 if str(x).strip() == '-' else 1)
        elif 'attack' in data_df.columns: # Fallback for older file format
            data_df['label'] = data_df['attack'].apply(lambda x: 0 if str(x).strip() == 'normal.' else 1)
        else:
            logger.error("Cannot find 'Class' or 'attack' column for DARPA98 labeling.")
            data_df['label'] = 0
    elif args.file_type in ['CICIDS2017', 'CICIDS']:
        if 'Label' in data_df.columns:
            data_df['label'] = data_df['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
        else:
            logger.error(f"ERROR: 'Label' column not found for {args.file_type}.")
            data_df['label'] = 0
    elif args.file_type in ['CICModbus23', 'CICModbus']:
        data_df['label'] = data_df['Attack'].apply(lambda x: 0 if str(x).strip() == 'Baseline Replay: In position' else 1)
    elif args.file_type in ['IoTID20', 'IoTID']:
        data_df['label'] = data_df['Label'].apply(lambda x: 0 if str(x).strip() == 'Normal' else 1)
    elif args.file_type == 'Kitsune':
        if 'Label' in data_df.columns:
            data_df['label'] = data_df['Label']
        else:
            logger.error(f"ERROR: 'Label' column not found for {args.file_type}.")
            data_df['label'] = 0
    else:
        logger.warning(f"WARNING: No specific labeling logic for {args.file_type}. Trying generic 'Label' -> 'BENIGN' mapping.")
        if 'Label' in data_df.columns:
            data_df['label'] = data_df['Label'].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)
        elif 'label' in data_df.columns:
             logger.info("Using existing 'label' column.")
        else:
            logger.error(f"Cannot determine labeling for {args.file_type}. Defaulting all to label 0.")
            data_df['label'] = 0
            
    logger.info("Labeling complete.")
    if 'label' in data_df.columns:
        logger.info(f"Value counts in 'label' column:\n{data_df['label'].value_counts()}")
    # --- End: Specific Labeling Logic ---

    logger.info("--- Mapping raw data to group IDs ---")
    group_mapped_df = map_data_using_category_mapping(data_df, category_mapping, file_type=args.file_type)

    if group_mapped_df.empty:
        logger.error("Mapped data is empty. Cannot proceed with validation.")
        return

    if 'label' not in group_mapped_df.columns:
        logger.error("Stopping: Ground truth 'label' column is missing from the loaded dataset.")
        return
        
    # --- Step 2: Calculate Initial Performance (Before FP Removal) ---
    logger.info("\n--- Evaluating Initial Performance (Original Signatures) ---")
    initial_alerts_df = apply_signatures_to_dataset(group_mapped_df, original_signatures)
    initial_performance = calculate_performance_metrics(initial_alerts_df, group_mapped_df)
    logger.info(f"Initial Performance Metrics: {initial_performance}")

    # --- Step 3: Generate and Add Fake Signatures for FP Analysis ---
    logger.info("\n--- Generating Fake FP Signatures for Analysis ---")
    
    # User request: Temporarily disable auto-generation and use hardcoded signatures to ensure warnings are raised.
    logger.info("Temporarily disabling automatic fake signature generation and using hardcoded ones.")
    
    # fake_sigs_list, _, _, _, _, _ = generate_fake_fp_signatures(
    #     file_type=args.file_type,
    #     file_number=args.file_number,
    #     category_mapping=category_mapping,
    #     data_list=[], # No longer used for mapping here
    #     association_method=args.association,
    #     association_metric=args.association_metric,
    #     num_fake_signatures=args.num_fake_signatures
    # )

    # List of hardcoded fake signatures that are likely to cause high FPs
    # These signatures do not use 'Date_scalar'.
    if args.file_type in ['DARPA', 'DARPA98']:
        fake_sigs_list = [
            {
                'id': 'fake_fp_sig_1',
                'name': 'Fake FP - Common Protocol',
                # Assuming 'Protocol' is a feature and '1' is a common mapped value (e.g., for TCP).
                'rule_dict': {'Protocol': 1}
            },
            {
                'id': 'fake_fp_sig_2',
                'name': 'Fake FP - Low Group TargetIP',
                # Assuming '1' is a more common group ID for TargetIP than the previous '5'.
                'rule_dict': {'TargetIP': 1}
            },
            {
                'id': 'fake_fp_sig_3',
                'name': 'Fake FP - Very Common Combo',
                # Combining two likely-noisy conditions.
                'rule_dict': {'Protocol': 1, 'TargetIP': 1}
            },
            {
                'id': 'fake_fp_sig_4',
                'name': 'Fake FP - Generic Attack Group',
                # Targeting group '0' of the Attack feature, which might be a noisy "normal-like" category.
                'rule_dict': {'Attack': 0}
            }
        ]
    elif args.file_type in ['CICModbus23']:
        fake_sigs_list = [
            {
                'id': 'fake_fp_sig_1',
                'name': 'Fake FP - Unmapped TransactionID',
                # This value (-1) was observed to be very common in logs, making it a great FP candidate.
                'rule_dict': {'TransactionID': -1}
            },
            {
                'id': 'fake_fp_sig_2',
                'name': 'Fake FP - Common TargetIP',
                # Using a common, low-numbered group ID as a guess for a frequent target IP.
                'rule_dict': {'TargetIP': 1}
            },
            {
                'id': 'fake_fp_sig_3',
                'name': 'Fake FP - Very Common Combo',
                 # Combining two likely common values to test combined FP detection.
                'rule_dict': {'TransactionID': -1, 'TargetIP': 1}
            },
            {
                'id': 'fake_fp_sig_4',
                'name': 'Fake FP - Generic Attack Group',
                # This targets a specific group of the 'Attack' feature itself, which could be noisy.
                'rule_dict': {'Attack': 4} # Using the same logic as the DARPA98 fake sig
            }
        ]
    args.num_fake_signatures = len(fake_sigs_list)

    if not fake_sigs_list:
        logger.warning("No fake FP signatures were generated. FP analysis will be limited.")
    
    # Create the combined set of signatures for evaluation
    combined_signatures = original_signatures + fake_sigs_list
    logger.info(f"Total signatures for validation (original + fake): {len(combined_signatures)}")

    # --- Step 4: Evaluate Combined Signatures and Identify FPs ---
    logger.info("\n--- Evaluating Combined Signatures to Identify FPs ---")
    combined_alerts_df = apply_signatures_to_dataset(group_mapped_df, combined_signatures)
    logger.info(f"Generated {len(combined_alerts_df)} total alerts from combined signatures.")

    if combined_alerts_df.empty:
        logger.info("No alerts generated from the combined data. Cannot perform FP analysis.")
        # Output only initial results if no alerts are generated
        final_report = {
            "initial_performance": initial_performance,
            "final_performance": initial_performance, # Same as initial
            "performance_change": {"precision_change": 0.0, "recall_change": 0.0},
            "fp_analysis": {
                "removed_signatures_count": 0,
                "removed_signatures": []
            }
        }
        logger.info("\n--- FINAL REPORT ---")
        logger.info(json.dumps(final_report, indent=4))
        return

    '''
    logger.info("\n--- Evaluating Signatures on Mapped Data ---")
    alerts_df = apply_signatures_to_dataset(group_mapped_df, signatures)

    if alerts_df.empty:
        logger.info("No alerts generated from the data.")
        logger.info("Validation process finished.")
        return

    logger.info(f"Generated {len(alerts_df)} total alerts.")
    
    # Separate alerts from original vs fake signatures
    fake_sig_ids = {s['id'] for s in fake_sigs_list}
    original_sig_alerts = alerts_df[~alerts_df['signature_id'].isin(fake_sig_ids)]
    fake_sig_alerts = alerts_df[alerts_df['signature_id'].isin(fake_sig_ids)]

    logger.info(f"Alerts from original signatures: {len(original_sig_alerts)}")
    logger.info(f"Alerts from fake signatures: {len(fake_sig_alerts)}")

    # Evaluate all alerts together
    signatures_map = {sig['id']: sig for sig in signatures}
    '''
    signatures_map = {sig['id']: sig for sig in combined_signatures}
    evaluated_fp_df = evaluate_false_positives(
        alerts_df=combined_alerts_df,
        current_signatures_map=signatures_map,
        attack_free_df=group_mapped_df, # Using full mapped df as stand-in for attack-free
        t0_nra=args.t0_nra, n0_nra=args.n0_nra,
        lambda_haf=args.lambda_haf, lambda_ufp=args.lambda_ufp,
        combine_method=args.combine_method, belief_threshold=args.belief_threshold,
        file_type=args.file_type
    )

    summary_df = summarize_fp_results(evaluated_fp_df)
    
    # Identify signatures to be removed
    fp_signatures_to_remove_df = summary_df[summary_df['final_likely_fp']]
    fp_signatures_to_remove_ids = set(fp_signatures_to_remove_df['signature_id'])
    
    logger.info(f"\n--- FP ANALYSIS COMPLETE ---")
    logger.info(f"Identified {len(fp_signatures_to_remove_ids)} signatures as likely FPs to be removed.")
    if fp_signatures_to_remove_ids:
        logger.info(f"Removing IDs: {list(fp_signatures_to_remove_ids)}")

    # --- Step 5: Calculate Final Performance (After FP Removal) ---
    logger.info("\n--- Evaluating Final Performance (Cleaned Signatures) ---")
    cleaned_signatures = [sig for sig in combined_signatures if sig['id'] not in fp_signatures_to_remove_ids]
    final_alerts_df = apply_signatures_to_dataset(group_mapped_df, cleaned_signatures)
    final_performance = calculate_performance_metrics(final_alerts_df, group_mapped_df)
    logger.info(f"Final Performance Metrics: {final_performance}")

    # --- Step 6: Generate Final Report ---
    # Get details of removed signatures for the report
    removed_sig_details = []
    for _, row in fp_signatures_to_remove_df.iterrows():
        detail = row.to_dict()
        # Ensure avg_belief is serializable (it should be float, but good practice)
        detail['avg_belief'] = float(row['avg_belief'])
        # Add pre-removal alert count for this signature
        detail['alerts_before_removal'] = len(combined_alerts_df[combined_alerts_df['signature_id'] == row['signature_id']])
        removed_sig_details.append(detail)
    
    final_report = {
        "initial_performance": initial_performance,
        "final_performance": final_performance,
        "performance_change": {
            "precision_change": final_performance['precision'] - initial_performance['precision'],
            "recall_change": final_performance['recall'] - initial_performance['recall']
        },
        "fp_analysis": {
            "removed_signatures_count": len(fp_signatures_to_remove_ids),
            "removed_signatures": removed_sig_details
        }
    }

    logger.info("\n--- FINAL REPORT ---")
    # Convert DataFrame to JSON for logging
    report_json = json.dumps(final_report, indent=4)
    logger.info(report_json)
    
    # --- Step 7: Save the report to a file ---
    logger.info("\n--- Saving Final Report to File ---")
    report_dir = os.path.join(GRANDPARENT_DIR, "Validation_Result", args.file_type)
    report_filename = f"report_{args.config_name_prefix}.json"
    report_filepath = os.path.join(report_dir, report_filename)
    
    try:
        # save_to_json is imported from utils.save_data_io and handles directory creation
        save_to_json(final_report, report_filepath)
        logger.info(f"Successfully saved final report to: {report_filepath}")
    except Exception as e:
        logger.error(f"Failed to save final report to {report_filepath}. Error: {e}")

    logger.info("Validation process finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Signature Validation and FP Evaluation Script")
    parser.add_argument("--file_type", type=str, default='CICModbus23', required=True, help="Type of the dataset (e.g., CICModbus23, DARPA98)")
    parser.add_argument("--file_number", type=str, default='1', required=True, help="Identifier for the dataset instance/config (e.g., '1', 'RARM_1')")
    parser.add_argument("--config_name_prefix", type=str, default='RARM_1_confidence_signature_train_ea15', required=True, help="Prefix for signature and mapping config files (e.g., RARM_1_confidence_signature_train_ea15)")
    parser.add_argument("--attack_free_suffix", type=str, default="Normal_train_ea15", help="Suffix for the attack-free (normal) dataset filename")
    parser.add_argument("--test_data_suffix", type=str, default="Test_All_data", help="Suffix for the test dataset filename")
    parser.add_argument("--t0_nra", type=int, default=60, help="Time window for NRA calculation")
    parser.add_argument("--n0_nra", type=int, default=20, help="Normalization factor for NRA")
    parser.add_argument("--lambda_haf", type=float, default=100.0, help="Lambda for HAF calculation")
    parser.add_argument("--lambda_ufp", type=float, default=10.0, help="Lambda for UFP calculation")
    parser.add_argument("--combine_method", type=str, default='max', choices=['max', 'avg', 'weighted'], help="Method to combine FP scores")
    parser.add_argument("--belief_threshold", type=float, default=0.5, help="Belief threshold for classifying as FP")
    parser.add_argument("--num_fake_signatures", type=int, default=5, help="Number of fake FP signatures to generate")
    parser.add_argument("--association", type=str, default='apriori', help="Association rule method for fake sigs (e.g. apriori, fpgrowth)")
    parser.add_argument("--association_metric", type=str, default='confidence', help="Association rule metric for fake sigs")
    parser.add_argument("--fake_min_support", type=float, default=0.2, help="Minimum support for generating fake FP signatures")
    cli_args = parser.parse_args()
    main(cli_args) 