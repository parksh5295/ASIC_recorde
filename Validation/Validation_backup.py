import argparse
import pandas as pd
import numpy as np
import time
import os
import json # Keep for loading category_mapping if still done via json here
import re # Keep if any regex is used directly in main
import logging
from datetime import datetime
import multiprocessing # Ensure multiprocessing is imported
from Dataset_Choose_Rule.association_data_choose import file_path_line_signatures, file_path_line_association
from Dataset_Choose_Rule.choose_amount_dataset import file_cut
from definition.Anomal_Judgment import anomal_judgment_label, anomal_judgment_nonlabel
from utils.time_transfer import time_scalar_transfer
from Dataset_Choose_Rule.dtype_optimize import load_csv_safely
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
from Dataset_Choose_Rule.time_save import time_save_csv_VS
from Modules.Signature_evaluation_module import signature_evaluate
from Rebuild_Method.FalsePositive_Check import apply_signatures_to_dataset, calculate_fp_scores, summarize_fp_by_signature, evaluate_false_positives, summarize_fp_results
from Rebuild_Method.Overfiting_Check import evaluate_signature_overfitting, print_signature_overfit_report
from Dataset_Choose_Rule.save_signature_validation import save_validation_results
import ast  # Added for ast.literal_eval
import random # Add random import 
from copy import deepcopy

# Custom Modules from the project
from Dataset_Choose_Rule.association_data_choose import file_path_line_association, file_path_line_signatures
from Dataset_Choose_Rule.choose_amount_dataset import file_cut
from definition.Anomal_Judgment import anomal_judgment_label, anomal_judgment_nonlabel
from utils.time_transfer import time_scalar_transfer # Assuming time_scalar_transfer is still used directly by main for initial data load
from utils.load_and_save_cat_map import load_category_mapping_from_json # If used by main
from Dataset_Choose_Rule.time_save import time_save_csv_VS

# Newly modularized functions
from Validation.generation_fp import generate_fake_fp_signatures
from Validation.Validation_util import (
    _parse_interval_rule_string_for_fake_sigs, # May not be called directly by main
    _apply_numeric_interval_mapping_for_fake_sigs, # May not be called directly by main
    calculate_recall_contribution, # If used by main
    calculate_overall_recall, 
    calculate_overall_precision, # Added this function
    ensure_directory_exists # If used by main for other paths
)
from Dataset_Choose_Rule.save_signature_validation_ex import save_validation_results

# FP Evaluation and other specific modules (ensure paths are correct based on your structure)
from Rebuild_Method.FalsePositive_Check_ex import (
    evaluate_false_positives_and_filter, 
    calculate_fp_scores_for_signatures, 
    apply_whitelist_and_penalty,
    get_fp_evaluation_params # Example: if params are fetched this way
)
from Rebuild_Method.Overfitting_Check_ex import evaluate_signature_overfitting, print_signature_overfit_report
from Modules.Signature_Matching.apply_signatures import apply_signatures_to_dataset
from Modules.Signature_Evaluation.evaluate_rules import signature_evaluate # Or your main evaluation function

# Initialize logger (if not already done or if needed in main)
logger = logging.getLogger(__name__)
# Configure logger if needed (e.g., logging.basicConfig(...))

DEFAULT_CATEGORY = 0 # Define if used in main for some reason, though likely encapsulated

KNOWN_FP_FILE = "known_high_fp_signatures.json" # Known FP signature save file
RECALL_CONTRIBUTION_THRESHOLD = 0.1 # Threshold for whitelisting signatures
NUM_FAKE_FP_SIGNATURES = 3 # Number of fake FP signatures to inject

# Helper function to parse interval rule strings specifically for fake signature generation needs
# This avoids modifying the global separate_group_mapping.py
def _parse_interval_rule_string_for_fake_sigs(rule_str):
    """
    Parses an interval rule string like "(L, U]=G" or "[L, U)=G".
    Returns (lower_bound, upper_bound, lower_inclusive, upper_inclusive, group_index).
    Handles '-inf' as lower bound.
    """
    rule_str = str(rule_str).strip() # Ensure it's a string
    match = re.match(r'([(\[])\s*(-inf|[-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*([)\]])\s*=\s*(\d+)', rule_str)
    if not match:
        # print(f"DEBUG_FAKE_SIG_MAP: Cannot parse interval rule string: {rule_str}")
        raise ValueError(f"Cannot parse interval rule string for fake sigs: {rule_str}")

    lower_bracket, lower_val_str, upper_val_str, upper_bracket, group_num_str = match.groups()
    lower_bound = -np.inf if lower_val_str == '-inf' else float(lower_val_str)
    upper_bound = float(upper_val_str)
    lower_inclusive = (lower_bracket == '[')
    upper_inclusive = (upper_bracket == ']')
    group_index = int(group_num_str)
    return lower_bound, upper_bound, lower_inclusive, upper_inclusive, group_index

# Helper function to apply parsed interval rules to a numeric data series
def _apply_numeric_interval_mapping_for_fake_sigs(numeric_data_series, rule_series):
    """
    Applies interval mapping rules to a numeric pandas Series.
    numeric_data_series: pd.Series of numeric data to be mapped.
    rule_series: pd.Series of interval rule strings (e.g., "(0,10]=0").
    Returns a pd.Series with mapped group indices.
    """
    parsed_rules = []
    for rule_str in rule_series.dropna():
        try:
            parsed_rules.append(_parse_interval_rule_string_for_fake_sigs(rule_str))
        except ValueError:
            # print(f"DEBUG_FAKE_SIG_MAP: Skipping unparsable rule for data mapping: {rule_str}")
            pass # Skip rules that can't be parsed by our helper
    
    if not parsed_rules:
        # print(f"DEBUG_FAKE_SIG_MAP: No valid rules parsed for column {numeric_data_series.name}. Returning NaNs.")
        return pd.Series(np.nan, index=numeric_data_series.index, dtype=np.float64)

    # Sort rules by lower bound, then upper bound (optional, but good practice)
    parsed_rules.sort(key=lambda x: (x[0], x[1]))

    mapped_values = pd.Series(np.nan, index=numeric_data_series.index, dtype=np.float64)
    
    # Ensure numeric_data_series is indeed numeric, coercing errors
    data_to_map = pd.to_numeric(numeric_data_series, errors='coerce')
    valid_data_mask = data_to_map.notna()

    for lower, upper, l_incl, u_incl, group_idx in parsed_rules:
        condition = pd.Series(True, index=data_to_map.index)
        if l_incl:
            condition &= (data_to_map >= lower)
        else:
            condition &= (data_to_map > lower)
        if u_incl:
            condition &= (data_to_map <= upper)
        else:
            condition &= (data_to_map < upper)
        
        final_condition = condition & valid_data_mask
        mapped_values.loc[final_condition] = group_idx
        
    return mapped_values

# Helper function for parallel calculation of single signature contribution
def _calculate_single_signature_contribution(sig_id, alerts_df_subset_cols, anomalous_indices_set, total_anomalous_alerts_count):
    """Calculates recall contribution for a single signature ID."""
    # Recreate alerts_df from the necessary columns passed
    # This is to avoid passing large DataFrames if only a subset is needed and pickling issues.
    # However, alerts_df is filtered by sig_id, so passing the relevant part or whole might be fine.
    # For simplicity here, assuming alerts_df_subset_cols is already filtered for the current sig_id OR we filter it here.
    # The original code did: sig_alerts = alerts_df[alerts_df['signature_id'] == sig_id]
    # This implies that alerts_df should be passed fully, or tasks should pre-filter.
    # For starmap, it's better if the worker function gets exactly what it needs.
    # Option 1: Pass full alerts_df and filter inside (less ideal for many tasks if alerts_df is huge)
    # Option 2: Pre-filter alerts_df for each sig_id before making tasks (more setup but cleaner worker)

    # Assuming alerts_df_subset_cols IS alerts_df (the full one, or a view with 'signature_id' and 'alert_index')
    # This will be re-evaluated based on how tasks are prepared.
    # For now, let's stick to the logic from the original loop:
    sig_alerts = alerts_df_subset_cols[alerts_df_subset_cols['signature_id'] == sig_id]
    
    detected_by_sig = anomalous_indices_set.intersection(set(sig_alerts['alert_index']))
    contribution = 0.0
    if total_anomalous_alerts_count > 0:
        contribution = len(detected_by_sig) / total_anomalous_alerts_count
    return sig_id, contribution

# ===== Helper Function: Calculate Recall Contribution Per Signature =====
def calculate_recall_contribution(group_mapped_df, alerts_df, signature_map):
    """
    Calculates the recall contribution for each signature using parallel processing.

    Args:
        group_mapped_df (pd.DataFrame): DataFrame with original data and 'label' column.
        alerts_df (pd.DataFrame): DataFrame from apply_signatures_to_dataset (covering all signatures).
        signature_map (dict): Dictionary mapping signature_id to signature rule dict.

    Returns:
        dict: Dictionary mapping signature_id to its recall contribution (0.0 to 1.0).
              Returns empty dict if errors occur.
    """
    recall_contributions = {}
    if 'label' not in group_mapped_df.columns:
        print("Error: 'label' column not found in group_mapped_df for recall contribution.")
        return recall_contributions
    if 'alert_index' not in alerts_df.columns or 'signature_id' not in alerts_df.columns:
         print("Error: 'alert_index' or 'signature_id' column not found in alerts_df for recall contribution.")
         return recall_contributions

    anomalous_indices = set(group_mapped_df[group_mapped_df['label'] == 1].index)
    total_anomalous_alerts = len(anomalous_indices)

    if total_anomalous_alerts == 0:
        print("Warning: No anomalous alerts found in group_mapped_df for recall contribution.")
        return {sig_id: 0.0 for sig_id in signature_map.keys()} # All contribute 0

    print(f"\nCalculating recall contribution for {len(signature_map)} signatures using parallel processing...")

    # Prepare tasks for parallel execution
    # Each task will be (sig_id, alerts_df, anomalous_indices, total_anomalous_alerts)
    # Pass alerts_df directly. Pandas DataFrames are picklable.
    tasks = [
        (sig_id, alerts_df[['signature_id', 'alert_index']], anomalous_indices, total_anomalous_alerts)
        for sig_id in signature_map.keys()
    ]

    num_processes = multiprocessing.cpu_count()
    print(f"Using {num_processes} processes for recall contribution calculation.")
    
    results = []
    if tasks: # Proceed only if there are signatures to process
        try:
            with multiprocessing.Pool(processes=num_processes) as pool:
                # Results will be a list of (sig_id, contribution) tuples
                results = pool.starmap(_calculate_single_signature_contribution, tasks)
        except Exception as e:
            print(f"An error occurred during parallel recall contribution calculation: {e}")
            # Fallback to sequential calculation or return empty/partial
            print("Falling back to sequential calculation for recall contribution...")
            for sig_id in signature_map.keys():
                sig_alerts = alerts_df[alerts_df['signature_id'] == sig_id]
                detected_by_sig = anomalous_indices.intersection(set(sig_alerts['alert_index']))
                contribution = 0.0
                if total_anomalous_alerts > 0:
                    contribution = len(detected_by_sig) / total_anomalous_alerts
                recall_contributions[sig_id] = contribution
                # Optional: print contribution per signature
                # print(f"  - {sig_id}: {contribution:.4f} (sequential)")
            return recall_contributions # Return sequentially computed results

    # Populate recall_contributions from parallel results
    for sig_id, contribution in results:
        recall_contributions[sig_id] = contribution
        # Optional: print contribution per signature
        # print(f"  - {sig_id}: {contribution:.4f} (parallel)")

    return recall_contributions
# ====================================================================

def ensure_directory_exists(filepath):
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        print(f"Creating directory: {directory}")
        os.makedirs(directory)

# ===== Recall Calculation Helper Functions =====
def calculate_overall_recall(group_mapped_df, alerts_df, signature_map, relevant_signature_ids=None):
    '''
    Calculates the overall recall for a given set of signatures.

    Args:
        group_mapped_df (pd.DataFrame): DataFrame with original data and 'label' column.
        alerts_df (pd.DataFrame): DataFrame returned by apply_signatures_to_dataset.
                                    Expected columns: 'alert_index', 'signature_id'.
        signature_map (dict): Dictionary mapping signature_id to signature rule dict.
        relevant_signature_ids (set, optional): Set of signature IDs to consider.
                                                If None, all signatures in alerts_df are considered.

    Returns:
        float: Overall recall value (0.0 to 1.0).
    '''
    if 'label' not in group_mapped_df.columns:
        print("Error: 'label' column not found in group_mapped_df for recall calculation.")
        return 0.0
    if 'alert_index' not in alerts_df.columns or 'signature_id' not in alerts_df.columns:
         print("Error: 'alert_index' or 'signature_id' column not found in alerts_df for recall calculation.")
         return 0.0

    total_anomalous_alerts = group_mapped_df['label'].sum()
    if total_anomalous_alerts == 0:
        print("Warning: No anomalous alerts found in group_mapped_df.")
        return 0.0 # Avoid division by zero

    # Get indices of anomalous alerts in the original data
    anomalous_indices = set(group_mapped_df[group_mapped_df['label'] == 1].index)

    # Filter alerts that correspond to anomalous original data
    anomalous_alerts_df = alerts_df[alerts_df['alert_index'].isin(anomalous_indices)].copy()

    # Filter by relevant signature IDs if provided
    if relevant_signature_ids is not None:
        print(f"Calculating recall based on {len(relevant_signature_ids)} signatures.")
        anomalous_alerts_df = anomalous_alerts_df[anomalous_alerts_df['signature_id'].isin(relevant_signature_ids)]
    else:
         print("Calculating recall based on all signatures present in alerts_df.")


    # Count unique anomalous alerts detected by the relevant signatures
    detected_anomalous_alerts = anomalous_alerts_df['alert_index'].nunique()

    recall = detected_anomalous_alerts / total_anomalous_alerts
    print(f"Total Anomalous Alerts: {total_anomalous_alerts}")
    print(f"Detected Anomalous Alerts (by relevant signatures): {detected_anomalous_alerts}")

    return recall

def generate_fake_fp_signatures(file_type, file_number, category_mapping, data_list, association_method, association_metric, num_fake_signatures=3, min_support=0.3, min_confidence=0.8):
    """
    Args:
        file_type (str): Type of the dataset (e.g., 'DARPA98').
        file_number (int): Number of the dataset file.
        category_mapping (dict): Mapping information loaded from mapped_info.csv.
        data_list (list): List used by map_intervals_to_groups.
        association_method (str): Association rule algorithm (e.g., 'apriori').
        association_metric (str): Metric to use for association rule mining (e.g., 'confidence').
        num_fake_signatures (int): Number of fake signatures to generate.
        min_support (float): Minimum support threshold for association mining on ANOMALOUS data.
        min_confidence (float): Original minimum confidence threshold from function signature (this function
                              will internally override and use 0.7 for the association_module call).

    Returns:
        list: A list of dictionaries, where each dictionary represents a fake signature rule.
              Returns empty list if generation fails.
    """
    print(f"\n--- Generating {num_fake_signatures} Fake FP Signatures from ANOMALOUS Data (using min_confidence=0.7) ---")
    fake_signatures = []
    try:
        # 1. Load data
        print("Loading data for fake signature generation...")
        file_path, _ = file_path_line_association(file_type, file_number)
        full_data = file_cut(file_type, file_path, 'all') # Load all data

        # --- Add time scalar transfer step --- 
        print("Applying time scalar transfer...")
        full_data = time_scalar_transfer(full_data, file_type)

        # === START DEBUG: Check full_data after time_scalar_transfer ===
        print("DEBUG: full_data.head() after time_scalar_transfer:")
        print(full_data.head().to_string())
        if 'Date_scalar' in full_data.columns and 'StartTime_scalar' in full_data.columns:
            print("DEBUG: full_data[['Date_scalar', 'StartTime_scalar']].isnull().sum() after time_scalar_transfer:")
            print(full_data[['Date_scalar', 'StartTime_scalar']].isnull().sum().to_string())
        elif 'Date_scalar' in full_data.columns:
            print("DEBUG: full_data['Date_scalar'].isnull().sum() after time_scalar_transfer:")
            print(full_data['Date_scalar'].isnull().sum().to_string())
        elif 'StartTime_scalar' in full_data.columns:
            print("DEBUG: full_data['StartTime_scalar'].isnull().sum() after time_scalar_transfer:")
            print(full_data['StartTime_scalar'].isnull().sum().to_string())
        else:
            print("DEBUG: Neither Date_scalar nor StartTime_scalar found after time_scalar_transfer.")
        # === END DEBUG ===

        # -------------------------------------

        # 2. Assign labels
        print("Assigning labels...")
        if file_type in ['MiraiBotnet', 'NSL-KDD', 'NSL_KDD']:
            full_data['label'], _ = anomal_judgment_nonlabel(file_type, full_data)
        elif file_type == 'netML':
            # print(f"[DEBUG netML MAR] Columns in 'data' DataFrame for netML before processing: {data.columns.tolist()}")
            full_data['label'] = full_data['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
        elif file_type == 'DARPA98':
            full_data['label'] = full_data['Class'].apply(lambda x: 0 if str(x).strip() == '-' else 1)
        elif file_type in ['CICIDS2017', 'CICIDS']:
            print(f"INFO: Processing labels for {file_type}. Mapping BENIGN to 0, others to 1.")
            # Ensure 'Label' column exists
            if 'Label' in full_data.columns:
                full_data['label'] = full_data['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
                logger.info(f"Applied BENIGN/Attack mapping for {file_type}.")
            else:
                logger.error(f"ERROR: 'Label' column not found in data for {file_type}. Cannot apply labeling.")
                # Potentially raise an error or exit if label column is critical and missing
                # For now, it will proceed and might fail later if 'label' is expected
                data['label'] = 0 # Default to 0 or some other placeholder if Label is missing
        elif file_type in ['CICModbus23', 'CICModbus']:
            full_data['label'] = full_data['Attack'].apply(lambda x: 0 if str(x).strip() == 'Baseline Replay: In position' else 1)
        elif file_type in ['IoTID20', 'IoTID']:
            full_data['label'] = full_data['Label'].apply(lambda x: 0 if str(x).strip() == 'Normal' else 1)
        else:
            # This is a fallback, ensure your file_type is covered above for specific handling
            logger.warning(f"WARNING: Using generic anomal_judgment_label for {file_type}.")
            full_data['label'] = anomal_judgment_label(full_data)

        # 3. Filter for ANOMALOUS data.
        #    The variable name `normal_data_df` is INTENTIONALLY PRESERVED from the original code
        #    to minimize diffs, but it will now hold anomalous data.
        normal_data_df = full_data[full_data['label'] == 1].copy() # << CORE LOGIC CHANGE: Filter for label == 1 (anomalous)
        if normal_data_df.empty:
            print("Warning: No ANOMALOUS data found after filtering. Cannot generate fake signatures.")
            return []
        print(f"Filtered for ANOMALOUS data. Rows obtained: {normal_data_df.shape[0]}")

        # 4. Map the ANOMALOUS data (using existing mapping info).
        normal_data_to_map = normal_data_df.drop(columns=['label'], errors='ignore')

        # --- START: Special handling for Date_scalar and StartTime_scalar ---
        mapped_date_scalar = None
        mapped_starttime_scalar = None
        cols_to_process_separately = ['Date_scalar', 'StartTime_scalar']
        remaining_cols_for_map_intervals = list(normal_data_to_map.columns)
        temp_category_mapping_interval = category_mapping['interval'].copy() # Work on a copy

        for col_name in cols_to_process_separately:
            if col_name in normal_data_to_map.columns and col_name in temp_category_mapping_interval.columns:
                print(f"INFO: Separately mapping '{col_name}' for fake signature generation.")
                # Ensure data is numeric before passing to our helper
                data_series = pd.to_numeric(normal_data_to_map[col_name], errors='coerce')
                rule_series = temp_category_mapping_interval[col_name]
                mapped_series = _apply_numeric_interval_mapping_for_fake_sigs(data_series, rule_series)
                
                if col_name == 'Date_scalar':
                    mapped_date_scalar = mapped_series.rename('Date_scalar_mapped') # Rename to avoid clash if needed, though we drop original
                elif col_name == 'StartTime_scalar':
                    mapped_starttime_scalar = mapped_series.rename('StartTime_scalar_mapped')
                
                # === START DEBUG 1: Check individually mapped scalar columns ===
                if col_name == 'Date_scalar' and mapped_date_scalar is not None:
                    print(f"DEBUG_FAKE_SIGS: mapped_date_scalar head:\n{mapped_date_scalar.head().to_string()}")
                    print(f"DEBUG_FAKE_SIGS: mapped_date_scalar NaNs: {mapped_date_scalar.isnull().sum()}")
                if col_name == 'StartTime_scalar' and mapped_starttime_scalar is not None:
                    print(f"DEBUG_FAKE_SIGS: mapped_starttime_scalar head:\n{mapped_starttime_scalar.head().to_string()}")
                    print(f"DEBUG_FAKE_SIGS: mapped_starttime_scalar NaNs: {mapped_starttime_scalar.isnull().sum()}")
                # === END DEBUG 1 ===

                # Remove from data and category_mapping before passing to map_intervals_to_groups
                if col_name in remaining_cols_for_map_intervals: # Should always be true here
                    remaining_cols_for_map_intervals.remove(col_name)
                if col_name in temp_category_mapping_interval.columns: # Should always be true here
                    temp_category_mapping_interval = temp_category_mapping_interval.drop(columns=[col_name])
            else:
                print(f"INFO: Column '{col_name}' not found in data or category_mapping for separate processing.")

        # Prepare data and category_mapping for the original map_intervals_to_groups
        data_for_map_intervals = normal_data_to_map[remaining_cols_for_map_intervals]
        # Create a new category_mapping dict for map_intervals_to_groups to use, with the modified interval part
        category_mapping_for_map_intervals = {
            'interval': temp_category_mapping_interval,
            'categorical': category_mapping.get('categorical', pd.DataFrame()),
            'binary': category_mapping.get('binary', pd.DataFrame())
        }
        # --- END: Special handling ---

        # Call original map_intervals_to_groups for remaining columns
        print(f"Mapping remaining interval columns using original map_intervals_to_groups: {temp_category_mapping_interval.columns.tolist()}")
        if not data_for_map_intervals.empty and not temp_category_mapping_interval.empty:
            other_mapped_df, _ = map_intervals_to_groups(data_for_map_intervals, category_mapping_for_map_intervals, data_list, regul='N')
            # === START DEBUG 2: Check other_mapped_df (from map_intervals_to_groups) ===
            print(f"DEBUG_FAKE_SIGS: other_mapped_df head after map_intervals_to_groups:\n{other_mapped_df.head().to_string()}")
            print(f"DEBUG_FAKE_SIGS: other_mapped_df NaNs after map_intervals_to_groups:\n{other_mapped_df.isnull().sum().to_string()}")
            # === END DEBUG 2 ===
        else:
            print("INFO: No remaining columns or interval rules for map_intervals_to_groups. Creating empty DataFrame for other_mapped_df.")
            other_mapped_df = pd.DataFrame(index=normal_data_to_map.index) # Ensure index compatibility

        # Combine manually mapped scalar time columns with other_mapped_df
        final_mapped_parts = []
        if other_mapped_df.shape[1] > 0:
             final_mapped_parts.append(other_mapped_df)
        if mapped_date_scalar is not None:
            # Use original name for consistency if no clash, or new name if preferred
            final_mapped_parts.append(mapped_date_scalar.rename('Date_scalar')) 
        if mapped_starttime_scalar is not None:
            final_mapped_parts.append(mapped_starttime_scalar.rename('StartTime_scalar'))
        
        if final_mapped_parts:
            # === START DEBUG 3a: Check parts before concat ===
            print("DEBUG_FAKE_SIGS: Checking parts before pd.concat:")
            for i, part_df in enumerate(final_mapped_parts):
                if part_df is not None:
                    print(f"  Part {i} ({part_df.name if hasattr(part_df, 'name') else 'DataFrame'}): shape={part_df.shape}, NaNs={part_df.isnull().sum().sum() if isinstance(part_df, pd.Series) else part_df.isnull().sum().sum()}")
                    print(f"    Head:\n{part_df.head().to_string()}")
                else:
                    print(f"  Part {i} is None")
            # === END DEBUG 3a ===
            normal_mapped_df = pd.concat(final_mapped_parts, axis=1)
            # === START DEBUG 3b: Check normal_mapped_df after concat (this is the state just before dropna) ===
            print(f"DEBUG_FAKE_SIGS: normal_mapped_df head AFTER concat (before dropna):\n{normal_mapped_df.head().to_string()}")
            print(f"DEBUG_FAKE_SIGS: normal_mapped_df NaNs AFTER concat (before dropna):\n{normal_mapped_df.isnull().sum().to_string()}")
            # === END DEBUG 3b ===
        else: # Should not happen if there was any data to map
            print("Warning: All parts for final mapped df are empty.")
            normal_mapped_df = pd.DataFrame(index=normal_data_to_map.index)

        print(f"Shape of combined mapped ANOMALOUS data: {normal_mapped_df.shape}")
        # DEBUG: Check normal_mapped_df before dropna (this was a previous debug point, can be re-enabled if needed)
        # print("DEBUG: normal_mapped_df.head() before dropna:")
        # print(normal_mapped_df.head().to_string())
        # print("DEBUG: normal_mapped_df.isnull().sum() before dropna:")
        # print(normal_mapped_df.isnull().sum().to_string())

        # --- Handle NaN values from the (now anomalous) mapped data --- 
        rows_before_dropna = normal_mapped_df.shape[0]
        normal_mapped_df = normal_mapped_df.dropna()
        rows_after_dropna = normal_mapped_df.shape[0]
        if rows_before_dropna > rows_after_dropna:
            print(f"Dropped {rows_before_dropna - rows_after_dropna} rows containing NaN values from mapped ANOMALOUS data.")
        if normal_mapped_df.empty:
            print("Warning: No data left after dropping NaN rows from mapped ANOMALOUS data. Cannot generate fake signatures.")
            return []
        # -------------------------------------------------

        # 5. Run association rule mining on the (now anomalous) mapped data.
        #    A fixed min_confidence of 0.7 will be used for this specific generation process.

        # === USER REQUESTED CHANGE: Force min_support to 0.2 for fake signature generation ===
        min_support = 0.2
        print(f"INFO: Overriding min_support to {min_support} for fake signature generation process (set before association).")
        # === END USER REQUESTED CHANGE ===

        _internal_fixed_confidence = 0.7 # Temporary internal variable for clarity
        print(f"Running {association_method} on ANOMALOUS data (min_support={min_support}, using fixed min_confidence={_internal_fixed_confidence})...")
        
        rules_df = association_module(
            normal_mapped_df, # This DataFrame, despite its name, now contains ANOMALOUS data
            association_method,
            association_metric=association_metric,
            min_support=min_support,
            min_confidence=_internal_fixed_confidence # << CORE LOGIC CHANGE: Using the fixed 0.7 confidence
        )

        # 6. Extract top rules as fake signatures
        if rules_df is not None and not rules_df.empty and 'rule' in rules_df.columns:
            potential_rules = rules_df['rule'].tolist()
            valid_rules = [rule for rule in potential_rules if isinstance(rule, dict)]

            fake_signatures = valid_rules[:num_fake_signatures]
            print(f"Generated {len(fake_signatures)} fake signature rules from ANOMALOUS data.")
        else:
            print("Warning: Association rule mining on ANOMALOUS data did not produce usable rules.")

    except Exception as e:
        print(f"Error during fake signature generation (intended from ANOMALOUS data): {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback

    print("--- Fake FP Signature Generation (from ANOMALOUS data with 0.7 confidence) Complete ---")
    return fake_signatures

def main():
    # argparser
    # Create an instance that can receive argument values
    parser = argparse.ArgumentParser(description='Argparser')

    # Set the argument values to be input (default value can be set)
    parser.add_argument('--file_type', type=str, default="MiraiBotnet")   # data file type
    parser.add_argument('--file_number', type=int, default=1)   # Detach files
    parser.add_argument('--train_test', type=int, default=0)    # train = 0, test = 1
    parser.add_argument('--heterogeneous', type=str, default="Normalized")   # Heterogeneous(Embedding) Methods
    parser.add_argument('--clustering', type=str, default="kmeans")   # Clustering Methods
    parser.add_argument('--eval_clustering_silhouette', type=str, default="n")
    parser.add_argument('--association', type=str, default="apriori")   # Association Rule
    parser.add_argument('--precision_underlimit', type=float, default=0.6)
    parser.add_argument('--signature_ea', type=int, default=15)
    parser.add_argument('--association_metric', type=str, default='confidence')
    parser.add_argument('--fp_belief_threshold', type=float, default=0.8)
    parser.add_argument('--fp_superset_strictness', type=float, default=0.9)
    parser.add_argument('--fp_t0_nra', type=int, default=60)
    parser.add_argument('--fp_n0_nra', type=int, default=20)
    parser.add_argument('--fp_lambda_haf', type=float, default=100.0)
    parser.add_argument('--fp_lambda_ufp', type=float, default=10.0)
    parser.add_argument('--fp_combine_method', type=str, default='max')
    parser.add_argument('--reset-known-fp', action='store_true',
                        help='Ignore existing known_high_fp_signatures.json and start fresh.')

    # Save the above in args
    args = parser.parse_args()

    # <<< START: Conditional FP Parameter Override for DARPA98 >>>
    if args.file_type in ["DARPA98", "DARPA"]:
        print("INFO: File type is 'DARPA98'. Applying specific stricter FP parameters.")
        # Override parameters only if they were not explicitly provided by the user
        # (We achieve this by checking if the current value is the standard default)
        if args.fp_belief_threshold == 0.5: # Standard default
             args.fp_belief_threshold = 0.95 # DARPA98 specific
        if args.fp_superset_strictness == 0.9:
             args.fp_superset_strictness = 0.6
        if args.fp_t0_nra == 60:
             args.fp_t0_nra = 180
        if args.fp_n0_nra == 20:
             args.fp_n0_nra = 100
        if args.fp_lambda_haf == 100.0:
             args.fp_lambda_haf = 25.0
        if args.fp_lambda_ufp == 10.0:
             args.fp_lambda_ufp = 2.5
    # <<< END: Conditional FP Parameter Override >>>

    # Output the value of the input arguments
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
    # Use potentially overridden values from args
    fp_belief_threshold = args.fp_belief_threshold
    fp_superset_strictness = args.fp_superset_strictness
    fp_t0_nra = args.fp_t0_nra
    fp_n0_nra = args.fp_n0_nra
    fp_lambda_haf = args.fp_lambda_haf
    fp_lambda_ufp = args.fp_lambda_ufp
    fp_combine_method = args.fp_combine_method
    reset_known_fp = args.reset_known_fp

    total_start_time = time.time()  # Start All Time
    timing_info = {}  # For step-by-step time recording


    # 1. Data loading
    start = time.time()

    file_path, file_number = file_path_line_association(file_type, file_number)
    print(f"INFO: Loading data using 'file_path_line_association' for {file_type} (typically TRAIN data).")
    # cut_type = str(input("Enter the data cut type: "))
    cut_type = 'all'
    data = file_cut(file_type, file_path, cut_type)

    timing_info['1_load_data'] = time.time() - start


    # 2. Handling judgments of Anomal or Nomal
    start = time.time()

    if file_type in ['MiraiBotnet', 'NSL-KDD']:
        data['label'], _ = anomal_judgment_nonlabel(file_type, data)
    elif file_type == 'netML':
        data['label'] = data['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    elif file_type == 'DARPA98':
        data['label'] = data['Class'].apply(lambda x: 0 if x == '-' else 1)
    elif file_type in ['CICModbus23', 'CICModbus']:
        data['label'] = data['Attack'].apply(lambda x: 0 if x.strip() == 'Baseline Replay: In position' else 1)
    elif file_type in ['IoTID20', 'IoTID']:
        data['label'] = data['Label'].apply(lambda x: 0 if x.strip() == 'Normal' else 1)
    else:
        data['label'] = anomal_judgment_label(data)

    timing_info['2_anomal_judgment'] = time.time() - start

    data = time_scalar_transfer(data, file_type)


    start = time.time()


    # Corrected paths to load from Dataset_Paral
    base_path = f"../Dataset_Paral/signature/{file_type}/"
    # ensure_directory_exists(base_path) # Not strictly needed for loading, but good if any temp writes happen

    mapped_info_path = f"{base_path}{file_type}_{file_number}_mapped_info.csv"
    association_result_path = f"{base_path}{file_type}_{Association_mathod}_{file_number}_{association_metric}_signature_train_ea{signature_ea}.csv"
    
    # Load data in an optimized way
    mapped_info_df = load_csv_safely(file_type, mapped_info_path)
    print("Loading association result from:", association_result_path)
    association_result = pd.read_csv(association_result_path)

    # Extract mapping information from mapped_info_df
    category_mapping = {
        'interval': {},
        'categorical': pd.DataFrame(),
        'binary': pd.DataFrame()
    }

    # Process interval mapping
    for column in mapped_info_df.columns:
        column_mappings = []
        for value in mapped_info_df[column].dropna():  # Process only non-NaN values
            if isinstance(value, str) and '=' in value:  # If mapping information exists
                column_mappings.append(value)
        
        if column_mappings:  # If mapping exists, add it
            category_mapping['interval'][column] = pd.Series(column_mappings)

    # Convert to DataFrame
    category_mapping['interval'] = pd.DataFrame(category_mapping['interval'])

    # === START DEBUG: Check category_mapping in main() after loading and construction ===
    print("DEBUG: category_mapping in main() - Structure Check")
    if isinstance(category_mapping, dict):
        print("  category_mapping is a dict.")
        for key, value in category_mapping.items():
            print(f"  Key: {key}, Type of Value: {type(value)}")
            if key == 'interval' and isinstance(value, pd.DataFrame):
                print(f"    Interval mapping is a DataFrame with columns: {value.columns.tolist()}")
                if 'Date_scalar' in value.columns:
                    print("    DEBUG: category_mapping['interval']['Date_scalar'] in main():")
                    print(value['Date_scalar'].to_string())
                    # Check data type of the first non-NaN mapping rule string
                    first_date_scalar_rule = value['Date_scalar'].dropna().iloc[0] if not value['Date_scalar'].dropna().empty else None
                    if first_date_scalar_rule:
                         print(f"    DEBUG: Type of first Date_scalar rule string: {type(first_date_scalar_rule)}") # e.g., <class 'str'>
                         print(f"    DEBUG: Example Date_scalar rule string: {first_date_scalar_rule}")
                else:
                    print("    DEBUG: 'Date_scalar' not found in category_mapping['interval'] columns in main().")

                if 'StartTime_scalar' in value.columns:
                    print("    DEBUG: category_mapping['interval']['StartTime_scalar'] in main():")
                    print(value['StartTime_scalar'].to_string())
                    first_starttime_scalar_rule = value['StartTime_scalar'].dropna().iloc[0] if not value['StartTime_scalar'].dropna().empty else None
                    if first_starttime_scalar_rule:
                         print(f"    DEBUG: Type of first StartTime_scalar rule string: {type(first_starttime_scalar_rule)}")
                         print(f"    DEBUG: Example StartTime_scalar rule string: {first_starttime_scalar_rule}")
                else:
                    print("    DEBUG: 'StartTime_scalar' not found in category_mapping['interval'] columns in main().")
    else:
        print("  category_mapping is NOT a dict.")
    print("DEBUG: End of category_mapping check in main()")
    # === END DEBUG ===

    # Create data_list - list of DataFrames with empty columns
    data_list = [pd.DataFrame(), pd.DataFrame()]  # Add empty DataFrames at the beginning and end

    # Save label column separately before mapping
    label_series = data['label'] if 'label' in data.columns else None

    # Perform mapping
    group_mapped_df, _ = map_intervals_to_groups(data, category_mapping, data_list, regul='N')

    # Add the label from the source data to group_mapped_df
    group_mapped_df['label'] = data['label']

    print("\nVerifying label addition:")
    print(f"Original data shape: {data.shape}")
    print(f"Mapped data shape: {group_mapped_df.shape}")
    print(f"Label column exists in mapped data: {'label' in group_mapped_df.columns}")

    # Signature evaluation
    timing_info['3_group_mapping'] = time.time() - start

    
    start = time.time()


    # Extract signatures from association_result
    signatures = []
    verified_sigs = ast.literal_eval(association_result['Verified_Signatures'].iloc[0])
    if isinstance(verified_sigs, list):
        try:
            # Evaluate string to Python object
            sig_list = verified_sigs  # This will be a list
            
            # Extract Signature_dict from each signature
            for sig in sig_list:
                if isinstance(sig, dict) and 'signature_name' in sig:
                    sig_info = sig['signature_name']
                    if isinstance(sig_info, dict) and 'Signature_dict' in sig_info:
                        signatures.append(sig_info['Signature_dict'])
            
            print(f"Found {len(signatures)} valid signatures")
            
        except Exception as e:
            print(f"Error parsing signatures: {e}")
    else:
        print(f"Unexpected type for Verified_Signatures: {type(verified_sigs)}")

    # 1. basic signature evaluation
    if signatures:
        signature_result = signature_evaluate(group_mapped_df, signatures)
    else:
        print("Error: No valid signatures found")
        signature_result = pd.DataFrame() # Initialize empty DataFrame if no signatures

    timing_info['4_basic_signature_evaluation'] = time.time() - start # Corrected timing key

    start = time.time() # Restart timer for FP/Overfitting

    # 2. False Positive check Preparation
    # Create a list of signatures formatted for the vectorized function
    formatted_signatures = [
        {
            'id': f'SIG_{idx}',
            'name': f'Signature_{idx}',
            # Store the actual rule dictionary instead of a lambda function
            'rule_dict': sig
        }
        for idx, sig in enumerate(signatures) # 'signatures' holds the original rule dicts
    ]

    # --- The following calls will eventually use the vectorized function ---
    # --- For now, the original apply_signatures_to_dataset might fail ---
    # --- because it expects 'condition' lambda, not 'rule_dict'.       ---
    # --- We will replace the function definition in the next step.     ---

    alerts_df = apply_signatures_to_dataset(group_mapped_df, formatted_signatures)

    # Prepare data for FP analysis
    normal_data = group_mapped_df[group_mapped_df['label'] == 0].copy()
    attack_free_alerts = apply_signatures_to_dataset(normal_data, formatted_signatures) # Alerts on normal data only

    # Traditional FP calculation (Optional: Keep for reference if needed)
    # fp_scores_traditional = calculate_fp_scores(alerts_df, attack_free_alerts) # Use original alerts_df for context if needed by func
    # fp_summary_traditional = summarize_fp_by_signature(fp_scores_traditional)
    # print("\n=== Traditional FP Summary (For Reference) ===")
    # print(fp_summary_traditional)

    # --- Load Known FP Signatures ---
    known_fp_sig_dicts = []
    if reset_known_fp:
        print("INFO: --reset-known-fp flag is set. Ignoring existing known FP file.")
    elif os.path.exists(KNOWN_FP_FILE):
        try:
            with open(KNOWN_FP_FILE, 'r') as f:
                known_fp_sig_dicts = json.load(f)
            print(f"Loaded {len(known_fp_sig_dicts)} known high-FP signatures from {KNOWN_FP_FILE}")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {KNOWN_FP_FILE}. Starting with empty list.")
        except Exception as e:
            print(f"Warning: Error loading {KNOWN_FP_FILE}: {e}. Starting with empty list.")

    # --- Create current signature map (ID -> Dict) ---
    current_signatures_map = {
        f"SIG_{idx}": sig_dict
        for idx, sig_dict in enumerate(signatures)
    }
    initial_signature_ids = set(current_signatures_map.keys()) # All initial signature IDs set

    # --- Calculate Recall Contribution & Determine Whitelist --- 
    print("=== Determining Whitelist based on Recall Contribution ===")
    # First apply signatures to get alerts_df needed for contribution calculation
    alerts_df = apply_signatures_to_dataset(group_mapped_df, formatted_signatures) 
    recall_contributions = calculate_recall_contribution(group_mapped_df, alerts_df, current_signatures_map)
    whitelist_ids = {
        sig_id for sig_id, contrib in recall_contributions.items() 
        if contrib >= RECALL_CONTRIBUTION_THRESHOLD
    }
    print(f"Recall Contribution Threshold: {RECALL_CONTRIBUTION_THRESHOLD}")
    print(f"Signatures to whitelist (contribution >= threshold): {len(whitelist_ids)}")
    if whitelist_ids:
        print(f"Whitelist IDs: {', '.join(sorted(list(whitelist_ids)))}")
    # ------------------------------------------------------------

    # --- Generate and Inject Fake FP Signatures ---
    print("=== Generating and Injecting Fake FP Signatures ===")
    fake_fp_rules = generate_fake_fp_signatures(
        file_type=file_type,
        file_number=file_number,
        category_mapping=category_mapping, # Pass existing mapping
        data_list=data_list, # Pass existing data_list
        association_method=Association_mathod, # Use same method as main analysis
        association_metric=association_metric, # Pass association_metric from main args
        num_fake_signatures=NUM_FAKE_FP_SIGNATURES,
        min_support=0.4, # Slightly higher support for common normal patterns
        min_confidence=0.9 # High confidence for normal patterns
    )

    injected_fake_count = 0
    original_signatures_for_recall = deepcopy(signatures) # Keep original rules for recall calculation
    original_formatted_signatures_for_recall = deepcopy(formatted_signatures) # Keep original formatted sigs
    original_current_signatures_map_for_recall = deepcopy(current_signatures_map) # Keep original map

    if fake_fp_rules:
        for i, fake_rule in enumerate(fake_fp_rules):
            fake_sig_id = f"FAKE_FP_SIG_{i}"
            # Check for ID collision
            if fake_sig_id in current_signatures_map:
                 print(f"Warning: Fake signature ID {fake_sig_id} already exists. Skipping.")
                 continue
            
            # Inject into signatures list
            signatures.append(fake_rule) 
            
            # Inject into formatted_signatures list
            formatted_signatures.append({
                'id': fake_sig_id,
                'name': f'FakeSignature_{i}',
                'rule_dict': fake_rule
            })
            
            # Inject into current_signatures_map
            current_signatures_map[fake_sig_id] = fake_rule
            injected_fake_count += 1
            
        print(f"Successfully injected {injected_fake_count} fake FP signatures.")
        
        # --- Re-apply signatures to include fake ones in alerts_df for FP analysis ---
        print("Re-applying all signatures (including fake ones) to dataset...")
        # Update alerts_df to include alerts from fake signatures
        alerts_df = apply_signatures_to_dataset(group_mapped_df, formatted_signatures) 
        # Update attack_free_alerts as well
        attack_free_alerts = apply_signatures_to_dataset(normal_data, formatted_signatures)
        # ---------------------------------------------------------------------------

    else:
        print("No fake FP signatures were generated or injected.")
    # ---------------------------------------------------

    # --- Enhanced FP analysis (Now includes fake signatures if injected) ---
    print("=== False Positive analysis (Enhanced + Superset Logic) ===")
    # Use the potentially updated alerts_df and attack_free_alerts
    fp_results_detailed = evaluate_false_positives(
        alerts_df.copy(), 
        current_signatures_map=current_signatures_map, # Use map with fake sigs included
        known_fp_sig_dicts=known_fp_sig_dicts,
        attack_free_df=attack_free_alerts.copy(), 
        belief_threshold=fp_belief_threshold,
        superset_strictness=fp_superset_strictness,
        t0_nra=fp_t0_nra,
        n0_nra=fp_n0_nra,
        lambda_haf=fp_lambda_haf,
        lambda_ufp=fp_lambda_ufp,
        combine_method=fp_combine_method,
        file_type=file_type
    )
    fp_summary_enhanced = summarize_fp_results(fp_results_detailed)

    # --- Add Signature Rule and Experimental Info to Enhanced FP Summary ---
    if 'signature_rule' not in fp_summary_enhanced.columns:
         fp_summary_enhanced['signature_rule'] = None
    if 'is_injected_fake' not in fp_summary_enhanced.columns: # Add column for tracking fake signatures
        fp_summary_enhanced['is_injected_fake'] = False 
    if 'is_removed_final' not in fp_summary_enhanced.columns: # Add column for tracking final removal decision
        fp_summary_enhanced['is_removed_final'] = False

    if not fp_summary_enhanced.empty:
        # Map signature rules (including potential fake ones)
        fp_summary_enhanced['signature_rule'] = fp_summary_enhanced['signature_id'].map(current_signatures_map) 
        
        # --- Mark injected fake signatures --- 
        fp_summary_enhanced['is_injected_fake'] = fp_summary_enhanced['signature_id'].str.startswith('FAKE_FP_SIG_')
        
        # --- Mark which signatures were finally removed (after whitelist) ---
        # Ensure actually_removed_ids is calculated before this point
        if 'actually_removed_ids' in locals(): # Check if the variable exists
             fp_summary_enhanced['is_removed_final'] = fp_summary_enhanced['signature_id'].isin(actually_removed_ids)
        else:
             print("Warning: 'actually_removed_ids' not found when trying to mark final removal status.")
             fp_summary_enhanced['is_removed_final'] = None # Indicate status unknown

    print("Enhanced FP analysis results (summary with experimental flags):")
    if not fp_summary_enhanced.empty:
        # ===== Setting Pandas Output Options =====
        # ... (Pandas display options remain the same) ...
        pd.set_option('display.width', 200) 
        pd.set_option('display.max_colwidth', None) 

        # Select and reorder columns for better readability in printout
        cols_to_print = [
            'signature_id', 'alerts_count', 'likely_fp_rate', 'avg_belief', 
            'final_likely_fp', 'is_whitelisted', # Add is_whitelisted if calculated earlier 
            'is_injected_fake', 'is_removed_final', 'signature_rule'
        ]
        # Add 'is_whitelisted' if available from fp_summary_enhanced, otherwise skip
        if 'is_whitelisted' not in fp_summary_enhanced.columns:
             cols_to_print.remove('is_whitelisted')
             
        # Ensure all selected columns exist before printing
        cols_to_print = [col for col in cols_to_print if col in fp_summary_enhanced.columns]
             
        print(fp_summary_enhanced[cols_to_print].to_string(index=False)) # Print selected columns

        # ===== Restore original options (optional) =====
        # ... (Restoring options remains the same) ... 
    else:
        print("Enhanced FP summary results not found.")

    # --- Identify and report high FP signatures --- 
    # Initialize the variable before the if/else block to ensure it's always defined
    initially_flagged_fp_ids = set()
    
    # Now, try to populate it based on FP summary results
    if not fp_summary_enhanced.empty and 'final_likely_fp' in fp_summary_enhanced.columns:
        # Identify ALL signatures initially flagged as high FP by the logic
        try: # Add try-except for robustness during set creation
            initially_flagged_fp_ids = set(fp_summary_enhanced[fp_summary_enhanced['final_likely_fp']]['signature_id'].tolist())
        except Exception as e:
            print(f"Error extracting initially flagged FP IDs: {e}")
            initially_flagged_fp_ids = set() # Fallback to empty set on error
    else:
        print("Warning: Could not determine newly identified FP signatures. 'final_likely_fp' column missing or summary empty.")
        # initially_flagged_fp_ids is already an empty set from initialization

    print(f"\nInitially flagged as High FP by logic: {len(initially_flagged_fp_ids)}")

    # --- Apply Whitelist --- 
    # Ensure whitelist_ids is defined (should be from recall contribution step)
    if 'whitelist_ids' not in locals():
         print("Error: whitelist_ids is not defined before applying whitelist! Initializing to empty set.")
         whitelist_ids = set()
         
    # Ensure initially_flagged_fp_ids is defined *right before* use
    if 'initially_flagged_fp_ids' not in locals():
        print("Warning: initially_flagged_fp_ids was not defined before Apply Whitelist. Initializing to empty set.")
        initially_flagged_fp_ids = set()
        
    # Now perform the set operation
    ids_to_remove = initially_flagged_fp_ids - whitelist_ids
    removed_due_to_whitelist = initially_flagged_fp_ids.intersection(whitelist_ids)
    actually_removed_ids = initially_flagged_fp_ids - removed_due_to_whitelist # IDs that are flagged AND not whitelisted

    print(f"Applying whitelist ({len(whitelist_ids)} IDs)...")
    if removed_due_to_whitelist:
        print(f"Prevented removal of {len(removed_due_to_whitelist)} whitelisted IDs: {', '.join(sorted(list(removed_due_to_whitelist)))}")
    print(f"Final IDs identified for removal (High FP & not whitelisted): {len(actually_removed_ids)}")
    if actually_removed_ids:
        print(f"IDs to remove: {', '.join(sorted(list(actually_removed_ids)))}")

    # --- Log NRA, HAF, UFP for caught FAKE signatures ---
    print("\n--- FP Metrics for Caught Fake Signatures (Loop 1) ---")
    _caught_fake_signature_metrics_log = [] # Use underscore for temp internal list
    # Ensure fp_results_detailed (output from evaluate_false_positives) is available and valid
    if 'fp_results_detailed' in locals() and isinstance(fp_results_detailed, pd.DataFrame) and not fp_results_detailed.empty and 'signature_id' in fp_results_detailed.columns:
        for _sig_id_to_check in actually_removed_ids: # Use temp var for loop iteration
            if _sig_id_to_check.startswith("FAKE_FP_SIG_"):
                # Filter fp_results_detailed for alerts triggered by this specific fake signature on normal data
                _alerts_for_this_fake_sig = fp_results_detailed[fp_results_detailed['signature_id'] == _sig_id_to_check]
                if not _alerts_for_this_fake_sig.empty:
                    _mean_nra = _alerts_for_this_fake_sig['nra_score'].mean() if 'nra_score' in _alerts_for_this_fake_sig else np.nan
                    _mean_haf = _alerts_for_this_fake_sig['haf_score'].mean() if 'haf_score' in _alerts_for_this_fake_sig else np.nan
                    _mean_ufp = _alerts_for_this_fake_sig['ufp_score'].mean() if 'ufp_score' in _alerts_for_this_fake_sig else np.nan
                    
                    _metric_detail = {
                        "fake_signature_id": _sig_id_to_check,
                        "loop_caught": 1, # Hardcoded to 1 for this run
                        "mean_nra_on_normal_data": _mean_nra,
                        "mean_haf_on_normal_data": _mean_haf,
                        "mean_ufp_on_normal_data": _mean_ufp,
                        "alerts_on_normal_data_count": len(_alerts_for_this_fake_sig)
                    }
                    _caught_fake_signature_metrics_log.append(_metric_detail)
                    print(f"  Caught Fake Sig: {_sig_id_to_check}, Loop: 1, Mean NRA: {_mean_nra:.4f}, Mean HAF: {_mean_haf:.4f}, Mean UFP: {_mean_ufp:.4f}, Alerts on Normal: {len(_alerts_for_this_fake_sig)}")
                else:
                    # This might happen if a fake signature is flagged due to other reasons (e.g., superset) 
                    # without having specific alert entries in fp_results_detailed from normal data.
                    print(f"  Caught Fake Sig: {_sig_id_to_check}, Loop: 1, but no detailed alert data found for it in fp_results_detailed (may be caught by other logic like superset, or no alerts on normal data).")
    else:
        print("Warning: `fp_results_detailed` DataFrame not available/valid. Cannot analyze caught fake signatures metrics.")
    
    if not _caught_fake_signature_metrics_log: # Check the temp list
        print("No FAKE signatures were caught and had detailed FP metrics to report in this run.")
    else:
        # Save the caught fake signature metrics to a CSV file
        _caught_fake_fp_metrics_df = pd.DataFrame(_caught_fake_signature_metrics_log)
        _output_dir = f"../Dataset_Paral/validation/{file_type}/" # Define output directory
        # Using Association_mathod as it is in the existing codebase, preserving original variable names
        _csv_filename = f"{file_type}_{file_number}_{Association_mathod}_caught_fake_fp_metrics.csv"
        _csv_full_path = os.path.join(_output_dir, _csv_filename)
        
        ensure_directory_exists(_output_dir) # Ensure the directory itself exists
        
        try:
            _caught_fake_fp_metrics_df.to_csv(_csv_full_path, index=False)
            print(f"Successfully saved caught fake FP metrics to: {_csv_full_path}")
        except Exception as e:
            print(f"Error saving caught fake FP metrics to CSV {_csv_full_path}: {e}")
    # ------------------------------------------------------

    # --- Update and save known FP list ---
    # ... (logic to update/save known FP list remains the same, using initially_flagged_fp_ids) ...

    # --- Overfitting check ---
    print("=== Overfitting score calculation ===")
    high_fp_signatures_count = len(initially_flagged_fp_ids) 
    total_signatures_count = len(signatures) # includes fake ones now
    overfit_results = evaluate_signature_overfitting(
        total_signatures_count=total_signatures_count,
        high_fp_signatures_count=high_fp_signatures_count
    )
    print_signature_overfit_report(overfit_results)
    # ... (explicit score printing remains the same) ...

    # --- Timing ---
    timing_info['5_fp_overfitting_check'] = time.time() - start

    # --- Filter signatures (based on whitelist-applied removal list) ---
    # Use 'actually_removed_ids' to get the final set of signatures
    filtered_signatures_dicts = [
        sig_dict for sig_id, sig_dict in current_signatures_map.items() 
        if sig_id not in actually_removed_ids
    ]
    final_signature_ids = set(current_signatures_map.keys()) - actually_removed_ids

    # Update print statements for clarity
    print(f"Original signature count (before injection): {len(original_signatures_for_recall)}") # Use count before injection
    print(f"Injected fake signature count: {injected_fake_count}")
    print(f"Total signatures before filtering: {len(current_signatures_map)}")
    print(f"Final count of signatures removed: {len(actually_removed_ids)}")
    print(f"Filtered signature count (remaining): {len(filtered_signatures_dicts)}")
    # -----------------------------------

    # ===== Overall Recall Calculation and Output =====
    print("=== Overall Recall Calculation ===")
    # Use the original signatures and alerts from before injection/filtering for 'before' recall
    if 'label' in group_mapped_df.columns: # Check if label exists
         print("--- Recall BEFORE FP Removal (Original Signatures) ---")
         # We need alerts generated ONLY by original signatures
         original_alerts_df = apply_signatures_to_dataset(group_mapped_df, original_formatted_signatures_for_recall)
         recall_before_fp = calculate_overall_recall(
             group_mapped_df, 
             original_alerts_df, 
             original_current_signatures_map_for_recall, 
             set(original_current_signatures_map_for_recall.keys())
         )
         if recall_before_fp is not None:
             print(f"Recall (Original Signatures): {recall_before_fp:.4f}")
         else:
             print("Could not calculate recall for original signatures.")

         print("--- Recall AFTER FP Removal (Final Filtered Signatures) ---")
         if final_signature_ids:
             # Use the alerts_df that includes *all* signatures (original + fake)
             # But filter based on the final_signature_ids (whitelisted OK, removed FPs excluded)
             recall_after_fp = calculate_overall_recall(
                 group_mapped_df, 
                 alerts_df, # Use alerts_df containing triggers from all sigs (incl. whitelisted)
                 current_signatures_map, # Map containing all sigs
                 final_signature_ids # Only consider alerts from the final set
             )
             if recall_after_fp is not None:
                 print(f"Recall (After FP Removal & Whitelisting): {recall_after_fp:.4f}")
             else:
                  print("Could not calculate recall for final filtered signatures.")
         else:
             print("No signatures left after filtering, Recall (After FP Removal): 0.0000")
             recall_after_fp = 0.0
    else:
         print("Warning: Cannot calculate overall recall because 'label' is missing in group_mapped_df.")
         recall_before_fp = None
         recall_after_fp = None
    # =======================================

    # --- Evaluate performance with filtered signatures ---
    print("=== Evaluating Filtered Signatures ===")
    if filtered_signatures_dicts:
        # Use the final filtered list of rule dictionaries
        filtered_signature_result_list = signature_evaluate(group_mapped_df, filtered_signatures_dicts)
        filtered_signature_result = pd.DataFrame(filtered_signature_result_list) 
        print("Filtered Signature Evaluation Results (first 5 rows):")
        print(filtered_signature_result.head().to_string() if not filtered_signature_result.empty else "No results")
    else:
        print("No signatures remaining after filtering.")
        filtered_signature_result = pd.DataFrame()

    # --- Save all results to CSV ---
    print("\n--- Saving Validation Results ---")
    ensure_directory_exists(f"../Dataset_Paral/validation/{file_type}/") # Corrected path
    save_validation_results(
        file_type=file_type,
        file_number=file_number,
        association_rule=Association_mathod,
        basic_eval=signature_result, # Original evaluation results
        fp_results=fp_summary_enhanced, # FP summary (includes fake ones if generated)
        overfit_results=overfit_results,
        filtered_eval=filtered_signature_result, # Eval based on final filtered signatures
        recall_before=recall_before_fp, # Recall based on original signatures
        recall_after=recall_after_fp # Recall based on final filtered signatures
    )

    # --- Save Timing Information ---
    # Also modify timing filename to distinguish experiment runs
    timing_info['total_execution_time'] = time.time() - total_start_time
    ensure_directory_exists(f"../Dataset_Paral/time_log/validation_signature/{file_type}/") # Corrected path
    time_save_csv_VS(file_type, file_number, Association_mathod, timing_info)


if __name__ == "__main__":
    main()