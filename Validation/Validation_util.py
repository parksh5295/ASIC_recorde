import pandas as pd
import numpy as np
import re
import multiprocessing # Added to prevent potential NameError if other functions use it
import os
import logging


logger = logging.getLogger(__name__)

# Helper function to parse interval rule strings specifically for fake signature generation needs
def _parse_interval_rule_string_for_fake_sigs(rule_str):
    """
    Parses an interval rule string like "(L, U]=G" or "[L, U)=G".
    Returns (lower_bound, upper_bound, lower_inclusive, upper_inclusive, group_index).
    Handles '-inf' as lower bound for numeric types.
    Assumes rule_str is in the format like "(val1,val2]=group_id"
    """
    rule_str = str(rule_str).strip()
    # Original simpler regex for numeric intervals: "([(\[])\s*(-inf|[-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*([)\]])\s*=\s*(\d+)"
    # Regex to match formats like "(0.999, 9213.0]=0"
    match = re.match(r'([(\[])\s*(-inf|[-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s*([)\]])\s*=\s*(\d+)', rule_str)
    if not match:
        # Fallback for rules that might not have '=' for group, if separate_group_mapping produces them differently.
        # However, category_mapping from debug log shows "(...]=group" format.
        raise ValueError(f"Cannot parse interval rule string for fake sigs: {rule_str}. Expected format like \"(val1,val2]=group_id\"")

    lower_bracket, lower_val_str, upper_val_str, upper_bracket, group_num_str = match.groups()
    
    lower_bound = -np.inf if lower_val_str == '-inf' else float(lower_val_str)
    upper_bound = float(upper_val_str)
    lower_inclusive = (lower_bracket == '[')
    upper_inclusive = (upper_bracket == ']')
    group_index = int(group_num_str)
        
    return lower_bound, upper_bound, lower_inclusive, upper_inclusive, group_index

# Helper function to apply parsed interval rules to a data series
def _apply_numeric_interval_mapping_for_fake_sigs(data_series, rule_series, feature_name=None):
    """
    Applies interval mapping rules to a pandas Series.
    data_series: pd.Series of data to be mapped (can be numeric or datetime for Date_scalar).
    rule_series: pd.Series of interval rule strings (e.g., "(0,10]=0").
    feature_name: Name of the feature, used to determine special handling (e.g., 'Date_scalar').
    Returns a pd.Series with mapped group indices.
    """
    parsed_rules = []
    for rule_str in rule_series.dropna():
        try:
            parsed_rules.append(_parse_interval_rule_string_for_fake_sigs(rule_str))
        except ValueError as e:
            # print(f"DEBUG_FAKE_SIG_MAP: Skipping unparsable rule for {feature_name or data_series.name}: {rule_str}. Error: {e}")
            pass 
    
    if not parsed_rules:
        return pd.Series(np.nan, index=data_series.index, dtype=np.float64)

    try:
        # Sort by lower bound, then upper bound to handle overlapping rules if any (though usually not expected for well-defined bins)
        parsed_rules.sort(key=lambda x: (x[0], x[1]))
    except TypeError:
        print(f"Warning: Could not sort parsed_rules for {feature_name or data_series.name}. Proceeding without sorting.")

    mapped_values = pd.Series(np.nan, index=data_series.index, dtype=np.float64)
    
    data_to_map = data_series
    # Ensure data_to_map is numeric. Date_scalar from convert_cic_time_to_numeric_scalars should already be float.
    if not pd.api.types.is_numeric_dtype(data_to_map):
        if feature_name == 'Date_scalar' and pd.api.types.is_datetime64_any_dtype(data_series):
            # This case should ideally not be hit if convert_cic_time_to_numeric_scalars was called for Date_scalar
            data_to_map = data_series.apply(lambda x: x.timestamp() if pd.notna(x) else np.nan).astype(float)
        else:
            data_to_map = pd.to_numeric(data_series, errors='coerce')
    
    valid_data_mask = data_to_map.notna()
    if not valid_data_mask.any(): # if all data is NaN after to_numeric, return all NaNs
        return mapped_values

    # Apply rules
    for lower, upper, l_incl, u_incl, group_idx in parsed_rules:
        condition = pd.Series(True, index=data_to_map.index)
        # Apply conditions only on valid (non-NaN) data to avoid comparison errors
        if l_incl:
            condition[valid_data_mask] &= (data_to_map[valid_data_mask] >= lower)
        else:
            condition[valid_data_mask] &= (data_to_map[valid_data_mask] > lower)
        if u_incl:
            condition[valid_data_mask] &= (data_to_map[valid_data_mask] <= upper)
        else:
            condition[valid_data_mask] &= (data_to_map[valid_data_mask] < upper)
        
        # Only update mapped_values for rows that match the current rule AND were originally valid
        # This ensures that original NaNs in data_to_map remain NaN in mapped_values unless explicitly mapped
        final_condition_for_assignment = condition & valid_data_mask
        mapped_values.loc[final_condition_for_assignment] = group_idx
    
    # --- Overflow handling for Date_scalar --- 
    if feature_name == 'Date_scalar' and parsed_rules:
        # Find the rule with the maximum upper bound
        max_upper_bound = -np.inf
        group_for_max_upper_bound = np.nan
        # Find the rule with the minimum lower bound
        min_lower_bound = np.inf
        group_for_min_lower_bound = np.nan

        for r_lower, r_upper, _, _, r_group in parsed_rules:
            if r_upper > max_upper_bound:
                max_upper_bound = r_upper
                group_for_max_upper_bound = r_group
            if r_lower < min_lower_bound:
                min_lower_bound = r_lower
                group_for_min_lower_bound = r_group

        if pd.notna(group_for_max_upper_bound):
            # Identify values that are still NaN after rule application, were originally valid,
            # and are greater than the max upper bound of all rules.
            overflow_condition = (
                mapped_values.isna() & 
                valid_data_mask & 
                (data_to_map > max_upper_bound)
            )
            mapped_values.loc[overflow_condition] = group_for_max_upper_bound
        
        # Optional: Handle underflow (values less than the minimum lower bound)
        if pd.notna(group_for_min_lower_bound):
            underflow_condition = (
                mapped_values.isna() & 
                valid_data_mask & 
                (data_to_map < min_lower_bound)
            )
            # Decide how to treat underflow, e.g., map to the first group or a special group
            # mapped_values.loc[underflow_condition] = group_for_min_lower_bound # Or 0, or specific underflow group_idx
            # For now, only explicit overflow handling as per the primary issue observed.
        
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

def map_data_using_category_mapping(data_df: pd.DataFrame, category_mapping: dict, file_type: str = None) -> pd.DataFrame:
    """
    Maps raw data columns to group indices based on provided category mapping rules.
    Retains original columns that are not part of the mapping rules.
    Handles interval, categorical, and binary features.
    """
    if not isinstance(category_mapping, dict):
        logger.error("Invalid category_mapping format. Expected a dict.")
        return pd.DataFrame()

    # Initialize the new DataFrame with the original data.
    # This ensures all columns not in the mapping are preserved.
    mapped_df = data_df.copy()
    logger.info(f"Initialized mapped_df with shape {mapped_df.shape} to preserve all original columns.")

    if file_type in ['CICIDS2017', 'CICIDS2018', 'CICModbus2022', 'CICModbus23']:
        logger.info(f"Applying time scalar conversion for {file_type}...")
        # This conversion is crucial for time-based features in signatures.
        # Ensure the 'Time' column exists and is in the correct format.
        if 'Time' in mapped_df.columns:
            mapped_df = convert_cic_time_to_numeric_scalars(mapped_df, 'Time')
            logger.info("Time scalar conversion complete.")
            # The new columns are 'Date_scalar' and 'Time_of_day_scalar'
        else:
            logger.warning("Time column not found, skipping scalar conversion. This may affect time-based signatures.")

    # --- 1. Interval Mapping ---
    interval_mapping_df = category_mapping.get('interval')
    if isinstance(interval_mapping_df, pd.DataFrame) and not interval_mapping_df.empty:
        features_to_map = [col for col in interval_mapping_df.columns if col in mapped_df.columns]
        logger.info(f"Applying interval mapping for {len(features_to_map)} features: {features_to_map}")
        for feature in features_to_map:
            if mapped_df[feature].notna().any():
                rule_series = interval_mapping_df[feature].dropna()
                if not rule_series.empty:
                    mapped_series = _apply_numeric_interval_mapping_for_fake_sigs(mapped_df[feature], rule_series, feature_name=feature)
                    mapped_df[feature] = mapped_series
            else:
                logger.warning(f"Feature '{feature}' for interval mapping not found in DataFrame or has all NaN values.")
    else:
        logger.info("No 'interval' mapping rules found or rules are empty.")

    # --- 2. Categorical Mapping ---
    categorical_rules = category_mapping.get('categorical')
    if isinstance(categorical_rules, dict) and categorical_rules:
        features_to_map = [col for col in categorical_rules.keys() if col in mapped_df.columns]
        logger.info(f"Applying categorical mapping for {len(features_to_map)} features: {features_to_map}")
        for feature in features_to_map:
            mapping_dict = categorical_rules[feature]
            if isinstance(mapping_dict, dict) and mapped_df[feature].notna().any():
                # --- FINAL FIX: Robust mapping by converting both to string ---
                logger.info(f"    Applying robust string-based mapping for '{feature}'.")
                type_unified_mapping_dict = {str(k): v for k, v in mapping_dict.items()}
                mapped_series = mapped_df[feature].astype(str).map(type_unified_mapping_dict)
                
                # Add enhanced logging to diagnose mapping issues
                nan_count = mapped_series.isnull().sum()
                if nan_count > 0:
                    total_count = len(mapped_df[feature].dropna())
                    logger.warning(f"    For '{feature}', {nan_count} out of {total_count} values could not be mapped and resulted in NaN.")
                    unmapped_values = mapped_df[feature][mapped_series.isnull()].unique()
                    logger.warning(f"    Sample of unmapped values for '{feature}': {unmapped_values[:5]}")
                
                mapped_df[feature] = mapped_series
                # --- END FINAL FIX ---
    else:
        logger.info("No 'categorical' mapping rules found or rules are empty.")
        
    # --- 3. Binary Mapping ---
    binary_rules = category_mapping.get('binary')
    if isinstance(binary_rules, dict) and binary_rules:
        features_to_map = [col for col in binary_rules.keys() if col in mapped_df.columns]
        logger.info(f"Applying binary mapping for {len(features_to_map)} features: {features_to_map}")
        for feature in features_to_map:
            if mapped_df[feature].notna().any():
                 # For binary, often a direct boolean-to-int cast is sufficient if no explicit map is given
                 if mapped_df[feature].dtype == bool:
                     mapped_df[feature] = mapped_df[feature].astype(int)
                 else:
                     logger.warning(f"Binary mapping for non-boolean column '{feature}' is ambiguous and was skipped.")
    else:
        logger.info("No 'binary' mapping rules found or rules are empty.")

    logger.info(f"Final shape of mapped DataFrame: {mapped_df.shape}")

    # --- Debugging: Check the state of key columns after mapping ---
    for col_to_check in ['TargetIP', 'TransactionID']:
        if col_to_check in mapped_df.columns:
            logger.info(f"Debug: State of '{col_to_check}' column after all mapping. Non-null count: {mapped_df[col_to_check].notna().sum()}")
            if mapped_df[col_to_check].notna().any():
                logger.info(f"Debug: First 5 non-null '{col_to_check}' values: {mapped_df[col_to_check].dropna().head().tolist()}")
            else:
                logger.info(f"Debug: '{col_to_check}' column contains all null values after mapping.")

    return mapped_df

def _parse_categorical_rule_string_for_fake_sigs(rule_str, data_series_dtype):
    """
    Parses a categorical rule string like 'key=value' into a (key, mapped_value) tuple.
    Converts key to match data_series_dtype if numeric.
    """
    if not isinstance(rule_str, str) or '=' not in rule_str:
        raise ValueError(f"Rule string '{rule_str}' is not in 'key=value' format.")
    
    key_str, mapped_val_str = rule_str.split('=', 1)
    
    try:
        mapped_value = int(mapped_val_str)
    except ValueError:
        raise ValueError(f"Cannot parse mapped value '{mapped_val_str}' as int in rule '{rule_str}'.")

    key = key_str # Default key is string
    # Attempt to convert key_str to numeric type if data_series_dtype is numeric
    if pd.api.types.is_float_dtype(data_series_dtype): # Includes float32, float64
        try:
            key = float(key_str)
        except ValueError:
            # If key_str cannot be float (e.g. "unknown"), and dtype is float, this rule might be problematic
            # However, we allow it and let the mapping resolve it (likely won't match if data has actual floats)
            logger.debug(f"Key '{key_str}' could not be parsed as float for a float-type series in rule '{rule_str}'. Using string key.")
            pass # Keep key as key_str if conversion fails for a float series (e.g. 'unknown' in a float series)
    elif pd.api.types.is_integer_dtype(data_series_dtype): # Includes int32, int64 etc.
        try:
            key = int(key_str)
        except ValueError:
            logger.debug(f"Key '{key_str}' could not be parsed as int for an int-type series in rule '{rule_str}'. Using string key.")
            pass # Keep key as key_str if conversion fails

    # If data_series_dtype is object (string), key remains key_str, which is intended.
    return key, mapped_value

def _apply_categorical_mapping_for_fake_sigs(data_series, rule_series, feature_name=None):
    """
    Applies categorical mapping rules to a data series.
    Rules are expected to be 'key=value' strings.
    """
    if feature_name is None:
        feature_name = data_series.name if data_series.name else "Unnamed_Series"

    mapping_dict = {}
    valid_rules_count = 0
    for rule_str in rule_series.dropna().unique():
        try:
            key, mapped_value = _parse_categorical_rule_string_for_fake_sigs(rule_str, data_series.dtype)
            mapping_dict[key] = mapped_value
            valid_rules_count += 1
        except ValueError as e:
            logger.warning(f"Skipping unparsable categorical rule for {feature_name}: '{rule_str}'. Error: {e}")
            pass
    
    if not mapping_dict:
        logger.warning(f"No valid categorical rules parsed for {feature_name}. All values will be NaN.")
        # Return a series of NaNs with the same index and Int64 dtype
        return pd.Series([pd.NA] * len(data_series), index=data_series.index, dtype='Int64')

    # Apply the mapping
    # Pandas .map() handles missing keys by outputting NaN by default.
    mapped_series = data_series.map(mapping_dict)

    # Log if all values became NaN after mapping, which might indicate a type mismatch or other issue.
    if mapped_series.isnull().all() and not data_series.isnull().all() and valid_rules_count > 0 :
        # Take a small sample of unique data values and map keys for logging
        sample_data_uniques = data_series.dropna().unique()
        sample_data_uniques_str = [str(x) for x in sample_data_uniques[:3]]
        if len(sample_data_uniques) > 3: sample_data_uniques_str.append("...")
        
        sample_map_keys = list(mapping_dict.keys())
        sample_map_keys_str = [str(x) for x in sample_map_keys[:3]]
        if len(sample_map_keys) > 3: sample_map_keys_str.append("...")

        logger.warning(
            f"All values became NaN after categorical mapping for {feature_name}. "
            f"This might indicate a mismatch between data values and rule keys. "
            f"Input unique values (sample, type: {data_series.dtype}): [{', '.join(sample_data_uniques_str)}]. "
            f"Map keys (sample, types might vary): [{', '.join(sample_map_keys_str)}]."
        )
                        
    return mapped_series.astype('Int64') # Ensure Int64 dtype for consistency and NA support