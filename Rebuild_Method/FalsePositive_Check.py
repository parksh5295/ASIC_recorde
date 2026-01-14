# Machines for detecting False positives

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import multiprocessing # Added for parallel processing
import logging
from tqdm import tqdm


logger = logging.getLogger(__name__)

# --- NEW: Global data for worker processes in apply_signatures_to_dataset ---
_worker_data_apply_sigs = None

def _init_worker_apply_sigs(df_for_worker, df_cols_list):
    """
    Initializer for worker processes. Sets a global dataframe for the process.
    """
    global _worker_data_apply_sigs
    _worker_data_apply_sigs = (df_for_worker, df_cols_list)

# --- REFACTORED: This is now the ONLY worker function for applying signatures ---
def _apply_single_signature_worker(sig_info):
    """
    The actual task for each worker. It applies a single signature
    against the globally available dataframe subset.
    """
    global _worker_data_apply_sigs
    if _worker_data_apply_sigs is None:
        logger.error("Worker data not initialized for _apply_single_signature_worker. Returning no matches.")
        # Return an empty series on error, which will be handled downstream
        return sig_info.get('id', 'UNKNOWN_ID'), pd.Series(dtype=bool)

    df_subset, df_columns = _worker_data_apply_sigs
    sig_id = sig_info.get('id', 'UNKNOWN_ID')

    # --- This is the original logic from the old _apply_single_signature_task ---
    is_debug_signature = sig_id == "fake_fp_sig_1"
    if is_debug_signature:
        logger.info(f"[DEBUG] Processing signature: {sig_id}")
        logger.info(f"[DEBUG] Rule to apply: {sig_info.get('rule_dict')}")
        logger.info(f"[DEBUG] Data sample (first 5 rows) dtypes:\n{df_subset.head().info()}")
        logger.info(f"[DEBUG] Data sample (first 5 rows) values:\n{df_subset.head()}")
    # --- END DEBUG LOGGING ---
    
    if 'rule_dict' not in sig_info or not isinstance(sig_info['rule_dict'], dict):
        # print(f"Warning: Skipping signature {sig_id} due to missing or invalid 'rule_dict'.")
        return sig_id, pd.Series(False, index=df_subset.index) # Return an all-False mask
    
    sig_condition_dict = sig_info['rule_dict']
    if not sig_condition_dict:
        # print(f"Info: Skipping signature {sig_id} because its rule_dict is empty.")
        return sig_id, pd.Series(False, index=df_subset.index)

    mask = pd.Series(True, index=df_subset.index)
    valid_signature_for_mask = True
    try:
        for col, value in sig_condition_dict.items():
            if col in df_columns: # Check against original df_columns passed to worker
                col_series = df_subset[col]

                # --- START DEBUG LOGGING (for fake_fp_sig_1 only) ---
                if is_debug_signature:
                    logger.info(f"  [DEBUG] Condition: Key='{col}', Sig_Value='{value}' (Type: {type(value)})")
                    logger.info(f"  [DEBUG] Data Column '{col}' Dtype: {col_series.dtype}")
                # --- END DEBUG LOGGING ---

                # Perform comparison
                current_mask = pd.Series(False, index=df_subset.index)
                try:
                    # Explicitly handle type casting for comparison
                    # If signature value's type can be losslessly converted to the column's type, do it.
                    if pd.api.types.is_numeric_dtype(col_series.dtype) and isinstance(value, (int, float)):
                         # Convert both to float for safe comparison (e.g., int 10 and float 10.0)
                        current_mask = col_series.astype(float) == float(value)
                    elif pd.api.types.is_string_dtype(col_series.dtype) or col_series.dtype == 'object':
                        # Convert both to string for comparison
                        current_mask = col_series.astype(str) == str(value)
                    else: # Fallback to general equality
                        current_mask = col_series == value
                except Exception:
                    pass
                mask &= current_mask

                # --- START DEBUG LOGGING (for fake_fp_sig_1 only) ---
                if is_debug_signature:
                    matches_found = mask.sum()
                    logger.info(f"  [DEBUG] After Key '{col}', matches remaining: {matches_found}")
                # --- END DEBUG LOGGING ---

                if not mask.any():
                    break
            else:
                if is_debug_signature:
                    logger.warning(f"  [DEBUG] Skipping condition: Key '{col}' not in DataFrame columns.")
                valid_signature_for_mask = False
                break
        if not valid_signature_for_mask:
            mask = pd.Series(False, index=df_subset.index)
    except Exception as e:
        logger.error(f"Error creating mask for signature {sig_id} in worker: {e}")
        mask = pd.Series(False, index=df_subset.index)
    
    return sig_id, mask


# Helper function for parallel NRA calculation (MOVED TO TOP LEVEL)
def _calculate_nra_for_alert_task(args_nra):
    i_nra, t_i_nra, src_ip_i_nra, dst_ip_i_nra, all_timestamps_nra, all_src_ips_nra, all_dst_ips_nra, t0_nra_val, n0_nra_val_local = args_nra
    
    t_start_nra = t_i_nra - pd.Timedelta(seconds=t0_nra_val)
    t_end_nra = t_i_nra + pd.Timedelta(seconds=t0_nra_val)

    start_idx_nra = all_timestamps_nra.searchsorted(t_start_nra, side='left')
    end_idx_nra = all_timestamps_nra.searchsorted(t_end_nra, side='right')

    if start_idx_nra >= end_idx_nra:
        nra_val = 0
    else:
        window_src_ips_nra = all_src_ips_nra.iloc[start_idx_nra:end_idx_nra]
        window_dst_ips_nra = all_dst_ips_nra.iloc[start_idx_nra:end_idx_nra]
        
        src_match_mask_nra = np.logical_or(window_src_ips_nra == src_ip_i_nra, window_src_ips_nra == dst_ip_i_nra)
        dst_match_mask_nra = np.logical_or(window_dst_ips_nra == src_ip_i_nra, window_dst_ips_nra == dst_ip_i_nra)
        combined_ip_mask_nra = np.logical_or(src_match_mask_nra, dst_match_mask_nra)
        nra_val = np.sum(combined_ip_mask_nra)
        
    return min(nra_val, n0_nra_val_local) / n0_nra_val_local

# 1. Apply a signature to create an alert (Optimized Vectorized Version)
def apply_signatures_to_dataset(df, signatures, base_time=datetime(2025, 4, 14, 12, 0, 0)):
    """
    Applies signatures to a DataFrame using vectorized operations for potentially faster performance.

    Args:
        df (pd.DataFrame): Input data, pre-processed (e.g., group mapped).
        signatures (list): List of signature dictionaries, each with 'id', 'name',
                           and 'rule_dict' containing the rule conditions.
        base_time (datetime): Base timestamp for alerts.

    Returns:
        pd.DataFrame: DataFrame containing generated alerts.
    """
    # --- NEW: Create a copy to avoid fragmentation and SettingWithCopyWarning ---
    df = df.copy()
    # --- END NEW ---

    alerts = []
    # Preview the label column name in the source data
    label_col_name = None
    label_cols_present = []
    for col in ['label', 'class', 'Class']:
        if col in df.columns:
            label_cols_present.append(col)
            if label_col_name is None: # Use the first found label column
                label_col_name = col

    # Preserve original index by adding it as a column before any potential resets.
    df['_original_index'] = df.index

    # Ensure input DataFrame index is unique if it's not already
    original_df_index = df.index # Preserve original index
    if not df.index.is_unique:
        print("Warning: DataFrame index is not unique. Resetting index for processing.")
        df = df.reset_index(drop=True)


    # Initialize temporary columns to store results (indexes to calculate matched signature ID and time)
    # Use a temporary DataFrame to avoid modifying the original df if passed by reference elsewhere
    temp_df = pd.DataFrame(index=df.index)
    temp_df['_match_sig_id'] = pd.NA # Use pandas NA for better compatibility
    temp_df['_row_index'] = np.arange(len(df)) # For time calculation

    # Create signature_id and name mapping (pre-generate for faster lookup)
    sig_id_to_name = {s.get('id'): s.get('name', 'UNKNOWN_NAME') for s in signatures if s.get('id')}

    # Prepare data subset for workers - select all unique keys from all signatures
    all_sig_keys = set()
    for sig_info in signatures:
        if 'rule_dict' in sig_info and isinstance(sig_info['rule_dict'], dict):
            all_sig_keys.update(sig_info['rule_dict'].keys())
    
    # Ensure all keys are actual columns in df
    df_cols_for_worker = [key for key in all_sig_keys if key in df.columns]
    if not df_cols_for_worker and all_sig_keys: # Some keys specified but none are in df
        print("Warning: None of the keys specified in signature rules are present in the DataFrame. No alerts will be generated.")
        return pd.DataFrame()
    elif not df_cols_for_worker: # No keys in any signature rule_dict
        print("Info: No rule keys found in any signature. No rule-based alerts will be generated.")
        # Depending on desired behavior, could return empty or proceed if other alert types are possible
        # For now, assume rule-based is primary, so return empty.
        return pd.DataFrame()
        
    df_subset_for_workers = df[df_cols_for_worker].copy()

    # Iterate through each signature condition and apply vectorized approach
    # Parallel processing for applying signatures
    #tasks = [(df_subset_for_workers, sig_info, df_cols_for_worker) for sig_info in signatures]
    # The tasks are now just the list of signatures, not tuples with the dataframe
    tasks = signatures
    
    # Sort signatures by some criteria (e.g. complexity or specificity) if strict first-match priority is needed
    # For now, we assume the order in `signatures` list is acceptable or that subsequent logic handles multiple matches if any.

    if tasks:
        num_processes = min(len(tasks), multiprocessing.cpu_count())
        # print(f"[ApplySigs] Using {num_processes} processes for {len(tasks)} signatures.")
        try:
            # --- MODIFIED: Use the efficient worker initializer pattern ---
            with multiprocessing.Pool(
                processes=num_processes,
                initializer=_init_worker_apply_sigs,
                initargs=(df_subset_for_workers, df_cols_for_worker)
            ) as pool:
                # Pass only the signature info to the worker
                results_iterator = pool.imap(_apply_single_signature_worker, tasks)
                pbar = tqdm(results_iterator, total=len(tasks), desc="[EVAL] Applying Signatures")
                results_parallel = list(pbar)
        except Exception as e:
            print(f"Error during parallel signature application: {e}. Falling back to sequential.")
            #results_parallel = [_apply_single_signature_task(task) for task in tasks]
            # MODIFIED: Manually initialize for sequential fallback and use the same worker function
            _init_worker_apply_sigs(df_subset_for_workers, df_cols_for_worker)
            results_parallel = []
            pbar_seq = tqdm(tasks, desc="[EVAL] Applying Signatures (Sequential Fallback)")
            for task in pbar_seq:
                #results_parallel.append(_apply_single_signature_task(task))
                results_parallel.append(_apply_single_signature_worker(task))
            
            # Clean up the global variable after the sequential run
            _worker_data_apply_sigs = None

        # Process parallel results to update temp_df['_match_sig_id']
        # This needs to respect the "first match wins" logic implicitly handled by sequential iteration before.
        # Iterate through results (which are in the original order of signatures)
        for sig_id, sig_mask in results_parallel:
            if sig_mask.any():
                # Apply this signature's matches only to rows that haven't been matched yet
                unmatched_and_current_match_indices = df.index[sig_mask & temp_df['_match_sig_id'].isna()]
                if not unmatched_and_current_match_indices.empty:
                    temp_df.loc[unmatched_and_current_match_indices, '_match_sig_id'] = sig_id
    
    # Filter only matched alerts (Use temp_df to filter)
    alerts_df_raw = temp_df[temp_df['_match_sig_id'].notna()].copy()
    # Join back with original df to get necessary columns like labels AND the original index
    # Ensure join works correctly even if index was reset
    columns_to_join = [col for col in label_cols_present if col in df.columns]
    if '_original_index' in df.columns:
        columns_to_join.append('_original_index')

    alerts_df_raw = alerts_df_raw.join(df[columns_to_join], lsuffix='_left')


    if alerts_df_raw.empty:
        print("Info: No alerts generated after applying all signatures.")
        return pd.DataFrame()

    print(f"Info: Generated {len(alerts_df_raw)} raw alerts.")

    # Create final alert DataFrame
    alerts_final = pd.DataFrame({
        'alert_index': alerts_df_raw['_original_index'].values, # Use the preserved original index
        'timestamp': alerts_df_raw['_row_index'].apply(lambda i: base_time + timedelta(seconds=i * 2)),
        'src_ip': [f"192.168.1.{random.randint(1, 254)}" for _ in range(len(alerts_df_raw))],
        'dst_ip': [f"10.0.0.{random.randint(1, 254)}" for _ in range(len(alerts_df_raw))],
        'signature_id': alerts_df_raw['_match_sig_id'],
        'signature_name': alerts_df_raw['_match_sig_id'].map(sig_id_to_name)
    })

    # Copy label information from original data
    # Use original_df_index if df was reset, to map back to original data if needed for labels
    # However, alerts_df_raw.join(df[label_cols_present]) should handle this if df (with potentially reset index) was used.
    # The crucial part is that alerts_df_raw.index aligns with what was used for the join.
    for col in label_cols_present:
         if col in alerts_df_raw.columns: # Check if column exists after join
            alerts_final[col] = alerts_df_raw[col].values
         else:
             print(f"Warning: Label column '{col}' not found in alerts_df_raw after join.")

    return alerts_final


# --- Start Replacement ---
# 2. Paper-based false positive determination (HAF Optimized, NRA slightly optimized)
def calculate_fp_scores(alerts_df: pd.DataFrame, attack_free_df: pd.DataFrame, 
                        t0_nra: int = 60, n0_nra: int = 20, 
                        lambda_haf: float = 100.0, 
                        lambda_ufp: float = 10.0, 
                        belief_threshold: float = 0.5,
                        combine='max', 
                        file_type: str = None):
    """Calculates FP scores (NRA, HAF, UFP) with HAF optimized and NRA slightly improved."""
    if alerts_df.empty:
        print("Warning: calculate_fp_scores received an empty alerts_df. Returning empty DataFrame.")
        # Return DataFrame with expected columns for downstream compatibility
        # Make sure columns match those expected by evaluate_false_positives
        return pd.DataFrame(columns=['nra_score', 'haf_score', 'ufp_score', 'belief', 'is_false_positive', 'signature_id', 'timestamp', 'src_ip', 'dst_ip', 'signature_name']) # Added signature_name if needed later

    df = alerts_df.copy()
    # Ensure timestamp is datetime and handle potential errors
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        print(f"Error converting timestamp column to datetime: {e}. Attempting to continue, but results may be affected.")
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Drop rows where timestamp conversion failed
    initial_rows = len(df)
    df.dropna(subset=['timestamp'], inplace=True)
    if len(df) < initial_rows:
         print(f"Warning: Dropped {initial_rows - len(df)} rows due to invalid timestamps.")
    if df.empty:
        print("Warning: All rows dropped after handling invalid timestamps.")
        return pd.DataFrame(columns=['nra_score', 'haf_score', 'ufp_score', 'belief', 'is_false_positive', 'signature_id', 'timestamp', 'src_ip', 'dst_ip', 'signature_name'])

    n0_nra_to_use = n0_nra
    if file_type and file_type in ['DARPA', 'DARPA98']:
        n0_nra_to_use = 10
        print(f"INFO: Using adjusted n0_nra = {n0_nra_to_use} for file type '{file_type}'.")

    # --- NRA Optimization Step 1: Sort DataFrame by timestamp BEFORE the loop ---
    print("Sorting alerts by timestamp for NRA calculation...")
    df.sort_values(by='timestamp', inplace=True)
    df_original_index = df.index
    df.reset_index(drop=True, inplace=True)
    n = len(df)
    print("Sorting finished.")

    # --- 1. NRA Calculation (Parallelized) ---
    print("Calculating NRA scores (using itertuples with searchsorted)...")
    nra_scores = []
    # Need to make sure 'src_ip' and 'dst_ip' actually exist
    required_cols_nra = ['timestamp', 'src_ip', 'dst_ip']
    missing_cols = [col for col in required_cols_nra if col not in df.columns]
    if missing_cols:
        print(f"Error: Required columns for NRA {missing_cols} not found.")
        df['nra_score'] = 0.0 # Example: Set default NRA score
    else:
        # Create timestamp Series once for efficient filtering
        timestamps = df['timestamp']
        # Prepare src_ip and dst_ip Series (or arrays) for isin check
        src_ips = df['src_ip']
        dst_ips = df['dst_ip']

        # Prepare tasks for parallel NRA calculation
        nra_tasks = []
        for i_task in range(n):
            nra_tasks.append((
                i_task, timestamps.iloc[i_task], src_ips.iloc[i_task], dst_ips.iloc[i_task],
                timestamps, src_ips, dst_ips, # Pass the full series for windowing
                t0_nra, n0_nra_to_use
            ))

        if nra_tasks:
            num_processes_nra = min(len(nra_tasks), multiprocessing.cpu_count())
            # print(f"[CalcFP NRA] Using {num_processes_nra} processes for {len(nra_tasks)} alerts.")
            try:
                with multiprocessing.Pool(processes=num_processes_nra) as pool:
                    nra_scores = pool.map(_calculate_nra_for_alert_task, nra_tasks)
            except Exception as e_nra:
                print(f"Error during parallel NRA calculation: {e_nra}. Falling back to sequential.")
                # Sequential fallback (original loop)
                nra_scores = []
                for i_seq in range(n):
                    t_i_seq = timestamps.iloc[i_seq]
                    src_ip_i_seq = src_ips.iloc[i_seq]
                    dst_ip_i_seq = dst_ips.iloc[i_seq]
                    t_start_seq = t_i_seq - pd.Timedelta(seconds=t0_nra)
                    t_end_seq = t_i_seq + pd.Timedelta(seconds=t0_nra)
                    start_idx_seq = timestamps.searchsorted(t_start_seq, side='left')
                    end_idx_seq = timestamps.searchsorted(t_end_seq, side='right')
                    if start_idx_seq >= end_idx_seq:
                        nra_seq = 0
                    else:
                        window_src_ips_seq = src_ips.iloc[start_idx_seq:end_idx_seq]
                        window_dst_ips_seq = dst_ips.iloc[start_idx_seq:end_idx_seq]
                        src_match_mask_seq = np.logical_or(window_src_ips_seq == src_ip_i_seq, window_src_ips_seq == dst_ip_i_seq)
                        dst_match_mask_seq = np.logical_or(window_dst_ips_seq == src_ip_i_seq, window_dst_ips_seq == dst_ip_i_seq)
                        combined_ip_mask_seq = np.logical_or(src_match_mask_seq, dst_match_mask_seq)
                        nra_seq = np.sum(combined_ip_mask_seq)
                    nra_scores.append(min(nra_seq, n0_nra_to_use) / n0_nra_to_use)
                    # Optional progress for sequential fallback
                    # if (i_seq + 1) % 50000 == 0: print(f"  NRA sequential progress: {i_seq + 1}/{n}")
        else: # No tasks
            nra_scores = [0.0] * n # Or handle as appropriate for empty df

    # Assign after the loop completes
    df['nra_score'] = nra_scores

    # Restore original index if it was stored
    df.index = df_original_index

    print("NRA calculation finished.")

    # --- 2. HAF Calculation (Optimized using groupby and diff) ---
    print("Calculating HAF scores (vectorized)...")
    # Need signature_id and timestamp
    if 'signature_id' not in df.columns or 'timestamp' not in df.columns:
         print("Error: 'signature_id' or 'timestamp' columns missing for HAF calculation.")
         df['haf_score'] = 0.0 # Set default HAF score
    else:
        # Ensure sorting is done on the DataFrame with the correct index if needed later
        df_sorted_haf = df.sort_values(by=['signature_id', 'timestamp']).copy()
        df_sorted_haf['time_diff_prev'] = df_sorted_haf.groupby('signature_id')['timestamp'].diff().dt.total_seconds()
        df_sorted_haf['time_diff_next'] = df_sorted_haf.groupby('signature_id')['timestamp'].diff(-1).dt.total_seconds().abs()
        df_sorted_haf['mtd'] = df_sorted_haf[['time_diff_prev', 'time_diff_next']].abs().min(axis=1, skipna=True)
        df_sorted_haf['mtd'].fillna(np.inf, inplace=True)
        df_sorted_haf['fi'] = 1 / (1 + df_sorted_haf['mtd'])

        sig_stats = df_sorted_haf.groupby('signature_id')['timestamp'].agg(['min', 'max', 'count'])
        sig_stats['duration'] = (sig_stats['max'] - sig_stats['min']).dt.total_seconds()
        sig_stats['duration'] = sig_stats['duration'].clip(lower=0) # Ensure non-negative
        # Calculate avg_interval carefully for count=1 case
        sig_stats['avg_interval'] = np.where(sig_stats['count'] > 1, sig_stats['duration'] / (sig_stats['count'] - 1), np.inf)
        # Handle avg_interval=0 or inf for saf calculation
        sig_stats['saf'] = np.where(sig_stats['avg_interval'] > 1e-9, 1 / sig_stats['avg_interval'], np.inf)
        saf_map = sig_stats['saf']
        # Map saf back using the index of df_sorted_haf, then align with original df index
        df_sorted_haf['saf'] = df_sorted_haf['signature_id'].map(saf_map).fillna(np.inf) # Default saf to inf if not found


        df_sorted_haf['nf'] = df_sorted_haf['fi'] / df_sorted_haf['saf']
        df_sorted_haf['nf'].replace([np.inf, -np.inf, np.nan], 0, inplace=True) # Handle inf/nan from division

        df_sorted_haf['haf_score'] = (df_sorted_haf['nf'].clip(upper=lambda_haf) / lambda_haf)

        # Merge HAF scores back using the DataFrame's index (should align correctly now)
        # Ensure the index used for merging is the one from the original df
        df = df.merge(df_sorted_haf[['haf_score']], left_index=True, right_index=True, how='left')
        df['haf_score'].fillna(0, inplace=True)
    print("HAF calculation finished.")


    # --- 3. UFP Calculation (Optimized using map) ---
    print("Calculating UFP scores...")
    if attack_free_df is None or attack_free_df.empty or 'signature_id' not in attack_free_df.columns:
        print("Warning: attack_free_df is unsuitable for UFP. Scores set to 0.")
        df['ufp_score'] = 0.0
    elif 'signature_id' not in df.columns:
         print("Error: 'signature_id' column missing from main DataFrame for UFP.")
         df['ufp_score'] = 0.0
    else:
        af_counts = attack_free_df['signature_id'].value_counts()
        af_total = len(attack_free_df)
        af_freqs_map = (af_counts / af_total).to_dict() if af_total > 0 else {}
        test_counts_map = df['signature_id'].value_counts().to_dict()
        n_test = len(df)

        # Use map for potentially faster lookup
        test_freqs_series = df['signature_id'].map(test_counts_map).fillna(0) / n_test if n_test > 0 else pd.Series(0.0, index=df.index)
        af_freqs_series = df['signature_id'].map(af_freqs_map).fillna(1e-9) # Map known af freqs, default small

        # Calculate ratio avoiding division by zero
        ratio = np.where(af_freqs_series < 1e-9, lambda_ufp, test_freqs_series / af_freqs_series)
        df['ufp_score'] = np.minimum(ratio, lambda_ufp) / lambda_ufp
        df['ufp_score'].fillna(0, inplace=True) # Handle potential NaNs from calculation

    print("UFP calculation finished.")

    # --- 4. Determining combinations and false positives (No change) ---
    print("Combining scores...")
    valid_combine_methods = ['max', 'avg', 'min']
    if combine not in valid_combine_methods:
        print(f"Warning: Invalid combine method '{combine}'. Defaulting to 'max'.")
        combine = 'max' # Default to max

    score_cols = ['nra_score', 'haf_score', 'ufp_score']
    # Ensure score columns exist before attempting calculation
    missing_score_cols = [col for col in score_cols if col not in df.columns]
    if missing_score_cols:
        print(f"Error: Score columns {missing_score_cols} not found for combining.")
        df['belief'] = 0.0
        df['is_false_positive'] = True
    else:
        # Fill NaN in score columns before combining to avoid NaN results
        df[score_cols] = df[score_cols].fillna(0)
        if combine == 'max':
            df['belief'] = df[score_cols].max(axis=1)
        elif combine == 'avg':
            df['belief'] = df[score_cols].mean(axis=1)
        elif combine == 'min':
            df['belief'] = df[score_cols].min(axis=1)

    df['is_false_positive'] = df['belief'] < belief_threshold

    print("FP score calculation complete.")
    # Ensure all required columns are present before returning
    for col in score_cols + ['belief', 'is_false_positive']:
         if col not in df.columns:
              df[col] = 0 # Add missing columns with default value

    return df

# --- End Replacement ---


# 3. FP propensity summary by signature
def summarize_fp_by_signature(result_df: pd.DataFrame):
    summary = result_df.groupby(['signature_id', 'signature_name']).agg(
        total_alerts=('is_false_positive', 'count'),
        false_positives=('is_false_positive', 'sum')
    )
    summary['fp_rate'] = summary['false_positives'] / summary['total_alerts']
    return summary.sort_values('fp_rate', ascending=False)

def check_alert_frequency(alerts_df, time_window=3600, threshold_multiplier=3):
    """
    1. Check if adding a signature causes excessive alerts
    
    Args:
        alerts_df (DataFrame): Alert data
        time_window (int): Time window (seconds)
        threshold_multiplier (float): How many times the average to consider it excessive
    
    Returns:
        dict: Whether each signature has excessive alerts
    """
    alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
    
    # Total monitoring period
    total_time = (alerts_df['timestamp'].max() - alerts_df['timestamp'].min()).total_seconds()
    total_windows = max(1, total_time / time_window)
    
    # Calculate the alert frequency per signature
    signature_counts = alerts_df['signature_id'].value_counts()
    average_alerts_per_window = signature_counts / total_windows
    
    # Calculate the number of alerts per hour
    alerts_per_hour = {}
    excessive_alerts = {}
    
    for sig_id in signature_counts.index:
        sig_alerts = alerts_df[alerts_df['signature_id'] == sig_id]
        
        # Calculate the number of alerts per hour
        hourly_counts = sig_alerts.groupby(sig_alerts['timestamp'].dt.hour).size()
        max_hourly = hourly_counts.max()
        avg_hourly = hourly_counts.mean()
        
        alerts_per_hour[sig_id] = max_hourly
        # If the maximum number of alerts per hour is more than threshold_multiplier times the average
        excessive_alerts[sig_id] = max_hourly > (avg_hourly * threshold_multiplier)
    
    return excessive_alerts

def check_superset_signatures(new_signatures, known_fp_signatures):
    """
    2. Check if the newly created signature is a superset of existing FP signatures
    
    Args:
        new_signatures (list): List of new signatures
        known_fp_signatures (list): List of existing FP signatures
    
    Returns:
        dict: Whether each new signature is a superset
    """
    superset_check = {}
    
    for new_sig in new_signatures:
        # Check if the structure is as expected
        if 'signature_name' not in new_sig or not isinstance(new_sig['signature_name'], dict) or 'Signature_dict' not in new_sig['signature_name']:
            print(f"Warning: Skipping superset check for signature due to unexpected structure: {new_sig.get('id', 'N/A')}")
            continue 
        new_sig_dict = new_sig['signature_name']['Signature_dict']
        is_superset = False
        
        for fp_sig in known_fp_signatures:
             # Check FP signature structure too
             if 'signature_name' not in fp_sig or not isinstance(fp_sig['signature_name'], dict) or 'Signature_dict' not in fp_sig['signature_name']:
                 print(f"Warning: Skipping known FP signature in superset check due to unexpected structure.")
                 continue
             fp_sig_dict = fp_sig['signature_name']['Signature_dict']
            
             # Check if all conditions of fp_sig are included in new_sig
             if isinstance(new_sig_dict, dict) and isinstance(fp_sig_dict, dict): # Ensure they are dicts
                 if all(k in new_sig_dict and new_sig_dict[k] == v for k, v in fp_sig_dict.items()):
                     is_superset = True
                     break
             else:
                  print(f"Warning: Invalid dictionary types for superset check. New: {type(new_sig_dict)}, FP: {type(fp_sig_dict)}")

        
        # Use signature ID if name is complex/missing
        sig_key = new_sig.get('id', new_sig.get('signature_name', str(new_sig))) 
        superset_check[sig_key] = is_superset
    
    return superset_check

def check_temporal_ip_patterns(alerts_df, time_window=300):
    """
    3. Check if similar source/destination IP alerts occur in a temporal manner when an attack occurs
    
    Args:
        alerts_df (DataFrame): Alert data
        time_window (int): Time window to check (seconds)
    
    Returns:
        dict: Pattern score for each signature
    """
    if 'timestamp' not in alerts_df.columns or 'signature_id' not in alerts_df.columns:
         print("Error: Required columns 'timestamp' or 'signature_id' not found for temporal pattern check.")
         return {}
         
    alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'], errors='coerce')
    alerts_df.dropna(subset=['timestamp'], inplace=True) # Drop rows where conversion failed
    if alerts_df.empty:
        return {}
        
    pattern_scores = {}
    
    # Pre-calculate required columns if they exist
    has_src_ip = 'src_ip' in alerts_df.columns
    has_dst_ip = 'dst_ip' in alerts_df.columns
    if not has_src_ip or not has_dst_ip:
        print("Warning: 'src_ip' or 'dst_ip' columns missing. IP pattern scores will be 0.")

    # Use groupby for potentially faster processing per signature
    for sig_id, sig_alerts_group in alerts_df.groupby('signature_id'):
        sig_alerts = sig_alerts_group.sort_values('timestamp').copy() # Sort within group
        
        if len(sig_alerts) < 2:
            pattern_scores[sig_id] = 0
            continue
        
        # Use rolling window or other vectorized approaches if possible?
        # Sticking to iterrows for now as rolling window logic is complex here
        ip_pattern_scores = []
        timestamps_np = sig_alerts['timestamp'].to_numpy() # For faster lookup maybe?

        for idx, alert in sig_alerts.iterrows():
            # Find window using searchsorted on the group's sorted timestamps
            t_i = alert['timestamp']
            t_start = t_i - pd.Timedelta(seconds=time_window)
            t_end = t_i + pd.Timedelta(seconds=time_window)

            # Find indices for the window within the current group
            start_idx = np.searchsorted(timestamps_np, t_start, side='left')
            end_idx = np.searchsorted(timestamps_np, t_end, side='right')
            
            # Slice the group DataFrame using iloc for the window
            window_alerts = sig_alerts.iloc[start_idx:end_idx]


            if window_alerts.empty or len(window_alerts) <= 1: # Need at least one other alert in window
                ip_pattern_scores.append(0) # Or just continue? Depends on desired score calculation
                continue

            # Calculate IP similarity only if IP columns exist
            same_src = 0
            same_dst = 0
            if has_src_ip:
                same_src = (window_alerts['src_ip'] == alert['src_ip']).sum()
            if has_dst_ip:
                same_dst = (window_alerts['dst_ip'] == alert['dst_ip']).sum()
            
            # Time proximity weight
            time_diffs = np.abs((window_alerts['timestamp'] - alert['timestamp']).dt.total_seconds())
            # Avoid division by zero if time_diff is exactly 0 (the alert itself)
            time_weights = 1 / (1 + time_diffs[time_diffs > 1e-9]) # Exclude self comparison potentially
            
            # Calculate IP pattern score
            window_size = len(window_alerts) # Already checked > 1
            ip_similarity = (same_src + same_dst) / (window_size * 2) if (has_src_ip or has_dst_ip) else 0
            # Handle case where time_weights might be empty if all diffs are zero
            time_density = time_weights.mean() if not time_weights.empty else 0
            # Align the following lines correctly within the loop
            pattern_score = (ip_similarity + time_density) / 2
            ip_pattern_scores.append(pattern_score)
        
        # Final pattern score for each signature (after loop)
        pattern_scores[sig_id] = np.mean(ip_pattern_scores) if ip_pattern_scores else 0
    
    return pattern_scores

def is_superset_of_known_fps(current_sig_dict, known_fp_sig_dicts):
    """Verify that the current signature is a superset of one of the known FP signatures"""
    if not known_fp_sig_dicts or not isinstance(known_fp_sig_dicts, list):
        return False # False if there is no known FP
    if not current_sig_dict or not isinstance(current_sig_dict, dict):
        return False # False if there is no current signature

    for fp_sig_dict in known_fp_sig_dicts:
        if not isinstance(fp_sig_dict, dict): continue # Skip if FP signature format error

        # Check if all items (keys, values) in fp_sig_dict exist in current_sig_dict
        try:
            # Use items view for potentially faster check in Python 3
            is_superset = fp_sig_dict.items() <= current_sig_dict.items()
        except TypeError:
            # Handle cases where values might not be comparable (e.g., NaN)
            # Fallback to original check
            is_superset = all(
                 k in current_sig_dict and current_sig_dict[k] == v for k, v in fp_sig_dict.items()
                )

        # Original logic checked only subset, let's keep the superset check
        # is_superset = all(
        #    item in current_sig_dict.items() for item in fp_sig_dict.items()
        # )

        # Ensure it's a PROPER superset (current longer than fp)
        if is_superset and len(current_sig_dict) > len(fp_sig_dict):
             # print(f"Debug: {current_sig_dict} is superset of {fp_sig_dict}")
             return True
    return False

def evaluate_false_positives(
        alerts_df: pd.DataFrame,
        current_signatures_map: dict,
        known_fp_sig_dicts: list = None,
        attack_free_df: pd.DataFrame = None,
        t0_nra: int = 60,
        n0_nra: int = 20,
        lambda_haf: float = 100.0,
        lambda_ufp: float = 10.0,
        combine_method: str = 'max',
        belief_threshold: float = 0.5,
        superset_strictness: float = 0.9,
        file_type: str = None):
    """
    Calculate FP scores and apply superset logic to determine final FP decision.
    """
    if alerts_df.empty:
         print("Warning: evaluate_false_positives received empty alerts_df. No FP analysis performed.")
         return pd.DataFrame(columns=['signature_id', 'signature_name', 'nra_score', 'haf_score', 'ufp_score', 'belief', 'is_superset', 'applied_threshold', 'likely_false_positive'])

    if attack_free_df is None:
         print("Warning: attack_free_df not provided for UFP calculation.")
         # Create an empty DataFrame with necessary columns to avoid errors in calculate_fp_scores
         attack_free_df = pd.DataFrame(columns=['signature_id'])


    # 1. Calculate basic FP scores (use received parameters)
    print("Calculating initial FP scores...")
    fp_scores_df = calculate_fp_scores(
        alerts_df,
        attack_free_df,
        t0_nra=t0_nra,
        n0_nra=n0_nra,
        lambda_haf=lambda_haf,
        lambda_ufp=lambda_ufp,
        combine=combine_method,
        file_type=file_type
    )
    print("Initial FP score calculation finished.")

    # Check if necessary columns exist after calculate_fp_scores
    required_fp_cols = ['signature_id', 'belief', 'nra_score', 'haf_score', 'ufp_score']
    if not all(col in fp_scores_df.columns for col in required_fp_cols):
        print("Error: calculate_fp_scores did not return required columns. Cannot proceed.")
        return pd.DataFrame(columns=['signature_id', 'signature_name', 'nra_score', 'haf_score', 'ufp_score', 'belief', 'is_superset', 'applied_threshold', 'likely_false_positive'])


    # Initialize for saving results
    fp_results = fp_scores_df.copy()
    # Ensure required columns for the loop exist
    fp_results['is_superset'] = False
    fp_results['applied_threshold'] = belief_threshold
    fp_results['likely_false_positive'] = False

    # Add signature_name if it's missing (needed for summarize step)
    if 'signature_name' not in fp_results.columns and 'signature_id' in fp_results.columns:
         print("Adding signature_name based on current_signatures_map...")
         sig_id_to_name_map = {sig_id: sig_data.get('name', 'UNKNOWN') for sig_id, sig_data in current_signatures_map.items()}
         fp_results['signature_name'] = fp_results['signature_id'].map(sig_id_to_name_map)


    # If known FP list is not provided, initialize
    if known_fp_sig_dicts is None:
        known_fp_sig_dicts = []

    # 2. Check superset and determine final FP decision for each signature
    # Avoid iterrows if possible, maybe apply a function?
    # For now, keep iterrows but ensure it works correctly
    print("Applying superset logic and final FP decision...")
    num_rows_fp = len(fp_results)
    for index, row in fp_results.iterrows():
        sig_id = row['signature_id']
        belief_score = row['belief']

        # Get current signature dictionary
        current_sig_dict = current_signatures_map.get(sig_id)
        if not current_sig_dict:
            # print(f"Warning: Signature dictionary not found for {sig_id} in current_signatures_map.")
            continue # Skip if not in map

        # Check superset (using the potentially optimized is_superset_of_known_fps)
        is_super = is_superset_of_known_fps(current_sig_dict, known_fp_sig_dicts)
        fp_results.loc[index, 'is_superset'] = is_super

        # Apply threshold
        threshold = belief_threshold * superset_strictness if is_super else belief_threshold
        fp_results.loc[index, 'applied_threshold'] = threshold

        # Final FP decision
        fp_results.loc[index, 'likely_false_positive'] = belief_score < threshold

        # Optional progress
        # if (fp_results.index.get_loc(index) + 1) % 50000 == 0:
        #      print(f"  Superset/Final FP progress: {fp_results.index.get_loc(index) + 1}/{num_rows_fp}")

    print("Superset logic and final FP decision applied.")

    # Return final result DataFrame (before summarization)
    # Ensure all expected columns by summarize_fp_results are present
    final_cols = ['signature_id', 'signature_name', 'nra_score', 'haf_score', 'ufp_score', 'belief', 'is_superset', 'applied_threshold', 'likely_false_positive']
    missing_final_cols = [col for col in final_cols if col not in fp_results.columns]
    if missing_final_cols:
         print(f"Warning: Columns {missing_final_cols} missing before returning from evaluate_false_positives.")
         # Add missing columns with default values
         for col in missing_final_cols:
             if col == 'signature_name':
                 fp_results[col] = 'UNKNOWN'
             elif col == 'is_superset' or col == 'likely_false_positive':
                  fp_results[col] = False
             else:
                  fp_results[col] = 0.0


    return fp_results[final_cols]

def summarize_fp_results(detailed_fp_results: pd.DataFrame):
     """ Summarize FP decision results by group """
     if detailed_fp_results.empty:
          print("Info: detailed_fp_results is empty in summarize_fp_results.")
          return pd.DataFrame()

     # Ensure required columns exist
     required_summary_cols = ['signature_id', 'signature_name', 'likely_false_positive', 'belief', 'is_superset', 'applied_threshold']
     if not all(col in detailed_fp_results.columns for col in required_summary_cols):
          print(f"Error: Missing required columns in detailed_fp_results for summary. Need: {required_summary_cols}")
          return pd.DataFrame()


     summary = detailed_fp_results.groupby(['signature_id', 'signature_name']).agg(
         alerts_count=('likely_false_positive', 'size'), # Total alerts (group size)
         likely_fp_count=('likely_false_positive', 'sum'), # Number of alerts determined as FP
         avg_belief=('belief', 'mean'),
         is_superset=('is_superset', 'first'), # Superset status (same for each signature)
         applied_threshold=('applied_threshold', 'first') # Applied threshold
     ).reset_index()

     # Calculate FP rate safely avoiding division by zero
     summary['likely_fp_rate'] = np.where(summary['alerts_count'] > 0,
                                         summary['likely_fp_count'] / summary['alerts_count'],
                                         0)
     # Determine final FP status based on rate (e.g., > 50%)
     summary['final_likely_fp'] = summary['likely_fp_rate'] > 0.5

     # Ensure correct final columns are returned
     final_summary_cols = ['signature_id', 'signature_name', 'alerts_count', 'likely_fp_count', 'likely_fp_rate', 'avg_belief', 'is_superset', 'applied_threshold', 'final_likely_fp']

     # Add any missing columns from the final list if needed (shouldn't be necessary)
     for col in final_summary_cols:
         if col not in summary.columns:
              print(f"Warning: Column '{col}' missing in final summary. Adding default.")
              if col == 'signature_name':
                   summary[col] = 'UNKNOWN'
              elif col == 'is_superset' or col == 'final_likely_fp':
                   summary[col] = False
              else:
                   summary[col] = 0.0


     return summary[final_summary_cols]
