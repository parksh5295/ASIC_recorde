# Evaluate TP, TN, FP, FN by comparing each signature (list dictionary) to a real dataset
# Return: [{'Signature_dict': signature_name, 'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}, {}, ...]

import pandas as pd
import numpy as np
import multiprocessing
import traceback
from tqdm import tqdm
from functools import partial
import gc

# --- NEW: Worker Initializer Pattern for Inverted Index ---
# Global variables for each worker process
_worker_inverted_index = None
_worker_signatures = None
_worker_anomaly_data = None


def _normalize_val(v):
    """
    Normalize values so that signature values and dataframe values compare consistently.
    - Numeric types are converted to Python float (or int if safe).
    - Strings are stripped.
    - Others are left as-is.
    """
    try:
        import numpy as _np  # local import to avoid top-level dependency for users without numpy
        if isinstance(v, (_np.integer, int)):
            return int(v)
        if isinstance(v, (_np.floating, float)):
            # use float for consistent string repr across int/float mixes
            return float(v)
    except Exception:
        pass

    # Fallbacks
    if isinstance(v, str):
        return v.strip()
    return v

def _init_worker_inverted_index(inverted_index, signatures, anomaly_data):
    """Initializes global variables for each worker process."""
    global _worker_inverted_index, _worker_signatures, _worker_anomaly_data
    _worker_inverted_index = inverted_index
    _worker_signatures = signatures
    _worker_anomaly_data = anomaly_data

def _build_inverted_index(signatures):
    """Builds an inverted index from signatures for fast lookup."""
    inverted_index = {}
    for i, sig_dict in enumerate(signatures):
        for key, value in sig_dict.items():
            # condition = f"{key}={value}"
            norm_val = _normalize_val(value)
            condition = (key, norm_val)
            if condition not in inverted_index:
                inverted_index[condition] = []
            inverted_index[condition].append(i)
    return inverted_index

def _find_detected_indices_worker(anomaly_indices_chunk):
    """
    Worker function that finds which anomalies in a chunk are detected by any signature.
    Uses the globally available inverted index and signature list.
    """
    global _worker_inverted_index, _worker_signatures, _worker_anomaly_data
    
    detected_indices_in_chunk = set()

    for index in anomaly_indices_chunk:
        try:
            row = _worker_anomaly_data.loc[index]
            
            # Find candidate signatures using the inverted index (union of matched conditions)
            candidate_indices = set()
            #first = True
            for col, value in row.items():
                # condition = f"{col}={value}"
                norm_val = _normalize_val(value)
                condition = (col, norm_val)
                if condition in _worker_inverted_index:
                    '''
                    sig_indices_for_cond = set(_worker_inverted_index[condition])
                    if first:
                        candidate_indices = sig_indices_for_cond
                        first = False
                    else:
                        candidate_indices.intersection_update(sig_indices_for_cond)
                    '''
                    candidate_indices.update(_worker_inverted_index[condition])
            
            if not candidate_indices:
                continue

            # Verify candidate signatures
            for sig_index in candidate_indices:
                signature = _worker_signatures[sig_index]
                # is_match = all(row.get(key) == value for key, value in signature.items())
                is_match = all(_normalize_val(row.get(key)) == _normalize_val(value) for key, value in signature.items())
                
                if is_match:
                    detected_indices_in_chunk.add(index)
                    break # Move to the next row once a match is found
        except KeyError:
            # This might happen if an index from the chunk is not in _worker_anomaly_data
            # Should be rare if data is handled correctly, but good to be safe.
            continue
            
    return detected_indices_in_chunk

# Helper function for parallel signature calculation
def _calculate_single_signature_metrics(args):
    data_subset, signature = args
    # Ensure all keys in signature exist in data_subset columns to avoid KeyError during .eq()
    # This might be overly cautious if data is guaranteed to have all keys, 
    # but helps if signatures can have keys not in data (though that implies a problem upstream)
    valid_signature_keys = [k for k in signature.keys() if k in data_subset.columns]
    if not valid_signature_keys: # or if set(signature.keys()) != set(valid_signature_keys):
        # Handle cases where signature keys are not in data, or are empty.
        # This indicates an issue, perhaps log it. For now, return zero metrics.
        # print(f"Warning: Signature {signature} has keys not in data or is effectively empty against data. Skipping.")
        return {
            'Signature_dict': signature,
            'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0
        }

    # Proceed with calculation only using valid keys present in the data
    # Create a filtered signature for the .eq() comparison
    filtered_signature_for_comparison = {k: signature[k] for k in valid_signature_keys}

    # It is crucial that data_subset passed to this worker function already has the 'label' column.
    # Ensure data[valid_signature_keys] doesn't create an empty DataFrame if valid_signature_keys is empty.
    # The check for `if not valid_signature_keys:` above should handle this.
    matches = data_subset[valid_signature_keys].eq(pd.Series(filtered_signature_for_comparison)).all(axis=1)

    # Calculate TP, FN, FP, TN
    TP = ((matches) & (data_subset['label'] == 1)).sum()
    FN = ((~matches) & (data_subset['label'] == 1)).sum()
    FP = ((matches) & (data_subset['label'] == 0)).sum()
    TN = ((~matches) & (data_subset['label'] == 0)).sum()

    return {
        'Signature_dict': signature,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN
    }

def calculate_signature(data, signatures):
    if not signatures: # Handle empty signatures list
        return []

    # Prepare data for workers: select all unique keys from all signatures plus 'label'
    # This avoids sending the whole dataframe `data` to each worker if it's very wide.
    all_signature_keys = set()
    for sig in signatures:
        all_signature_keys.update(sig.keys())
    
    columns_to_select = list(all_signature_keys)
    if 'label' not in columns_to_select:
        columns_to_select.append('label')
    
    # Ensure all selected columns actually exist in the input dataframe `data`
    existing_columns_in_data = [col for col in columns_to_select if col in data.columns]
    if not existing_columns_in_data or 'label' not in existing_columns_in_data:
        print(f"Warning: 'label' column ({'label' in existing_columns_in_data}) or critical signature columns (all_signature_keys present: {all(k in existing_columns_in_data for k in all_signature_keys)}) missing from data in calculate_signature. Returning empty results.") # More detailed warning
        # It might be better to raise an error or log extensively here
        return [{'Signature_dict': sig, 'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0} for sig in signatures]

    data_subset_for_processing = data[existing_columns_in_data].copy() # Use .copy() to avoid SettingWithCopyWarning on slices if data is a slice

    tasks = [(data_subset_for_processing, sig) for sig in signatures]
    results = []

    if tasks:
        # Restore dynamic process count based on available cores and number of tasks
        num_processes = min(len(tasks), multiprocessing.cpu_count())
        if num_processes == 0 and len(tasks) > 0: # Should not happen if cpu_count() >=1
             num_processes = 1
        
        print(f"[CalcSig] Using {num_processes} processes for {len(tasks)} signatures.")
        try:
            if num_processes > 0:
                 with multiprocessing.Pool(processes=num_processes) as pool:
                    results = pool.map(_calculate_single_signature_metrics, tasks)
            else:
                results = []

        except Exception as e:
            print(f"Error during parallel signature calculation: {e}")
            traceback.print_exc()
            print("Falling back to sequential processing for calculate_signature...")
            results = []
            for task_arg in tasks:
                results.append(_calculate_single_signature_metrics(task_arg))
    
    return results

'''
# Helper function for parallel recall calculation
def _calculate_recall_for_row_chunk(args):
    data_chunk, signatures_list = args
    # Ensure signatures_list is not empty and data_chunk is not empty
    if not signatures_list or data_chunk.empty:
        return 0, 0, 0, 0 # TP, FN, FP, TN

    TP_chunk = 0
    FN_chunk = 0
    FP_chunk = 0
    TN_chunk = 0

    for _, row in data_chunk.iterrows():
        row_satisfied = False
        # Check if the row satisfies any of the signatures
        for sig_item in signatures_list:
            actual_signature = sig_item['signature_name']['Signature_dict']
            # Check if all conditions in the current signature are met by the row
            # Also ensure all keys in actual_signature exist in the row to prevent KeyError
            if all(k in row and row[k] == v for k, v in actual_signature.items()):
                row_satisfied = True
                break # Row matches at least one signature
        
        if row['label'] == 1:
            if row_satisfied:
                TP_chunk += 1
            else:
                FN_chunk += 1
        else: # label == 0
            if row_satisfied:
                FP_chunk += 1
            else:
                TN_chunk += 1
    return TP_chunk, FN_chunk, FP_chunk, TN_chunk
'''

# Tools for evaluating recall in an aggregated signature collection
# --- REWRITTEN FOR PERFORMANCE USING INVERTED INDEX ---
def calculate_signatures(data, signatures):
    if not signatures:
        return 0.0

    # 1. Prepare data and signatures
    actual_signatures = []
    needed_columns = set()
    #processed_signatures_for_workers = []
    for sig_eval_metric in signatures:
        try:
            actual_sig = sig_eval_metric['signature_name']['Signature_dict']
            if isinstance(actual_sig, dict):
                actual_signatures.append(actual_sig)
                needed_columns.update(actual_sig.keys())
        except (KeyError, TypeError):
            continue

    if not actual_signatures:
        return 0.0
    
    # Get anomaly data and indices
    anomaly_data = data[data['label'] == 1]
    total_anomalies = len(anomaly_data)
    if total_anomalies == 0:
        return 0.0 # Or handle as a special case, recall is arguably undefined or 1.0

    #needed_columns.add('label')
    # Select only the columns needed for matching to save memory
    existing_needed_columns = [col for col in needed_columns if col in anomaly_data.columns]
    anomaly_data_subset = anomaly_data[existing_needed_columns].copy()

    # --- DEBUG: upfront visibility on signature/data alignment ---
    try:
        print(f"[CalcRecall Debug] #signatures={len(actual_signatures)}, total_anomalies={total_anomalies}")
        missing_cols = [c for c in needed_columns if c not in anomaly_data.columns]
        print(f"[CalcRecall Debug] Needed cols: {len(needed_columns)}, Missing in data: {missing_cols}")

        sample_n = min(5, len(actual_signatures))
        for i in range(sample_n):
            sig = actual_signatures[i]
            valid_keys = [k for k in sig.keys() if k in anomaly_data_subset.columns]
            if not valid_keys:
                print(f"[CalcRecall Debug] Sig{i}: no valid keys")
                continue
            mask = anomaly_data_subset[valid_keys].eq(pd.Series({k: sig[k] for k in valid_keys})).all(axis=1)
            matches = mask.sum()
            print(f"[CalcRecall Debug] Sig{i}: valid_keys={valid_keys}, matches={matches}")
    except Exception as e:
        print(f"[CalcRecall Debug] Debug logging failed: {e}")
    
    # Small-case brute force path to avoid over-pruning via inverted index intersections
    brute_force_allowed = (total_anomalies <= 20000 and len(actual_signatures) <= 5000)
    detected_indices = set()
    inverted_index = None  # ensure defined for cleanup section

    if brute_force_allowed:
        print(f"[CalcRecall Debug] Using brute-force matching (sig={len(actual_signatures)}, anomalies={total_anomalies})")
        for sig in actual_signatures:
            valid_keys = [k for k in sig.keys() if k in anomaly_data_subset.columns]
            if not valid_keys:
                continue
            # Normalize signature values
            norm_sig = {k: _normalize_val(sig[k]) for k in valid_keys}
            mask = pd.Series(True, index=anomaly_data_subset.index)
            for k in valid_keys:
                col_norm = anomaly_data_subset[k].map(_normalize_val)
                mask &= (col_norm == norm_sig[k])
                if not mask.any():
                    break
            if mask.any():
                detected_indices.update(mask[mask].index.tolist())
    else:
        # 2. Build the inverted index
        print("[CalcRecall] Building inverted index for signatures...")
        inverted_index = _build_inverted_index(actual_signatures)
        
        # 3. Parallel processing using the inverted index
        print(f"[CalcRecall] Evaluating {len(actual_signatures)} signatures against {total_anomalies} anomalies...")
        
        num_processes = multiprocessing.cpu_count()
        anomaly_indices = list(anomaly_data_subset.index)
        
        # Split anomaly indices into chunks for workers
        chunk_size = max(1, (total_anomalies + num_processes - 1) // num_processes)
        index_chunks = [anomaly_indices[i:i + chunk_size] for i in range(0, total_anomalies, chunk_size)]
        
        try:
            # Use Pool with worker initializer to avoid sending large objects repeatedly
            with multiprocessing.Pool(
                processes=num_processes,
                initializer=_init_worker_inverted_index,
                initargs=(inverted_index, actual_signatures, anomaly_data_subset)
            ) as pool:
                
                pbar = tqdm(pool.imap_unordered(_find_detected_indices_worker, index_chunks), total=len(index_chunks), desc="[CalcRecall] Processing Data Chunks")
                
                for detected_set in pbar:
                    detected_indices.update(detected_set)

        except Exception as e:
            print(f"Error during parallel recall calculation: {e}")
            traceback.print_exc()
            # Fallback to sequential processing for debugging or robustness
            print("Falling back to sequential processing for recall calculation...")
            
            # --- NEW: Fallback sequential logic ---
            detected_indices = set() # Reset results from any partial parallel execution
            # Initialize the 'worker' globals in the main process for the sequential run
            _init_worker_inverted_index(inverted_index, actual_signatures, anomaly_data_subset)
            
            pbar_fallback = tqdm(index_chunks, desc="[CalcRecall] Processing Chunks (Sequential Fallback)")
            for chunk in pbar_fallback:
                # Call the worker function directly
                detected_set = _find_detected_indices_worker(chunk)
                detected_indices.update(detected_set)
            # --- END NEW ---

    # 4. Calculate final recall
    recall = len(detected_indices) / total_anomalies if total_anomalies > 0 else 0.0
    print(f"[CalcRecall Debug] Detected size after processing: {len(detected_indices)}")
    
    # Memory cleanup (guard inverted_index for brute-force path)
    try:
        del inverted_index
    except Exception:
        pass
    del anomaly_data_subset, anomaly_data, actual_signatures
    gc.collect()
    
    print(f"[CalcRecall] Finished. total_anomalies={total_anomalies}, detected={len(detected_indices)}, Recall: {recall:.4f}")
    return recall
