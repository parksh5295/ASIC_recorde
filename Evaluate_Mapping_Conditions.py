import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
# --- Dataset supple is no longer needed ---
# from sklearn.model_selection import train_test_split
import multiprocessing
import gc # Import garbage collector

# --- NEW: Level limits for consistency ---
LEVEL_LIMITS_BY_FILE_TYPE = {
    'MiraiBotnet': 3,
    'NSL-KDD': 3,
    'NSL_KDD': 3,
    'DARPA98': 5,
    'DARPA': 5,
    'CICIDS2017': 3,
    'CICIDS': 3,
    'CICModbus23': None,
    'CICModbus': None,
    'IoTID20': None,
    'IoTID': None,
    'CICIoT': 3,
    'CICIoT2023': 3,
    'netML': 5,
    'Kitsune': 3,
    'default': None
}

# --- Project Root Path Correction ---
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except (NameError, IndexError):
    if '..' not in sys.path:
        sys.path.insert(0, '..')

# --- Required Project Modules (aligned with Main_Association_Rule_ex.py) ---
from Modules.Heterogeneous_module import choose_heterogeneous_method
from Modules.Association_module import association_module
from definition.Anomal_Judgment import anomal_judgment_label, anomal_judgment_nonlabel
from Dataset_Choose_Rule.time_save import time_save_csv_mapping_conditions
from utils.remove_rare_columns import remove_rare_columns
from Heterogeneous_Method.separate_group_mapping import map_intervals_to_groups
from Heterogeneous_Method.Interval_normalized import Heterogeneous_Interval_Inverse
from Dataset_Choose_Rule.association_data_choose import get_clustered_data_path # MODIFIED: Import the new function
from Dataset_Choose_Rule.choose_amount_dataset import file_cut
from utils.time_transfer import time_scalar_transfer
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from Modules.Whitelist_module import create_whitelist
from Rebuild_Method.FalsePositive_Check import apply_signatures_to_dataset
from Tuning_hyperparameter.Grid_search import Grid_search_all


# --- Helper function for checking support (from _ex versions) ---
def calculate_support_for_itemset(itemset, df, regrouping_map=None):
    if not itemset or df.empty:
        return 0
    mask = pd.Series([True] * len(df), index=df.index)
    
    for key, value in itemset.items():
        if key in df.columns:
            if regrouping_map and key in regrouping_map:
                # Apply regrouping logic for comparison
                current_r_map = regrouping_map[key]
                # The 'value' from a regrouped signature is already the target group.
                signature_group = value
                # Map the entire dataframe column and check for equality
                mask &= (df[key].map(current_r_map).fillna(-2) == signature_group)
            else:
                # Original direct comparison
                mask &= (df[key] == value)
        else:
            return 0
    return mask.sum() / len(df)

# --- REVISED Performance Calculation ---
def calculate_performance_revised(signatures, test_df, regrouping_map=None):
    if not signatures or test_df.empty:
        return 0.0, 0.0, 0.0

    test_anomalous = test_df[test_df['label'] == 1]
    test_normal = test_df[test_df['label'] == 0]

    total_tp_alerts = 0
    total_fp_alerts = 0
    detected_anomalous_indices = set()

    # Calculate total alerts on anomalous and normal data
    for rule in signatures:
        # TP Alerts
        mask_tp = pd.Series([True] * len(test_anomalous), index=test_anomalous.index)
        for key, value in rule.items():
            if key in test_anomalous.columns:
                if regrouping_map and key in regrouping_map:
                    r_map = regrouping_map[key]
                    # FIX: The 'value' from a regrouped signature is already the target group.
                    sig_group = value
                    mask_tp &= (test_anomalous[key].map(r_map).fillna(-2) == sig_group)
                else:
                    mask_tp &= (test_anomalous[key] == value)
        
        matches_tp = test_anomalous[mask_tp]
        total_tp_alerts += len(matches_tp)
        detected_anomalous_indices.update(matches_tp.index)

        # FP Alerts
        mask_fp = pd.Series([True] * len(test_normal), index=test_normal.index)
        for key, value in rule.items():
            if key in test_normal.columns:
                if regrouping_map and key in regrouping_map:
                    r_map = regrouping_map[key]
                    # FIX: The 'value' from a regrouped signature is already the target group.
                    sig_group = value
                    mask_fp &= (test_normal[key].map(r_map).fillna(-2) == sig_group)
                else:
                    mask_fp &= (test_normal[key] == value)
        total_fp_alerts += mask_fp.sum()

    # Precision based on total alerts
    precision = total_tp_alerts / (total_tp_alerts + total_fp_alerts) if (total_tp_alerts + total_fp_alerts) > 0 else 0.0

    # Recall based on unique instances detected
    TP_for_recall = len(detected_anomalous_indices)
    total_actual_anomalies = len(test_anomalous)
    recall = TP_for_recall / total_actual_anomalies if total_actual_anomalies > 0 else 0.0
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1_score


# --- NEW: Helper for regrouping ---
def apply_regrouping(mapped_df, regrouping_map_dict):
    """Applies a regrouping map to an already mapped (0-99) dataframe."""
    regrouped_df = mapped_df.copy()
    for column, r_map in regrouping_map_dict.items():
        if column in regrouped_df.columns:
            # Use .get(x, x) to keep original value if it's not in the map (handles non-numeric like labels)
            regrouped_df[column] = regrouped_df[column].apply(lambda x: r_map.get(x, x))
    return regrouped_df


# --- REVISED Main Evaluation Logic ---
def run_evaluation_for_condition(
    condition_value, train_df_base, test_df_base, args, num_processes_for_algo,
    base_mapped_train, base_mapped_test, regrouping_maps, signatures_from_train_100, signatures_from_test_100,
    original_whitelist # --- Pass whitelist ---
):
    start_time = time.time()
    print(f"\n{'='*30}\n[INFO] Running Evaluation for n_splits = {condition_value}\n{'='*30}")

    pregrouped_signature_count = len(signatures_from_train_100)

    current_regrouping_map = regrouping_maps[condition_value]

    # --- Regroup signatures to match condition_value ---
    # Regroup the fine-grained signatures (0-99) into conceptual groups (0 to condition_value-1)
    print(f"[INFO] Regrouping signatures from 100-split to {condition_value}-split level...")
    
    regrouped_train_signatures_set = set()
    for sig in signatures_from_train_100:
        regrouped_sig = {}
        for key, value in sig.items():
            if key in current_regrouping_map:
                try:
                    int_value = int(float(value))
                    regrouped_sig[key] = current_regrouping_map[key].get(int_value, int_value)
                except (ValueError, TypeError):
                    regrouped_sig[key] = value
            else:
                regrouped_sig[key] = value
        regrouped_train_signatures_set.add(frozenset(regrouped_sig.items()))
    
    regrouped_signatures_from_train_100 = [dict(s) for s in regrouped_train_signatures_set]
    regrouped_cluster_count = len(regrouped_signatures_from_train_100)
    print(f"[INFO] Regrouped {len(signatures_from_train_100)} train signatures into {regrouped_cluster_count} conceptual signatures for n_splits={condition_value}.")

    # --- Stages 1 & 2: Filter the REGROUPED signatures ---
    stage1_start = time.time()
    print(f"[1-2/9] Filtering {len(regrouped_signatures_from_train_100)} regrouped TRAIN signatures...")
    
    # Data remains at 100-split level (0-99), but we use regrouping_map for comparison
    train_normal_base = base_mapped_train[base_mapped_train['label'] == 0].drop(columns=['label'])
    
    signatures_from_train = [
        rule for rule in regrouped_signatures_from_train_100
        if calculate_support_for_itemset(rule, train_normal_base, regrouping_map=current_regrouping_map) < args.min_support
    ]
    print(f"-> Found {len(signatures_from_train)} signatures from TRAIN data.")
    stage1_2_time = time.time() - stage1_start

    # Stages 3, 4, 5...
    # ... (The rest of the logic needs to be adapted to this new paradigm)
    
    # --- If in Virtual Labeling mode, filter the second set of signatures ---
    test_normal_virtual_base = pd.DataFrame()  # Initialize to avoid NameError
    if 'original_label' in test_df_base.columns:
        stage4_start = time.time()
        print(f"[4/9] Filtering pre-generated TEST signatures using VIRTUAL labels (no regrouping)...")

        # Regroup test signatures
        regrouped_test_signatures_set = set()
        for sig in signatures_from_test_100:
            regrouped_sig = {}
            for key, value in sig.items():
                if key in current_regrouping_map:
                    try:
                        int_value = int(float(value))
                        regrouped_sig[key] = current_regrouping_map[key].get(int_value, int_value)
                    except (ValueError, TypeError):
                        regrouped_sig[key] = value
                else:
                    regrouped_sig[key] = value
            regrouped_test_signatures_set.add(frozenset(regrouped_sig.items()))
        
        regrouped_signatures_from_test_100 = [dict(s) for s in regrouped_test_signatures_set]
        print(f"[INFO] Regrouped {len(signatures_from_test_100)} test signatures into {len(regrouped_signatures_from_test_100)} conceptual signatures.")
        
        # We use the base_mapped_test data (100-split level) which has VIRTUAL labels
        test_normal_virtual_base = base_mapped_test[base_mapped_test['label'] == 0].drop(columns=['label'])
        
        signatures_from_test = [
            rule for rule in regrouped_signatures_from_test_100
            if calculate_support_for_itemset(rule, test_normal_virtual_base, regrouping_map=current_regrouping_map) < args.min_support
        ]
        print(f"-> Found {len(signatures_from_test)} signatures from TEST data (virtual labels).")
        stage4_time = time.time() - stage4_start
    else:
        signatures_from_test = []
        stage4_time = 0.0

    # --- NEW 6. Cross False Positive Check ---
    # Check all preliminary signatures against the virtual normal test data if it exists.
    stage6_start = time.time()
    if not test_normal_virtual_base.empty: # Use test_normal_virtual_base here
        '''
        print(f"[6/9] Cross-validating {len(signatures_from_train)} signatures against VIRTUAL NORMAL test data...")
        final_signatures = [
            rule for rule in signatures_from_train 
            if calculate_support_for_itemset(rule, test_normal_virtual_base, regrouping_map=current_regrouping_map) < args.min_support
        ]
        print(f"-> {len(final_signatures)} signatures remain after cross-FP check.")
        '''

        print(f"[6/9] Cross-validating {len(signatures_from_train)} signatures against VIRTUAL NORMAL test data using strict alert-based check...")
        
        # --- NEW: Stricter, alert-based FP check ---
        # Format signatures for the check function
        signatures_to_check = [{'id': i, 'name': f'Sig_{i}', 'rule_dict': rule} for i, rule in enumerate(signatures_from_train)]
        
        # Find all signatures that cause at least one FP alert
        fp_alerts = apply_signatures_to_dataset(test_normal_virtual_base, signatures_to_check)
        
        if not fp_alerts.empty:
            bad_signature_ids = set(fp_alerts['signature_id'].unique())
            # Keep only the signatures whose ID is NOT in the set of bad IDs
            final_signatures = [
                sig['rule_dict'] for sig in signatures_to_check if sig['id'] not in bad_signature_ids
            ]
        else:
            # If no FP alerts were found, all signatures are good
            final_signatures = signatures_from_train
            
        # --- OLD Support-based logic (now replaced) ---
        # final_signatures = [
        #     rule for rule in signatures_from_train 
        #     if calculate_support_for_itemset(rule, test_normal_virtual_base, regrouping_map=current_regrouping_map) < args.min_support
        # ]
        print(f"-> {len(final_signatures)} signatures remain after strict cross-FP check.")
    else:
        # If there's no virtual normal data (e.g., not VL mode), skip this check.
        print("[6/9] Skipping cross-FP check as no virtual normal data is available.")
        final_signatures = signatures_from_train
    stage6_time = time.time() - stage6_start


    # --- DISABLED 7. Remove Inactive Signatures ---
    # Inactive removal is skipped - keep all signatures
    stage7_start = time.time()
    print(f"[7/9] Skipping inactive signature removal - keeping all signatures.")
    # final_signatures remains unchanged
    stage7_time = 0.0

    # --- NEW: Whitelist Application ---
    # Regroup the original fine-grained whitelist to the current split level and add them back in.
    # This ensures high-performers are not lost during filtering.
    print(f"[INFO] Applying whitelist... Regrouping {len(original_whitelist)} whitelisted signatures for n_splits={condition_value}.")
    regrouped_whitelist_set = set()
    for whitelisted_sig_frozenset in original_whitelist:
        regrouped_sig = {}
        for key, value in dict(whitelisted_sig_frozenset).items():
            if key in current_regrouping_map:
                try:
                    int_value = int(float(value))
                    regrouped_sig[key] = current_regrouping_map[key].get(int_value, int_value)
                except (ValueError, TypeError):
                    regrouped_sig[key] = value
            else:
                regrouped_sig[key] = value
        regrouped_whitelist_set.add(frozenset(regrouped_sig.items()))

    # Combine the filtered set with the regrouped whitelist
    final_signatures_frozensets = {frozenset(s.items()) for s in final_signatures}
    combined_signatures_frozensets = final_signatures_frozensets.union(regrouped_whitelist_set)
    final_signatures = [dict(s) for s in combined_signatures_frozensets]
    print(f"-> Signature count after re-applying whitelist: {len(final_signatures)}")

    # --- Negative-aware filtering on normal data (global support, batched to save memory) ---
    # --- Negative-aware filtering on normal data (global support, batched to save memory) ---
    # Use base_mapped_train (100-split level) for negative filtering, regrouping_map handles comparison
    regrouped_normal = base_mapped_train[base_mapped_train['label'] == 0].drop(columns=['label','cluster','adjusted_cluster'], errors='ignore')
    if regrouped_normal.empty:
        print("[NegFilter] No normal data available; skipping normal-based filtering.")
    else:
        nbf_batch = args.normal_batch_size_filter
        if nbf_batch and len(regrouped_normal) > nbf_batch:
            normal_chunks = [regrouped_normal.iloc[i:i+nbf_batch] for i in range(0, len(regrouped_normal), nbf_batch)]
            print(f"[NegFilter] Splitting normal data into {len(normal_chunks)} batches of up to {nbf_batch} rows (global support aggregation).")
        else:
            normal_chunks = [regrouped_normal]

        total_rows_normal = sum(len(c) for c in normal_chunks)

        def rule_support_global(rule):
            total_matches = 0.0
            for chunk in normal_chunks:
                supp = calculate_support_for_itemset(rule, chunk, regrouping_map=current_regrouping_map)
                total_matches += supp * len(chunk)
            return (total_matches / total_rows_normal) if total_rows_normal > 0 else 0.0

        if args.negative_filtering:
            before_nf = len(final_signatures)
            final_signatures = [r for r in final_signatures if rule_support_global(r) <= args.negative_filter_threshold]
            print(f"[NegFilter] Filtered {before_nf - len(final_signatures)} rules; remaining {len(final_signatures)}.")

    # --- 8. Final Evaluation ---
    stage8_start = time.time()
    
    # --- Correctly prepare the final test set for evaluation ---
    if 'original_label' in test_df_base.columns:
        # If in VL mode, we need the test set with REAL labels.
        # The data should remain 0-99 mapped, as calculate_performance_revised will apply the regrouping map.
        base_mapped_test_with_real_labels = base_mapped_test.drop(columns=['label', 'original_label'], errors='ignore').copy()
        base_mapped_test_with_real_labels['label'] = test_df_base['original_label'].values
        # REMOVED incorrect pre-regrouping. The calculation function handles this.
        final_eval_test_df = base_mapped_test_with_real_labels
    else:
        # If not in VL mode, the test set is just the re-mapped test data
        final_eval_test_df = base_mapped_test # Use base_mapped_test directly

    eval_stage_num = 5 if not 'original_label' in test_df_base.columns else 8
    print(f"[{eval_stage_num}/9] Final evaluation of {len(final_signatures)} signatures against {len(final_eval_test_df)} test samples...")
    
    # --- MODIFIED: Calculate all performance metrics ---
    
    # 1. Total Performance (from the combined set of signatures)
    total_precision, total_recall, total_f1_score = calculate_performance_revised(final_signatures, final_eval_test_df, current_regrouping_map)

    # 2. Average Performance (from individual signature sets)
    precisions_to_avg = []
    recalls_to_avg = []

    if signatures_from_train:
        p_train, r_train, _ = calculate_performance_revised(signatures_from_train, final_eval_test_df, current_regrouping_map)
        precisions_to_avg.append(p_train)
        recalls_to_avg.append(r_train)
    
    if signatures_from_test: # This list is only populated in VL mode
        p_test, r_test, _ = calculate_performance_revised(signatures_from_test, final_eval_test_df, current_regrouping_map)
        precisions_to_avg.append(p_test)
        recalls_to_avg.append(r_test)
        
    avg_precision = sum(precisions_to_avg) / len(precisions_to_avg) if precisions_to_avg else 0.0
    avg_recall = sum(recalls_to_avg) / len(recalls_to_avg) if recalls_to_avg else 0.0

    stage8_time = time.time() - stage8_start
    
    # --- 9. Return Results ---
    total_time = time.time() - start_time
    print(f"-> Results: Total(P={total_precision:.4f}, R={total_recall:.4f}, F1={total_f1_score:.4f}), "
          f"Avg(P={avg_precision:.4f}, R={avg_recall:.4f}), Time={total_time:.2f}s")

    final_signature_count = len(final_signatures)

    # --- Memory Cleanup for this iteration ---
    del train_normal_base, test_normal_virtual_base # Clean up intermediate data
    gc.collect()

    return {
        'condition_count': condition_value, 
        'pregrouped_signature_count': pregrouped_signature_count,
        'regrouped_cluster_count': regrouped_cluster_count,
        'signature_count': final_signature_count, 
        'avg_precision': avg_precision, 
        'avg_recall': avg_recall,
        'total_precision': total_precision,
        'total_recall': total_recall, 
        'total_f1_score': total_f1_score, 
        'time_seconds': total_time,
        's1_map_train': stage1_2_time, # Combined stage 1 and 2
        's2_gen_sig_train': 0.0, # No new signature generation here
        's3_map_test': 0.0, # No new mapping here
        's4_gen_sig_test_vl': stage4_time,
        's5_combine_sig': 0.0, # No new combination here
        's6_fp_check': stage6_time,
        's7_inactive_check': stage7_time,
        's6_s7_filtering_time': stage6_time + stage7_time,
        's8_evaluate': stage8_time
    }

# --- Hardcoded optimal cluster counts for specific datasets/algorithms ---
hardcoded_k_combinations = {
    ('NSL-KDD', 'KMeans'): 150,
    ('NSL-KDD', 'GMM'): 150,
    ('Kitsune', 'KMeans'): 150,
    ('Kitsune', 'GMM'): 150,
    ('MiraiBotnet', 'KMeans'): 150,
    ('MiraiBotnet', 'GMM'): 150,
    ('CICIDS2017', 'KMeans'): 150,
    ('CICIDS2017', 'GMM'): 150,
    ('DARPA98', 'KMeans'): 150,
    ('DARPA98', 'GMM'): 150,
    ('CICIoT2023', 'KMeans'): 150,
    ('CICIoT2023', 'GMM'): 150,
}


def main():
    parser = argparse.ArgumentParser(description="Evaluate the impact of mapping condition counts on signature performance.")
    parser.add_argument('--file_type', type=str, required=True)
    parser.add_argument('--file_number', type=int, default=1)
    parser.add_argument('--association_method', type=str, default='RARM')
    parser.add_argument('--min_support', type=float, default=0.1)
    parser.add_argument('--min_confidence', type=float, default=0.8)
    parser.add_argument('--num_processes', type=int, default=None, help='Total CPU cores to use. Defaults to all available.')
    parser.add_argument('--condition_values', nargs='+', type=int, default=[2, 5, 10, 20, 40, 60, 80, 100])
    # --- Negative-aware filtering ---
    parser.add_argument('--negative_filtering', action='store_true', help="Enable negative-aware filtering using P(rule|normal) thresholds.")
    parser.add_argument('--negative_filter_threshold', type=float, default=0.05, help="Maximum allowed P(rule|normal) when negative-aware filtering is enabled.")
    parser.add_argument('--normal_batch_size_filter', type=int, default=20000, help="Batch size for negative-filtering support calc (memory-only; support is global).")
    # --- Dominant column masking ---
    parser.add_argument('--mask_dominant_cols', action='store_true', default=True, help="Mask near-constant columns (freq > dominant_freq_threshold) during rule mining.")
    parser.add_argument('--dominant_freq_threshold', type=float, default=0.99, help="Frequency ratio to treat a column as dominant (near-constant).")
    # --- Itemset limit ---
    parser.add_argument('--itemset_limit', type=int, default=10000000, help="Limit on itemsets for association mining to prevent explosion.")
    # --- Level limits ---
    parser.add_argument('--max_level_override', type=int, default=None, help="Optional override for max rule length (falls back to LEVEL_LIMITS_BY_FILE_TYPE).")
    
    # --- Arguments for rare column removal ---
    parser.add_argument('--min_support_ratio_for_rare', type=float, default=0.1,
                        help='(Optional) Minimum support ratio for a column to be kept during rare column removal.')
    parser.add_argument('--min_distinct', type=int, default=2,
                        help='(Optional) Minimum number of distinct values for a column to be kept.')

    # --- Arguments for flexible test set handling ---
    parser.add_argument('--test_file_path', type=str, default=None, 
                        help='(Optional) Path to an external test file. If provided, uses the full dataset for training.')
    ''' %already use virtual label%
    parser.add_argument('--use_virtual_labeling', action='store_true', 
                        help='(Optional) If set, generates a test set and replaces its labels with virtual ones from clustering.')
    '''
    parser.add_argument('--test_ratio', type=float, default=0.1, 
                        help='Proportion of the dataset to use as the test set for virtual labeling.')
    parser.add_argument('--clustering_algo', type=str, default='KMeans', choices=['KMeans', 'GMM'],
                        help='Clustering algorithm for virtual labeling.')
    parser.add_argument('--n_clusters', type=int, default=None,
                        help='Number of clusters for virtual labeling. Overridden by hardcoded_k_combinations if available.')
    
    # --- Argument for sampling CICIoT2023 ---
    parser.add_argument('--sample_10pct', action='store_true',
                        help='(Optional) If set, randomly samples 10%% of the CICIoT2023 dataset for faster evaluation.')
    
    args = parser.parse_args()

    # --- NEW: Determine n_clusters using hardcoded values or args ---
    # hardcoded_k_combinations is not defined in this file, assuming it's imported or defined elsewhere if needed.
    # For now, using a placeholder or direct args.n_clusters.
    n_clusters = args.n_clusters if args.n_clusters is not None else 2 # Fallback default
    args.n_clusters = n_clusters # Update args with the determined n_clusters

    # --- Store original value to check if user explicitly provided --num_processes ---
    user_provided_num_processes = args.num_processes is not None
    
    if args.num_processes is None:
        args.num_processes = os.cpu_count()
        
    # --- MOVED: Define num_processes_for_algo early ---
    num_processes_for_algo = args.num_processes
    
    # --- NEW: Limit BLAS threads only if user explicitly provided --num_processes ---
    # This prevents OpenBLAS/MKL from using all cores when num_processes is explicitly limited
    if user_provided_num_processes:
        os.environ['OMP_NUM_THREADS'] = str(args.num_processes)
        os.environ['MKL_NUM_THREADS'] = str(args.num_processes)
        os.environ['OPENBLAS_NUM_THREADS'] = str(args.num_processes)
        os.environ['NUMEXPR_NUM_THREADS'] = str(args.num_processes)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(args.num_processes)
        print(f"[INFO] BLAS thread limits set to {args.num_processes} to match --num_processes")
    else:
        print(f"[INFO] Using all available cores ({args.num_processes}). BLAS threads not limited.")

    print("--- Starting Mapping Condition Evaluation ---")
    
    # --- 1. Load and Prepare Base Data ---
    print("Loading and preparing base dataset...")
    # MODIFIED: Use the new function to load the clustered data
    file_path, _ = get_clustered_data_path(args.file_type, args.file_number)
    data = file_cut(args.file_type, file_path, 'all')

    # The loaded data already has a 'label' and 'cluster' column.
    # The original label logic is no longer needed as it was done in the previous script.
    # We will use 'cluster' for association and 'label' for evaluation.
    
    base_df = time_scalar_transfer(data, args.file_type)
    
    # --- NEW: Conditional Sampling for CICIoT2023 ---
    if args.file_type in ['CICIoT2023', 'CICIoT'] and args.sample_10pct:
        sample_frac = 0.1
        print(f"\n[INFO] SAMPLING: Randomly sampling {sample_frac * 100}% of the {args.file_type} dataset as requested.")
        original_size = len(base_df)
        # .sample() shuffles, .sort_index() restores original relative order for the sample.
        base_df = base_df.sample(frac=sample_frac).sort_index().reset_index(drop=True)  # for Tendency (standard deviation)
        #base_df = base_df.sample(frac=sample_frac, random_state=42).sort_index().reset_index(drop=True)    # for reproducibility
        print(f"-> Dataset size reduced from {original_size} to {len(base_df)} rows.")

    # --- REMOVED: Rare column removal logic is now performed AFTER mapping ---
    
    print("Base dataset prepared.")
    del data
    gc.collect()

    # --- 2. Create Train/Test Split ---
    train_df_base = None
    test_df_base = None

    if args.test_file_path:
        print(f"MODE: Using external test file '{args.test_file_path}'. Full source data will be used for training.")
        train_df_base = base_df.copy()
        try:
            test_df_base = pd.read_csv(args.test_file_path, low_memory=False)
            if 'label' not in test_df_base.columns:
                raise ValueError("External test file must contain a 'label' column.")
        except Exception as e:
            print(f"[ERROR] Failed to load or validate external test file: {e}")
            return
    
    else: # Default mode: Virtual Labeling with Caching
        print(f"MODE: Virtual Labeling with Caching. Test ratio: {args.test_ratio}, Algorithm: {args.clustering_algo}")
        
        # --- MODIFIED: Temporarily disabled tuning logic to use hardcoded k values ---
        k_key = (args.file_type, args.clustering_algo)
        hardcoded_k = hardcoded_k_combinations.get(k_key)

        if hardcoded_k is not None:
            n_clusters = hardcoded_k
            print(f"  -> Using hardcoded k={n_clusters} for {k_key[0]}/{k_key[1]}.")
        else:
            n_clusters = args.n_clusters if args.n_clusters is not None else 10 # Fallback default
            print(f"  -> No hardcoded 'k' found for {k_key[0]}/{k_key[1]}. Using fallback k={n_clusters}.")
        
        '''
        # --- MODIFIED: Hyperparameter tuning for n_clusters with min-value comparison ---
        k_key = (args.file_type, args.clustering_algo)
        hardcoded_k = hardcoded_k_combinations.get(k_key)

        # --- Caching logic for tuned parameters ---
        cache_dir_tuned = f"../Dataset_Paral/Virtual_Labels_for_Mapping/tuned_params/{args.file_type}"
        os.makedirs(cache_dir_tuned, exist_ok=True)
        tuned_params_cache_filename = f"tuned_params_{args.clustering_algo}.json"
        tuned_params_cache_path = os.path.join(cache_dir_tuned, tuned_params_cache_filename)

        best_params = {}
        if os.path.exists(tuned_params_cache_path):
            try:
                import json
                with open(tuned_params_cache_path, 'r') as f:
                    best_params = json.load(f)
                print(f"  -> Found cached tuned parameters: {best_params}")
            except Exception as e:
                print(f"  -> WARN: Could not load cached tuned parameters: {e}")
                best_params = {}

        # If cache doesn't exist or is empty, run the grid search
        if not best_params:
            print("  -> No cached parameters found. Running Grid Search for hyperparameter tuning...")
            # Prepare data for tuning (only done if needed)
            split_index_for_tuning = int(len(base_df) * (1 - args.test_ratio))
            test_df_for_tuning = base_df.iloc[split_index_for_tuning:]
            numeric_cols_for_tuning = test_df_for_tuning.select_dtypes(include=np.number).columns.tolist()
            if 'label' in numeric_cols_for_tuning:
                numeric_cols_for_tuning.remove('label')
            X_test_for_tuning = test_df_for_tuning[numeric_cols_for_tuning].copy()
            X_test_for_tuning.replace([np.inf, -np.inf], np.nan, inplace=True)
            X_test_for_tuning.fillna(0, inplace=True)

            if not X_test_for_tuning.empty:
                # Note: Grid_search_all returns a dictionary that includes the best parameters.
                # FIX: We need the full result dictionary to get the best params, not just the parameter dict.
                # The 'all_params' key contains the dictionary with the best found hyperparameters.
                tuned_results = Grid_search_all(X_test_for_tuning, args.clustering_algo, num_processes_for_algo=num_processes_for_algo)
                best_params = tuned_results.get('all_params', {}) # Use all_params which has the full dict
                print(f"  -> Grid Search complete. Best parameters found: {best_params}")
                # Save the tuned parameters to cache
                try:
                    import json
                    with open(tuned_params_cache_path, 'w') as f:
                        json.dump(best_params, f)
                    print(f"  -> Saved tuned parameters to cache: {tuned_params_cache_path}")
                except Exception as e:
                    print(f"  -> WARN: Could not save tuned parameters to cache: {e}")
            else:
                print("  -> WARN: Test set for tuning is empty. Skipping Grid Search.")

        if args.clustering_algo == 'GMM':
            tuned_k = best_params.get('n_components')
        else: # KMeans and others
            tuned_k = best_params.get('n_clusters')

        if tuned_k is not None:
            if hardcoded_k is not None:
                n_clusters = min(hardcoded_k, tuned_k)
                print(f"  -> Comparing hardcoded k={hardcoded_k} and tuned k={tuned_k}. Using min value: {n_clusters}")
            else:
                n_clusters = tuned_k
                print(f"  -> Using tuned k={tuned_k} as no hardcoded value was found.")
        elif hardcoded_k is not None:
            n_clusters = hardcoded_k
            print(f"  -> Tuning failed or did not provide 'k'. Using hardcoded k={hardcoded_k}.")
        else:
            n_clusters = args.n_clusters if args.n_clusters is not None else 10 # Fallback default
            print(f"  -> Tuning failed and no hardcoded 'k' found. Using fallback default k={n_clusters}.")
        '''

        # --- Caching Logic for the generated dataset ---
        cache_dir = f"../Dataset_Paral/Virtual_Labels_for_Mapping/{args.file_type}"
        os.makedirs(cache_dir, exist_ok=True)
        cache_filename = f"{args.file_type}_test_vl_{args.clustering_algo}_{n_clusters}.csv"
        cache_path = os.path.join(cache_dir, cache_filename)

        split_index = int(len(base_df) * (1 - args.test_ratio))
        train_df_base = base_df.iloc[:split_index]

        if os.path.exists(cache_path):
            print(f"  -> Found cached virtually labeled test set: {cache_path}")
            test_df_base = pd.read_csv(cache_path, low_memory=False)
        else:
            print(f"  -> No cache found. Generating virtually labeled test set...")
            test_df_to_label = base_df.iloc[split_index:]

            if test_df_to_label.empty:
                print("[ERROR] Test set is empty after splitting. Aborting.")
                return

            print(f"  -> Generating virtual labels for {len(test_df_to_label)} test samples...")
            
            try:
                numeric_cols = test_df_to_label.select_dtypes(include=np.number).columns.tolist()
                if 'label' in numeric_cols:
                    numeric_cols.remove('label')
                
                X_test = test_df_to_label[numeric_cols].copy()
                X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
                X_test.fillna(0, inplace=True)

                if args.clustering_algo == 'KMeans':
                    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                elif args.clustering_algo == 'GMM':
                    model = GaussianMixture(n_components=n_clusters, random_state=42)
                
                virtual_labels_raw = model.fit_predict(X_test)

                majority_cluster = pd.Series(virtual_labels_raw).mode()[0]
                virtual_labels_binary = np.where(virtual_labels_raw == majority_cluster, 0, 1)

                test_df_base = test_df_to_label.copy()
                test_df_base['original_label'] = test_df_base['label']
                test_df_base['label'] = virtual_labels_binary
                
                print(f"  -> Saving generated test set to cache: {cache_path}")
                test_df_base.to_csv(cache_path, index=False)
                print("  -> Caching complete.")

            except Exception as e:
                print(f"[ERROR] An error occurred during virtual labeling: {e}")
                return
            
    if train_df_base.empty or test_df_base.empty:
        print("[ERROR] Training or testing dataframe is empty. Cannot proceed.")
        return

    # --- 3. Create Universal Base Mapping by combining train and test data first ---
    BASE_N_SPLITS = 100
    print(f"\n--- Creating Universal Base Map using combined train and test data (n_splits={BASE_N_SPLITS}) ---")
    base_map_start_time = time.time()
    
    # Store the split index to separate them later
    train_end_index = len(train_df_base)
    
    # Temporarily combine train and test dataframes for consistent mapping
    # Note: We keep original_label for test data if it exists
    combined_df = pd.concat([train_df_base, test_df_base], ignore_index=True)
    
    # Drop labels before mapping
    combined_labels = combined_df['label'].values
    combined_features = combined_df.drop(columns=['label', 'original_label'], errors='ignore')

    # Map the entire combined dataset at once
    print(f"Mapping combined data ({len(combined_df)} rows) with {BASE_N_SPLITS} splits...")
    # This function creates interval objects, not integer groups yet.
    mapped_combined_features_with_intervals, _, base_category_mapping, base_data_list = choose_heterogeneous_method(
        combined_features, args.file_type, 'Interval_inverse', 'N', n_splits_override=BASE_N_SPLITS
    )

    # --- FIX: Add the missing step to map intervals to integer group IDs ---
    print("-> Step 1/2 Complete: Intervals created. Now mapping intervals to integer group IDs (0-99)...")
    mapped_combined_features, _ = map_intervals_to_groups(mapped_combined_features_with_intervals, base_category_mapping, base_data_list)
    print("-> Step 2/2 Complete: Integer group mapping finished.")
    
    # Re-attach the labels to the mapped dataframe
    mapped_combined_df = mapped_combined_features.copy()
    mapped_combined_df['label'] = combined_labels
    # Re-attach cluster and adjusted_cluster if present in source data
    if 'cluster' in combined_df.columns:
        mapped_combined_df['cluster'] = combined_df['cluster'].values
    if 'adjusted_cluster' in combined_df.columns:
        mapped_combined_df['adjusted_cluster'] = combined_df['adjusted_cluster'].values
    
    # Split back into train and test sets
    base_mapped_train = mapped_combined_df.iloc[:train_end_index].copy()
    base_mapped_test = mapped_combined_df.iloc[train_end_index:].copy()
    
    # Restore the 'original_label' to the test set if it existed
    if 'original_label' in test_df_base.columns:
        base_mapped_test['original_label'] = test_df_base['original_label'].values

    print("Consistent mapping for TRAIN and TEST data is complete.")
    print(f"-> Universal Base Map created and applied in {time.time() - base_map_start_time:.2f}s.")
    
    # --- Conditionally remove rare columns AFTER mapping ---
    if args.file_type in ['CICIDS2017', 'CICIDS', 'NSL-KDD', 'NSL_KDD', 'CICIoT2023', 'CICIoT', 'netML']:    # Kitsune is excluded because its normal data is too uniform
        print(f"\n[INFO] Applying rare column removal for {args.file_type} AFTER mapping...")
        
        # Add safety checks for None from user's modification
        if args.min_support_ratio_for_rare is None:
            args.min_support_ratio_for_rare = 0.1
        if args.min_distinct is None:
            args.min_distinct = 2

        # --- Special handling for CICIoT2023 to protect the label column ---
        if args.file_type in ['CICIoT2023', 'CICIoT']:
            print(f"   -> Applying special 'label' protection for {args.file_type}.")
            # 1. Process training data
            original_train_cols = base_mapped_train.shape[1]
            train_cols_before = set(base_mapped_train.columns)
            
            # Separate labels before removal (protect cluster columns)
            train_labels = base_mapped_train['label']
            cluster_cols = [c for c in ['cluster', 'adjusted_cluster'] if c in base_mapped_train.columns]
            train_features = base_mapped_train.drop(columns=['label'] + cluster_cols, errors='ignore')

            train_features_filtered = remove_rare_columns(
                train_features, 
                min_support_ratio=args.min_support_ratio_for_rare,
                file_type=args.file_type,
                min_distinct_frequent_values=args.min_distinct
            )
            
            # Re-attach labels
            base_mapped_train = pd.concat([train_features_filtered, train_labels], axis=1)
            # Re-attach cluster columns
            for c in cluster_cols:
                base_mapped_train[c] = combined_df.loc[base_mapped_train.index, c].values
            
            train_cols_after = set(base_mapped_train.columns)
            # Correctly calculate removed columns, excluding 'label' from consideration
            removed_cols = (train_cols_before - {'label'}) - (train_cols_after - {'label'})

            print(f"-> TRAIN columns reduced from {original_train_cols} to {base_mapped_train.shape[1]}.")
            if removed_cols:
                print(f"   -> Removed {len(removed_cols)} columns: {list(removed_cols)}")

            # 2. Process test data, ensuring column consistency
            original_test_cols = base_mapped_test.shape[1]
            cols_to_drop_from_test = [col for col in removed_cols if col in base_mapped_test.columns and col not in ['cluster','adjusted_cluster']]
            base_mapped_test = base_mapped_test.drop(columns=cols_to_drop_from_test)
            
            print(f"-> TEST columns reduced from {original_test_cols} to {base_mapped_test.shape[1]} to match train set.")

        else:
            # --- Original logic for other datasets ---
            # 1. Process training data and get columns to remove (protect label & cluster cols)
            original_train_cols = base_mapped_train.shape[1]
            train_cols_before = set(base_mapped_train.columns)
            
            cluster_cols = [c for c in ['cluster', 'adjusted_cluster'] if c in base_mapped_train.columns]
            train_label_series = base_mapped_train['label'] if 'label' in base_mapped_train.columns else None
            train_features = base_mapped_train.drop(columns=['label'] + cluster_cols, errors='ignore')
            
            train_features = remove_rare_columns(
                train_features, 
                min_support_ratio=args.min_support_ratio_for_rare,
                file_type=args.file_type,
                min_distinct_frequent_values=args.min_distinct
            )
            
            # Re-attach label and cluster columns
            if train_label_series is not None:
                base_mapped_train = pd.concat([train_features, train_label_series], axis=1)
            else:
                base_mapped_train = train_features
            for c in cluster_cols:
                if c not in base_mapped_train.columns and c in combined_df.columns:
                    base_mapped_train[c] = combined_df.loc[base_mapped_train.index, c].values
            
            train_cols_after = set(base_mapped_train.columns)
            # Exclude label from removed-cols calculation
            removed_cols = (train_cols_before - {'label'}) - (train_cols_after - {'label'})
            
            print(f"-> TRAIN columns reduced from {original_train_cols} to {base_mapped_train.shape[1]}.")
            if removed_cols:
                print(f"   -> Removed {len(removed_cols)} columns: {list(removed_cols)}")
            
            # 2. Process test data, ensuring column consistency (preserve label/cluster)
            original_test_cols = base_mapped_test.shape[1]
            cols_to_drop_from_test = [col for col in removed_cols if col in base_mapped_test.columns and col not in ['label','cluster','adjusted_cluster']]
            base_mapped_test = base_mapped_test.drop(columns=cols_to_drop_from_test)
            
            print(f"-> TEST columns reduced from {original_test_cols} to {base_mapped_test.shape[1]} to match train set.")

    # --- 4. Generate Signatures ONCE using the base map ---
    print("\n--- Generating Signatures ONCE from Base Mapped Data (n_splits=100) ---")
    sig_gen_start_time = time.time()
    
    # Generate from train data using 'cluster' column
    # MODIFIED: Split by 'cluster' column
    print("[INFO] Splitting training data based on 'cluster' column for signature generation.")
    train_normal_base = base_mapped_train[base_mapped_train['cluster'] == 0].drop(columns=['label', 'cluster'])
    train_anomalous_base = base_mapped_train[base_mapped_train['cluster'] == 1].drop(columns=['label', 'cluster'])
    max_level = args.max_level_override if args.max_level_override is not None else LEVEL_LIMITS_BY_FILE_TYPE.get(args.file_type, LEVEL_LIMITS_BY_FILE_TYPE['default'])
    
    rules_from_train_100, _ = association_module(
        train_anomalous_base, args.association_method, args.min_support, args.min_confidence, 
        'confidence', num_processes=num_processes_for_algo,
        file_type_for_limit=args.file_type, max_level_limit=max_level,
        itemset_limit=args.itemset_limit
    )
    # Note: We do NOT filter them here. We pass the raw rules to the evaluation loop.
    signatures_from_train_100 = rules_from_train_100
    print(f"-> Generated {len(signatures_from_train_100)} raw signatures from TRAIN data.")

    # Generate from test data (if in VL mode)
    signatures_from_test_100 = []
    if 'original_label' in test_df_base.columns:
        test_anomalous_virtual_base = base_mapped_test[base_mapped_test['label'] == 1].drop(columns=['label'])
        if not test_anomalous_virtual_base.empty:
            rules_from_test_100, _ = association_module(
                test_anomalous_virtual_base, args.association_method, args.min_support, args.min_confidence,
                'confidence', num_processes=num_processes_for_algo,
                file_type_for_limit=args.file_type, max_level_limit=max_level
            )
            signatures_from_test_100 = rules_from_test_100
            print(f"-> Generated {len(signatures_from_test_100)} raw signatures from TEST data (virtual labels).")
    
    print(f"-> Signature generation finished in {time.time() - sig_gen_start_time:.2f}s.")

    # --- NEW: 4.5 Generate Whitelist from base signatures ---
    print("\n--- Generating Whitelist from n_splits=100 signatures ---")
    # Prepare the test set with real labels for accurate performance assessment
    whitelist_eval_df = base_mapped_test.copy()
    if 'original_label' in whitelist_eval_df.columns:
        whitelist_eval_df['label'] = whitelist_eval_df['original_label']

    # We evaluate the 100-split signatures against the 100-split mapped data, so no regrouping_map is needed here.
    whitelist_signatures = create_whitelist(
        signatures=signatures_from_train_100,
        test_df=whitelist_eval_df,
        recall_threshold=0.4
        # precision_threshold=0.5 # -- Disabled as per user request
    )
    print(f"-> Created a whitelist with {len(whitelist_signatures)} high-performance signatures.")

    # --- 5. Create Regrouping Maps ---
    print("\n--- Creating Regrouping Maps for all condition values ---")
    regrouping_maps = {}
    
    # --- FIX: Access the DataFrame within the dictionary ---
    interval_mapping_df = base_category_mapping.get('interval')
    if interval_mapping_df is None or not isinstance(interval_mapping_df, pd.DataFrame):
        print("[ERROR] Could not find the 'interval' mapping DataFrame in 'base_category_mapping'. Aborting.")
        # Handle the error appropriately, maybe return or raise an exception
        return 
        
    numerical_cols_in_map = [col for col in interval_mapping_df.columns if col != 'Feature']
    
    for condition_value in args.condition_values:
        regrouping_maps[condition_value] = {}
        # For each numerical column that was mapped...
        for col in numerical_cols_in_map:
            # Create a map from 0-99 to the new smaller group number
            # e.g., for condition_value=2, maps 0-49 to 0, 50-99 to 1
            regrouping_maps[condition_value][col] = {i: i // (BASE_N_SPLITS // condition_value) for i in range(BASE_N_SPLITS)}
    print("-> Regrouping maps created.")

    # The evaluation for each condition will run sequentially.
    # num_processes_for_algo = args.num_processes

    print("\n" + "="*50)
    print("Sequential Evaluation Configuration:")
    print(f"  - Total Available Cores: {args.num_processes}")
    print(f"  - Number of Condition Values to Test: {len(args.condition_values)}")
    print(f"  - Cores Allocated per Association Rule Task: {num_processes_for_algo}")
    print("="*50 + "\n")

    tasks = sorted(args.condition_values)
    all_results = []
    
    for condition_value in tasks:
        result = run_evaluation_for_condition(
            condition_value, train_df_base.copy(), test_df_base.copy(), args, num_processes_for_algo,
            base_mapped_train.copy(), base_mapped_test.copy(), regrouping_maps,
            signatures_from_train_100, signatures_from_test_100,
            whitelist_signatures # --- Pass whitelist ---
        )
        all_results.append(result)

    print("\n\n--- Final Evaluation Summary ---")
    
    results_df = pd.DataFrame(all_results)
    
    # --- MODIFIED: Add file_type and other params back into the dataframe for saving ---
    results_df['file_type'] = args.file_type
    results_df['association_method'] = args.association_method
    results_df['min_support'] = args.min_support
    results_df['min_confidence'] = args.min_confidence
    
    # --- MODIFIED: Add timing summary printout here ---
    print("\n--- Timing Summary ---")
    for _, row in results_df.iterrows():
        print(f"n_splits={int(row['condition_count'])}: Total={row['time_seconds']:.2f}s "
              f"(S6+S7 Filter={row['s6_s7_filtering_time']:.2f}s, "
              f"S1={row['s1_map_train']:.2f}s, S4={row['s4_gen_sig_test_vl']:.2f}s, "
              f"S6={row['s6_fp_check']:.2f}s, S7={row['s7_inactive_check']:.2f}s, S8={row['s8_evaluate']:.2f}s)")

    print("\n--- Performance Summary ---")
    
    # MODIFIED: Explicitly define columns for display to ensure all metrics are shown
    display_column_order = [
        'file_type', 'association_method', 'min_support', 'min_confidence',
        'condition_count', 'pregrouped_signature_count', 'regrouped_cluster_count', 'signature_count', 
        'total_precision', 'total_recall', 'total_f1_score',
        'avg_precision', 'avg_recall',
        'time_seconds'
    ]
    # Filter to only include columns that actually exist in the dataframe
    display_cols = [col for col in display_column_order if col in results_df.columns]
    print(results_df[display_cols].to_string())
    
    # Create the directory for the main results CSV if it doesn't exist
    output_dir = f"../Dataset_Paral/Evaluate_Mapping/{args.file_type}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"{args.file_type}_{args.association_method}_s{args.min_support}_c{args.min_confidence}_mapping_evaluation_{timestamp}.csv"
    output_csv_path = os.path.join(output_dir, output_filename)
    
    # Reorder columns before saving for a clean CSV layout
    final_column_order = [
        'file_type', 'association_method', 'min_support', 'min_confidence',
        'condition_count', 'pregrouped_signature_count', 'regrouped_cluster_count', 'signature_count', 
        'total_precision', 'total_recall', 'total_f1_score',
        'avg_precision', 'avg_recall',
        'time_seconds', 's6_s7_filtering_time', 's1_map_train', 's2_gen_sig_train', 's3_map_test', 
        's4_gen_sig_test_vl', 's5_combine_sig', 's6_fp_check', 's7_inactive_check', 's8_evaluate'
    ]
    results_df = results_df[[col for col in final_column_order if col in results_df.columns]]
    results_df.to_csv(output_csv_path, index=False)
    print(f"\nSummary results saved to: {output_csv_path}")

    # Save detailed time log
    time_log_dir = f"../Dataset_Paral/Evaluate_Mapping/Time_Logs"
    os.makedirs(time_log_dir, exist_ok=True)
    time_log_df = results_df[['file_type', 'association_method', 'min_support', 'min_confidence', 'condition_count', 'time_seconds', 's6_s7_filtering_time', 's1_map_train', 's2_gen_sig_train', 's3_map_test', 's4_gen_sig_test_vl', 's5_combine_sig', 's6_fp_check', 's7_inactive_check', 's8_evaluate']]
    
    time_log_filename = f"time_log_mapping_conditions_{args.file_type}_{len(args.condition_values)}_{args.association_method}_ms{args.min_support}_mc{args.min_confidence}.csv"
    time_log_path = os.path.join(time_log_dir, time_log_filename)
    time_log_df.to_csv(time_log_path, index=False)
    print(f"\nDetailed timing log saved to: {time_log_path}")


if __name__ == "__main__":
    main()
