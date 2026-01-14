#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Seed Stability Experiment for Clustering Algorithms
- Fixed CNI threshold
- Multiple seeds
- Mean / Std per algorithm
"""

import os
import gc
import argparse
import numpy as np
import pandas as pd
import logging
import random
import sys
import shutil
import importlib
from collections import defaultdict
from sklearn.metrics import jaccard_score

from utils.generate_data_hash import generate_stable_data_hash
from utils.best_clustering_sampling import apply_memory_optimization_sampling
from utils.best_clustering_sampling import run_chunked_virtual_labeling  # parity with main pipeline
from Clustering_Method.clustering_nomal_identify import clustering_nomal_identify
from Modules.Jaccard_Elbow_Method import test_all_algorithms_with_jaccard_elbow
from utils.max_score_utils import comprehensive_optimization_max_score
import utils.max_score_utils as max_score_utils
from Tuning_hyperparameter.jaccard_run_single_clustering import run_single_clustering
from Dataset_Choose_Rule.choose_amount_dataset import file_path_line_nonnumber, file_cut
from utils.apply_labeling import apply_labeling_logic
from utils.time_transfer import time_scalar_transfer
from Modules.Heterogeneous_module import choose_heterogeneous_method
from Modules.PCA import pca_func
from Heterogeneous_Method.Feature_Encoding import Heterogeneous_Feature_named_featrues
from Dataset_Choose_Rule import save_jaccard_elbow


# CONFIG
SEED_LIST = [0, 1, 2, 3, 4]
FIXED_CNI_THRESHOLD = 0.4

CLUSTERING_ALGORITHMS = [
    'Kmeans', 'CLARA', 'GMM', 'SGMM',
    'Gmeans', 'Xmeans', 'DBSCAN',
    'MShift', 'FCM', 'CK', 'NeuralGas'
]

# logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# max_score_utils reload
sys.modules['utils.max_score_utils'] = max_score_utils
importlib.reload(max_score_utils)


# Utils
def set_global_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)


def _determine_sample_size(file_type, override=None):
    # mirror utils.best_clustering_sampling sampling_configs
    sampling_configs = {
        'CICIDS2017': 30000,
        'CICIDS': 30000,
        'CICIoT2023': 300000,
        'CICIoT': 300000,
        'IoTID20': 30000,
        'IoTID': 30000,
        'Kitsune': 30000,
        'NSL-KDD': 30000,
        'NSL_KDD': 30000,
        'netML': None,
        'MiraiBotnet': None,
        'DARPA98': None,
        'DARPA': None,
        'CICModbus23': None,
        'CICModbus': None,
    }
    if override is not None:
        return override
    return sampling_configs.get(file_type, None)


def _stratified_sample_with_seed(X, y, sample_size, seed):
    """Stratified sampling with explicit seed; returns (X_sampled, y_sampled)."""
    total = len(X)
    if sample_size is None or sample_size <= 0 or total <= sample_size:
        return X, y
    rng = np.random.default_rng(seed)
    unique_labels, counts = np.unique(y, return_counts=True)
    sampled_indices = []
    for label, count in zip(unique_labels, counts):
        label_indices = np.where(y == label)[0]
        label_quota = max(1, int(sample_size * (count / total)))
        label_quota = min(label_quota, len(label_indices))
        chosen = rng.choice(label_indices, size=label_quota, replace=False)
        sampled_indices.extend(chosen.tolist())
    # If we overshoot/undershoot due to rounding, trim/augment deterministically
    sampled_indices = np.array(sampled_indices)
    if len(sampled_indices) > sample_size:
        sampled_indices = sampled_indices[:sample_size]
    elif len(sampled_indices) < sample_size:
        remaining = np.setdiff1d(np.arange(total), sampled_indices, assume_unique=False)
        if len(remaining) > 0:
            extra = rng.choice(remaining, size=min(sample_size - len(sampled_indices), len(remaining)), replace=False)
            sampled_indices = np.concatenate([sampled_indices, extra])
    return X[sampled_indices], y[sampled_indices]


# Main
def prepare_data(args, logger):
    """Load and preprocess data identical to best_clustering_selector_parallel_previousPU_refine.py (scaling_label_encoding path)."""
    file_path, _ = file_path_line_nonnumber(args.file_type, args.file_number)
    data = file_cut(args.file_type, file_path, 'all')
    data.columns = data.columns.str.strip()

    data = apply_labeling_logic(data, args.file_type)
    data = time_scalar_transfer(data, args.file_type)

    # --- Feature selection (same as main pipeline) ---
    feature_dict = Heterogeneous_Feature_named_featrues(args.file_type)
    categorical_features = feature_dict.get('categorical_features', [])
    time_features = feature_dict.get('time_features', [])
    packet_length_features = feature_dict.get('packet_length_features', [])
    count_features = feature_dict.get('count_features', [])
    binary_features = feature_dict.get('binary_features', [])
    features_for_clustering = categorical_features + time_features + packet_length_features + count_features + binary_features

    existing_features = [f for f in features_for_clustering if f in data.columns]
    missing_features = set(features_for_clustering) - set(existing_features)
    if missing_features:
        logger.warning(f"[FeatSelect] Missing features from Feature_Encoding: {missing_features}")

    try:
        data_for_processing = data[existing_features + ['label']].copy()
    except KeyError as e:
        logger.critical(f"[FeatSelect] Required column missing after selection: {e}")
        raise

    # --- Cleaning numeric features (match main pipeline) ---
    numerical_features = time_features + packet_length_features + count_features
    columns_to_clean = [col for col in numerical_features if col in data_for_processing.columns]

    if columns_to_clean:
        # 1) Coerce to numeric
        for col in columns_to_clean:
            if not pd.api.types.is_numeric_dtype(data_for_processing[col]):
                data_for_processing[col] = pd.to_numeric(data_for_processing[col], errors='coerce')
        # 2) Replace inf/-inf with finite max (or 0)
        for col in columns_to_clean:
            col_vals = data_for_processing[col]
            if not np.isfinite(col_vals.values).all():
                if np.isinf(col_vals.values).any():
                    finite_max = col_vals[np.isfinite(col_vals)].max()
                    replacement_val = 0 if pd.isna(finite_max) else finite_max
                    data_for_processing[col].replace([np.inf, -np.inf], replacement_val, inplace=True)
                    logger.warning(f"[Clean] Replaced inf in '{col}' with {replacement_val}")
        # 3) Drop rows with any remaining NaN in numeric columns
        rows_with_nan = data_for_processing[columns_to_clean].isnull().any(axis=1)
        if rows_with_nan.any():
            num_drop = rows_with_nan.sum()
            data_for_processing = data_for_processing.loc[~rows_with_nan].copy()
            logger.warning(f"[Clean] Dropped {num_drop} rows with NaN in numeric columns.")
        if data_for_processing.empty:
            raise RuntimeError("All rows removed during cleaning; cannot proceed.")

    # Split features/labels after cleaning
    features = data_for_processing.drop(columns=['label'])
    labels = data_for_processing['label'].to_numpy()

    X_processed, _, _, _ = choose_heterogeneous_method(features, args.file_type, 'scaling_label_encoding', 'N')
    if args.use_float32:
        X_processed = X_processed.astype(np.float32)

    # PCA step (mirrors main pipeline)
    pca_want = 'N' if args.file_type in ['DARPA98', 'DARPA', 'CICModbus23', 'CICModbus'] else 'Y'
    if pca_want == 'Y':
        data_for_clustering = pca_func(X_processed)
    else:
        data_for_clustering = X_processed.to_numpy() if hasattr(X_processed, 'to_numpy') else X_processed

    original_labels = labels
    known_normal_indices = np.where(original_labels == 0)[0]
    known_normal_samples_pca = data_for_clustering[known_normal_indices]

    return data_for_clustering, original_labels, known_normal_samples_pca, known_normal_indices


# Patch Jaccard progress paths to isolate from main pipeline caches
def _patch_jaccard_progress_paths():
    """
    Redirect all Jaccard progress files (including max_score utils) to a seed-stability-only dir.
    """
    base_dir = os.path.join("../Dataset_ex", "seed_stability_progress")
    os.makedirs(base_dir, exist_ok=True)
    # Ensure children (spawned processes) also use this directory
    os.environ["JACCARD_PROGRESS_DIR"] = base_dir

    def _progress_path(algo, data_hash):
        return os.path.join(base_dir, f"jaccard_elbow_{data_hash}_{algo}_progress.csv")

    def _grid_progress_path(algo, data_hash):
        return os.path.join(base_dir, f"jaccard_elbow_{data_hash}_{algo}_Grid_progress.csv")

    def _jej_path(algo, data_hash):
        return _progress_path(algo, data_hash)

    def _load_progress(algo, data_hash):
        progress_file = _progress_path(algo, data_hash)
        completed_values = set()
        existing_scores = {}
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r', newline='', encoding='utf-8') as f:
                    lines = f.readlines()
                for line in lines[1:]:
                    row = line.strip().split(',')
                    if len(row) >= 3:
                        try:
                            param_value = float(row[0])
                            completed_values.add(param_value)
                            surrogate_score_val = float(row[1])
                            jaccard_score_val = float(row[2])
                            existing_scores[param_value] = {'surrogate_score': surrogate_score_val, 'jaccard_score': jaccard_score_val}
                        except ValueError:
                            continue
            except Exception as e:
                print(f"[Jaccard Elbow Progress] Error loading {progress_file}: {e}")
        return completed_values, existing_scores

    # Monkey-patch the module functions
    save_jaccard_elbow.get_progress_file_path = _progress_path
    save_jaccard_elbow.get_grid_search_progress_file_path = _grid_progress_path
    save_jaccard_elbow.get_jaccard_elbow_progress_file_path = _jej_path
    save_jaccard_elbow.load_jaccard_elbow_progress = _load_progress

    # Reload max_score_utils so it re-imports patched functions
    importlib.reload(max_score_utils)
    # Re-patch max_score_utils references in case they were cached
    max_score_utils.load_jaccard_elbow_progress = _load_progress
    max_score_utils.get_grid_search_progress_file_path = _grid_progress_path
    max_score_utils.get_jaccard_elbow_progress_file_path = _jej_path

    logger.info(f"[PatchCheck] progress_dir = {base_dir}")
    logger.info(f"[PatchCheck] progress_path example = {_progress_path('NeuralGas', 'dummyhash')}")


def save_progress(progress_path, rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(progress_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_type', type=str, required=True)
    parser.add_argument('--file_number', type=int, default=1)
    parser.add_argument('--max_score', action='store_true')
    parser.add_argument('--disable_sampling', action='store_true')
    parser.add_argument('--use_float32', action='store_true')
    parser.add_argument('--num_processes', type=int, default=os.cpu_count())
    parser.add_argument('--known_normal_fraction', type=float, default=0.9, help="Fraction of normal samples for consistency check.")
    parser.add_argument('--reset', action='store_true', help="Remove previous seed stability artifacts for this file_type and exit.")
    parser.add_argument('--sample_size_override', type=int, default=20000, help="Optional stratified sample size override (applied if sampling is enabled).")
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    dataset_ex_root = "../Dataset_ex"
    progress_dir = os.path.join(dataset_ex_root, "seed_stability", args.file_type)
    os.makedirs(progress_dir, exist_ok=True)
    progress_path = os.path.join(progress_dir, f"progress_seed_stability_{args.file_type}_{args.file_number}.csv")

    # --- reset option: remove prior artifacts for this file_type ---
    if args.reset:
        if os.path.isdir(progress_dir):
            logger.info(f"[RESET] Removing seed stability artifacts at {progress_dir}")
            shutil.rmtree(progress_dir, ignore_errors=True)
        else:
            logger.info(f"[RESET] No seed stability artifacts found at {progress_dir}")
        # Also remove seed_stability_progress directory (Jaccard progress cache)
        progress_cache_dir = os.path.join(dataset_ex_root, "seed_stability_progress")
        if os.path.isdir(progress_cache_dir):
            logger.info(f"[RESET] Removing seed stability progress cache at {progress_cache_dir}")
            shutil.rmtree(progress_cache_dir, ignore_errors=True)
        sys.exit(0)

    # Isolate progress files from main pipeline
    _patch_jaccard_progress_paths()

    data_for_clustering, original_labels, known_normal_samples_pca, known_normal_indices = prepare_data(args, logger)

    results = defaultdict(list)
    progress_rows = []

    try:
        for seed in SEED_LIST:
            logger.info(f"\n===== Seed {seed} =====")
            set_global_seed(seed)

            # Sampling (memory optimization) with per-seed randomness
            if args.disable_sampling:
                X_sel = data_for_clustering
                y_sel = original_labels
                kn_idx = known_normal_indices
            else:
                desired_sample_size = _determine_sample_size(args.file_type, args.sample_size_override)
                X_sel, y_sel = _stratified_sample_with_seed(
                    data_for_clustering, original_labels, desired_sample_size, seed
                )
                kn_idx = np.where(y_sel == 0)[0]
                if desired_sample_size is not None:
                    logger.info(f"[Sampling] file_type={args.file_type}, sample_size={len(X_sel)} (seed={seed})")

            data_hash = generate_stable_data_hash(
                args.file_type, args.file_number, X_sel.shape
            )

            # Algorithm loop
            for algo in CLUSTERING_ALGORITHMS:
                logger.info(f"[Seed {seed}] {algo}")

                try:
                    if args.max_score:
                        result = comprehensive_optimization_max_score(
                            algo, X_sel, y_sel,
                            args.file_type, args.file_number,
                            known_normal_samples_pca,
                            num_processes_for_algo=args.num_processes,
                            known_normal_idx=kn_idx
                        )
                        raw_labels = result.get('best_labels')
                        best_params = result.get('best_params')
                        # Fallback: if cached result lacks labels, recompute once with best_params
                        if raw_labels is None and best_params:
                            try:
                                raw_labels = run_single_clustering(
                                    algo,
                                    X_sel,
                                    best_params,
                                    aligned_original_labels=y_sel,
                                    global_known_normal_samples_pca=known_normal_samples_pca,
                                    num_processes_for_algo=args.num_processes,
                                    file_type=args.file_type
                                )
                                logger.warning(f"[{algo}] Recomputed raw_labels due to missing cache entry.")
                            except Exception as e:
                                logger.error(f"[{algo}] Fallback clustering failed: {e}")
                                raw_labels = None
                    else:
                        result = test_all_algorithms_with_jaccard_elbow(
                            X_sel, y_sel, [algo],
                            args.file_type, args.file_number,
                            known_normal_samples_pca,
                            num_processes_for_algo=args.num_processes,
                            known_normal_idx=kn_idx
                        )[algo]
                        raw_labels = result.get('best_labels')
                        best_params = result.get('best_params')
                        # Fallback for elbow path as well
                        if raw_labels is None and best_params:
                            try:
                                raw_labels = run_single_clustering(
                                    algo,
                                    X_sel,
                                    best_params,
                                    aligned_original_labels=y_sel,
                                    global_known_normal_samples_pca=known_normal_samples_pca,
                                    num_processes_for_algo=args.num_processes,
                                    file_type=args.file_type
                                )
                                logger.warning(f"[{algo}] Recomputed raw_labels (elbow) due to missing cache entry.")
                            except Exception as e:
                                logger.error(f"[{algo}] Fallback clustering (elbow) failed: {e}")
                                raw_labels = None

                    # CNI (binary labeling)
                    # _, jaccard, _ = clustering_nomal_identify(
                    # CNI (binary labeling) - Ignore returned Jaccard and compute directly (like run_chunked_virtual_labeling)
                    final_labels, _, _ = clustering_nomal_identify(
                        data_features_for_clustering=X_sel,
                        clusters_assigned=raw_labels,
                        original_labels_aligned=y_sel,
                        global_known_normal_samples_pca=known_normal_samples_pca,
                        threshold_value=FIXED_CNI_THRESHOLD,
                        num_processes_for_algo=args.num_processes,
                        data_for_clustering=X_sel,
                        known_normal_idx=kn_idx
                    )
                    
                    # Calculate Jaccard directly from final_labels (same logic as run_chunked_virtual_labeling)
                    if final_labels is not None and len(final_labels) == len(y_sel) and np.any(final_labels != -1):
                        jaccard = jaccard_score(y_sel, final_labels, labels=[0,1], average='binary', zero_division=0)
                    else:
                        jaccard = 0.0

                    results[algo].append(jaccard)
                    progress_rows.append({
                        "seed": seed,
                        "algorithm": algo,
                        "jaccard": jaccard
                    })
                    logger.info(f"    Jaccard = {jaccard:.4f}")
                except Exception as e:
                    logger.error(f"[Seed {seed}] {algo} failed: {e}")
                    progress_rows.append({
                        "seed": seed,
                        "algorithm": algo,
                        "jaccard": np.nan,
                        "error": str(e)
                    })

            # flush progress after each seed
            save_progress(progress_path, progress_rows)
            gc.collect()

        # Aggregate results
        summary = []
        for algo, scores in results.items():
            summary.append({
                "algorithm": algo,
                "mean_jaccard": np.nanmean(scores),
                "std_jaccard": np.nanstd(scores),
                "scores": scores
            })

        df = pd.DataFrame(summary)
        save_path = os.path.join(progress_dir, f"seed_stability_{args.file_type}_{args.file_number}.csv")
        df.to_csv(save_path, index=False)
        logger.info(f"Saved result to {save_path}")

    finally:
        # Cleanup progress file if run completed successfully (all seeds covered)
        try:
            # Check completeness
            completed_seeds = {row["seed"] for row in progress_rows}
            if completed_seeds >= set(SEED_LIST) and os.path.exists(progress_path):
                os.remove(progress_path)
                logger.info(f"Removed progress file: {progress_path}")
        except Exception as e:
            logger.warning(f"Could not remove progress file {progress_path}: {e}")
        gc.collect()


if __name__ == "__main__":
    main()
