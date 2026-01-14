import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
import logging
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)

def handle_darpa98_mshift_special(X, params, file_type):
    """
    Special handler for MeanShift on large DARPA98 datasets to prevent excessive run times.

    This function overrides the tuned quantile/bandwidth with a more reasonable fixed value,
    then applies a "fit on sample, predict on chunks" strategy.

    Args:
        X (np.ndarray): The full dataset for clustering.
        params (dict): The original parameters (likely containing a problematic bandwidth).
        file_type (str): The type of the dataset file.

    Returns:
        np.ndarray: The resulting cluster labels for the full dataset.
    """
    logger.warning("[MShift-DARPA98] Activating special handling routine due to performance issues.")
    logger.warning(f"[MShift-DARPA98] Original bandwidth was {params.get('bandwidth')}. This will be overridden.")

    # 1. Override quantile and re-estimate bandwidth on a sample
    # Using a fixed quantile of 0.2 has been found to be more robust for DARPA98.
    # The sample size for bandwidth estimation should also be reasonable.
    fixed_quantile = 0.2
    bandwidth_sample_size = 20000 
    
    if X.shape[0] > bandwidth_sample_size:
        np.random.seed(42)
        sample_indices = np.random.choice(X.shape[0], bandwidth_sample_size, replace=False)
        X_sample_bw = X[sample_indices]
    else:
        X_sample_bw = X
        
    logger.info(f"[MShift-DARPA98] Re-estimating bandwidth with fixed quantile={fixed_quantile} on a sample of {X_sample_bw.shape[0]} rows.")
    
    try:
        # Use n_jobs=1 for estimation to avoid memory issues on large samples
        new_bandwidth = estimate_bandwidth(X_sample_bw, quantile=fixed_quantile, n_samples=min(10000, X_sample_bw.shape[0]), n_jobs=1)
        if new_bandwidth <= 0:
            new_bandwidth = 1.0 # Fallback for safety
        logger.info(f"[MShift-DARPA98] New bandwidth estimated: {new_bandwidth}")
    except Exception as e:
        logger.error(f"[MShift-DARPA98] Failed to estimate new bandwidth: {e}. Falling back to a default value of 1.0.")
        new_bandwidth = 1.0

    # 2. Fit the model on a sample using the new bandwidth
    fit_sample_size = 20000
    if X.shape[0] > fit_sample_size:
        np.random.seed(42) # Use same seed for consistency
        fit_indices = np.random.choice(X.shape[0], fit_sample_size, replace=False)
        X_sample_fit = X[fit_indices]
    else:
        X_sample_fit = X

    # Use n_jobs=1 for the fit process to prevent memory thrashing
    model = MeanShift(bandwidth=new_bandwidth, n_jobs=1, bin_seeding=True)
    
    logger.info(f"[MShift-DARPA98] Fitting model on a sample of size {X_sample_fit.shape[0]}... This may still take time.")
    start_fit_time = time.time()
    model.fit(X_sample_fit)
    end_fit_time = time.time()
    logger.info(f"[MShift-DARPA98] Model fitting completed in {end_fit_time - start_fit_time:.2f} seconds.")

    # 3. Predict on the full dataset in chunks
    chunk_size = 30000
    num_chunks = int(np.ceil(X.shape[0] / chunk_size))
    labels = np.zeros(X.shape[0], dtype=int)
    
    logger.info(f"[MShift-DARPA98] Predicting labels for the full dataset in {num_chunks} chunks...")
    
    for i in tqdm(range(num_chunks), desc="[MShift-DARPA98] Predicting Chunks"):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, X.shape[0])
        chunk = X[start_idx:end_idx]
        
        chunk_labels = model.predict(chunk)
        labels[start_idx:end_idx] = chunk_labels

    logger.info("[MShift-DARPA98] Special handling routine completed successfully.")
    return labels
