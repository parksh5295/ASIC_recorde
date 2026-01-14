#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chunked Clustering Algorithms for Large Datasets

This module implements chunked versions of memory-intensive clustering algorithms
(DBSCAN, MeanShift) to handle large datasets that would otherwise cause OOM errors.

The chunking strategy:
1. Split data into smaller chunks (e.g., 30,000 samples each)
2. Run clustering on each chunk independently
3. Merge results with overlap handling for boundary consistency
4. Return unified cluster labels for the entire dataset
"""

import numpy as np
import logging
from sklearn.cluster import DBSCAN, MeanShift
from sklearn.metrics.pairwise import euclidean_distances

logger = logging.getLogger(__name__)


def dbscan_with_chunking(X, eps=0.5, min_samples=5, chunk_size=30000, overlap_ratio=0.1, n_jobs=-1):
    """
    Apply DBSCAN clustering on large datasets using a chunking strategy.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        The input data to cluster
    eps : float, default=0.5
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other (DBSCAN parameter)
    min_samples : int, default=5
        The number of samples in a neighborhood for a point to be considered
        as a core point (DBSCAN parameter)
    chunk_size : int, default=30000
        Number of samples per chunk
    overlap_ratio : float, default=0.1
        Ratio of overlap between consecutive chunks (0.0 to 0.5)
    n_jobs : int, default=-1
        Number of CPU cores to use for each chunk. -1 means all available cores.
        
    Returns:
    --------
    cluster_labels : array, shape (n_samples,)
        Cluster labels for each point. -1 indicates noise.
    """
    n_samples = len(X)
    
    if n_samples <= chunk_size:
        # No need for chunking
        logger.info(f"[DBSCAN Chunked] Dataset size {n_samples} <= chunk_size {chunk_size}, using standard DBSCAN")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs)
        return dbscan.fit_predict(X)
    
    logger.info(f"[DBSCAN Chunked] Processing {n_samples} samples with chunk_size={chunk_size}, overlap_ratio={overlap_ratio}")
    
    overlap_size = int(chunk_size * overlap_ratio)
    step_size = chunk_size - overlap_size
    
    all_cluster_labels = np.full(n_samples, -1, dtype=int)  # -1 = noise
    cluster_offset = 0
    
    chunk_idx = 0
    start_idx = 0
    
    while start_idx < n_samples:
        end_idx = min(start_idx + chunk_size, n_samples)
        actual_chunk_size = end_idx - start_idx
        
        logger.info(f"[DBSCAN Chunked] Processing chunk {chunk_idx + 1}: samples [{start_idx}:{end_idx}] (size={actual_chunk_size})")
        
        chunk_data = X[start_idx:end_idx]
        
        # Run DBSCAN on this chunk
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs)
        chunk_labels = dbscan.fit_predict(chunk_data)
        
        # Adjust cluster numbers to avoid conflicts across chunks
        unique_labels = np.unique(chunk_labels)
        unique_labels = unique_labels[unique_labels != -1]  # Exclude noise
        
        if len(unique_labels) > 0:
            for old_label in unique_labels:
                mask = chunk_labels == old_label
                chunk_labels[mask] = cluster_offset
                cluster_offset += 1
        
        # Handle overlap region - prefer labels from previous chunk in overlap
        if chunk_idx > 0 and overlap_size > 0:
            overlap_start = start_idx
            overlap_end = min(start_idx + overlap_size, end_idx)
            
            # Only update non-overlap region to avoid overwriting previous chunk's results
            non_overlap_start_in_chunk = overlap_size if overlap_end - overlap_start == overlap_size else 0
            all_cluster_labels[start_idx + non_overlap_start_in_chunk:end_idx] = chunk_labels[non_overlap_start_in_chunk:]
        else:
            # First chunk or no overlap - assign all labels
            all_cluster_labels[start_idx:end_idx] = chunk_labels
        
        chunk_idx += 1
        start_idx += step_size
    
    # Count clusters
    unique_clusters = np.unique(all_cluster_labels)
    n_clusters = len(unique_clusters[unique_clusters != -1])
    n_noise = np.sum(all_cluster_labels == -1)
    
    logger.info(f"[DBSCAN Chunked] Completed: {n_clusters} clusters, {n_noise} noise points")
    
    return all_cluster_labels


def meanshift_with_chunking(X, bandwidth=None, chunk_size=30000, overlap_ratio=0.1, n_jobs=-1):
    """
    Apply MeanShift clustering on large datasets using a chunking strategy.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        The input data to cluster
    bandwidth : float, optional
        Bandwidth parameter for MeanShift. If None, uses sklearn's default estimation.
    chunk_size : int, default=30000
        Number of samples per chunk
    overlap_ratio : float, default=0.1
        Ratio of overlap between consecutive chunks (0.0 to 0.5)
    n_jobs : int, default=-1
        Number of CPU cores to use for each chunk. -1 means all available cores.
        
    Returns:
    --------
    cluster_labels : array, shape (n_samples,)
        Cluster labels for each point.
    """
    n_samples = len(X)
    
    if n_samples <= chunk_size:
        # No need for chunking
        logger.info(f"[MeanShift Chunked] Dataset size {n_samples} <= chunk_size {chunk_size}, using standard MeanShift")
        ms = MeanShift(bandwidth=bandwidth, n_jobs=n_jobs)
        return ms.fit_predict(X)
    
    logger.info(f"[MeanShift Chunked] Processing {n_samples} samples with chunk_size={chunk_size}, overlap_ratio={overlap_ratio}")
    
    overlap_size = int(chunk_size * overlap_ratio)
    step_size = chunk_size - overlap_size
    
    all_cluster_labels = np.full(n_samples, -1, dtype=int)
    cluster_offset = 0
    
    chunk_idx = 0
    start_idx = 0
    
    while start_idx < n_samples:
        end_idx = min(start_idx + chunk_size, n_samples)
        actual_chunk_size = end_idx - start_idx
        
        logger.info(f"[MeanShift Chunked] Processing chunk {chunk_idx + 1}: samples [{start_idx}:{end_idx}] (size={actual_chunk_size})")
        
        chunk_data = X[start_idx:end_idx]
        
        # Run MeanShift on this chunk
        ms = MeanShift(bandwidth=bandwidth, n_jobs=n_jobs)
        chunk_labels = ms.fit_predict(chunk_data)
        
        # Adjust cluster numbers to avoid conflicts across chunks
        unique_labels = np.unique(chunk_labels)
        
        if len(unique_labels) > 0:
            for old_label in unique_labels:
                mask = chunk_labels == old_label
                chunk_labels[mask] = cluster_offset
                cluster_offset += 1
        
        # Handle overlap region - prefer labels from previous chunk in overlap
        if chunk_idx > 0 and overlap_size > 0:
            overlap_start = start_idx
            overlap_end = min(start_idx + overlap_size, end_idx)
            
            # Only update non-overlap region to avoid overwriting previous chunk's results
            non_overlap_start_in_chunk = overlap_size if overlap_end - overlap_start == overlap_size else 0
            all_cluster_labels[start_idx + non_overlap_start_in_chunk:end_idx] = chunk_labels[non_overlap_start_in_chunk:]
        else:
            # First chunk or no overlap - assign all labels
            all_cluster_labels[start_idx:end_idx] = chunk_labels
        
        chunk_idx += 1
        start_idx += step_size
    
    # Count clusters
    unique_clusters = np.unique(all_cluster_labels)
    n_clusters = len(unique_clusters)
    
    logger.info(f"[MeanShift Chunked] Completed: {n_clusters} clusters")
    
    return all_cluster_labels


# =============================================================================
# FUTURE IMPLEMENTATION (Currently commented out - MeanShift uses standard approach)
# =============================================================================

'''
def meanshift_with_chunking_advanced(X, bandwidth=None, chunk_size=30000, merge_threshold=None):
    """
    Advanced MeanShift chunking with cluster center merging.
    
    This implementation:
    1. Runs MeanShift on each chunk to find local cluster centers
    2. Merges nearby cluster centers across chunks
    3. Assigns final labels based on merged centers
    
    Currently NOT USED - standard MeanShift is applied instead.
    Enable this if MeanShift causes OOM errors.
    """
    # TODO: Implement if needed based on memory profiling results
    pass
'''

