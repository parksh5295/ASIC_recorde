import os
import csv
import json
from datetime import datetime
import numpy as np


def get_surrogate_diagnostics_file_path(algorithm, data_hash):
    """Get the diagnostics file path for surrogate score issues."""
    progress_dir = "../Dataset_ex/progress_tracking"
    
    # Check if the hash indicates a temporary chunk file
    if data_hash.startswith("temp_chunk_"):
        filename = f"chunk_diagnostics_{data_hash}_{algorithm}.csv"
    else:
        filename = f"surrogate_diagnostics_{data_hash}_{algorithm}.csv"
        
    return os.path.join(progress_dir, filename)

def save_surrogate_diagnostics(algorithm, data_hash, k, issue_type, details, nan_count=0):
    """
    Save diagnostic information about surrogate score issues.
    
    Args:
        algorithm: Algorithm name
        data_hash: Data hash
        k: Parameter value
        issue_type: Type of issue (single_cluster, same_position, empty_cluster, outlier, nan_inf, etc.)
        details: Detailed information about the issue
        nan_count: Number of NaN values found
    """
    diagnostics_file = get_surrogate_diagnostics_file_path(algorithm, data_hash)
    
    # Define header
    header = ['timestamp', 'algorithm', 'data_hash', 'k', 'issue_type', 'details', 'nan_count']
    
    # Create row data
    row_data = [
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        algorithm,
        data_hash,
        k,
        issue_type,
        json.dumps(details) if isinstance(details, dict) else str(details),
        nan_count
    ]
    
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(diagnostics_file), exist_ok=True)
        
        # Write header if file doesn't exist, then append data
        file_exists = os.path.exists(diagnostics_file)
        with open(diagnostics_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row_data)
            
    except Exception as e:
        print(f"Error saving surrogate diagnostics: {e}")

def diagnose_clustering_issues(X, labels, k, algorithm, data_hash):
    """
    Diagnose potential issues that could cause NaN in surrogate score calculation.
    
    Returns:
        tuple: (issue_detected, issue_type, details, nan_count)
    """
    issues = []
    nan_count = 0
    
    # Convert to numpy arrays for easier handling
    X = np.array(X)
    labels = np.array(labels)
    
    # 1. Check for NaN/Inf in input data
    nan_mask = np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1)
    nan_count = nan_mask.sum()
    if nan_count > 0:
        issues.append({
            'type': 'nan_inf_data',
            'count': int(nan_count),
            'details': f"Found {nan_count} rows with NaN/Inf values"
        })
    
    # 2. Check for single cluster
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        issues.append({
            'type': 'single_cluster',
            'count': len(unique_labels),
            'details': f"Only {len(unique_labels)} unique cluster(s) found"
        })
    
    # 3. Check for empty clusters
    cluster_counts = np.bincount(labels[labels >= 0])  # Only count non-negative labels
    empty_clusters = np.sum(cluster_counts == 0)
    if empty_clusters > 0:
        issues.append({
            'type': 'empty_cluster',
            'count': int(empty_clusters),
            'details': f"Found {empty_clusters} empty clusters"
        })
    
    # 4. Check for same position data (within each cluster)
    for cluster_id in unique_labels:
        if cluster_id < 0:  # Skip noise labels
            continue
            
        cluster_mask = labels == cluster_id
        cluster_data = X[cluster_mask]
        
        if len(cluster_data) > 1:
            # Check if all points in cluster are at same position
            centroid = np.mean(cluster_data, axis=0)
            distances = np.linalg.norm(cluster_data - centroid, axis=1)
            
            if np.all(distances < 1e-10):  # Very close to centroid
                issues.append({
                    'type': 'same_position',
                    'cluster': int(cluster_id),
                    'details': f"All {len(cluster_data)} points in cluster {cluster_id} are at same position"
                })
    
    # 5. Check for outliers (using IQR method)
    for cluster_id in unique_labels:
        if cluster_id < 0:
            continue
            
        cluster_mask = labels == cluster_id
        cluster_data = X[cluster_mask]
        
        if len(cluster_data) > 3:  # Need at least 4 points for meaningful outlier detection
            # Check each dimension for outliers
            for dim in range(cluster_data.shape[1]):
                values = cluster_data[:, dim]
                if not np.isnan(values).all() and not np.isinf(values).all():
                    Q1 = np.percentile(values, 25)
                    Q3 = np.percentile(values, 75)
                    IQR = Q3 - Q1
                    
                    if IQR > 0:  # Avoid division by zero
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outliers = np.sum((values < lower_bound) | (values > upper_bound))
                        
                        if outliers > 0:
                            issues.append({
                                'type': 'outlier',
                                'cluster': int(cluster_id),
                                'dimension': int(dim),
                                'count': int(outliers),
                                'details': f"Found {outliers} outliers in cluster {cluster_id}, dimension {dim}"
                            })
    
    # 6. Test centroids calculation specifically
    try:
        import pandas as pd
        X_df = pd.DataFrame(X)
        X_df['label'] = labels
        centroids = X_df.groupby('label').transform('mean')
        
        # Check if centroids contain NaN
        centroid_nan_count = centroids.isnull().sum().sum()
        if centroid_nan_count > 0:
            issues.append({
                'type': 'centroid_nan',
                'count': int(centroid_nan_count),
                'details': f"Centroids calculation produced {centroid_nan_count} NaN values"
            })
            
    except Exception as e:
        issues.append({
            'type': 'centroid_error',
            'details': f"Error in centroids calculation: {str(e)}"
        })
    
    # Save diagnostics if issues found
    if issues:
        for issue in issues:
            save_surrogate_diagnostics(algorithm, data_hash, k, issue['type'], issue, nan_count)
    
    return len(issues) > 0, issues, nan_count
