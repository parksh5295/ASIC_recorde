import numpy as np
import pandas as pd

def compute_surrogate_score(X, labels, known_normal_idx):
    """
    Optimized surrogate score calculation using vectorized operations.
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    valid_labels_mask = unique_labels != -1
    
    if not np.any(valid_labels_mask):
        return 0.0
        
    valid_labels = unique_labels[valid_labels_mask]
    
    # --- 1. Vectorized Purity Calculation ---
    # Create a boolean array indicating if each sample is a known normal
    is_normal_arr = np.zeros(len(labels), dtype=bool)
    if len(known_normal_idx) > 0:
        is_normal_arr[known_normal_idx] = True

    # Use pandas for efficient grouped operations
    df = pd.DataFrame({'label': labels, 'is_normal': is_normal_arr})
    
    # Filter out noise points
    df = df[df['label'] != -1]
    
    # Count normal/attack samples per cluster
    counts_df = df.groupby('label')['is_normal'].value_counts().unstack(fill_value=0)
    
    # Ensure both True (normal) and False (attack) columns exist
    if True not in counts_df:
        counts_df[True] = 0
    if False not in counts_df:
        counts_df[False] = 0
    
    n_normal = counts_df[True]
    n_attack = counts_df[False]
    cluster_sizes = n_normal + n_attack
    
    # Calculate purity for all clusters at once
    purity_per_cluster = np.maximum(n_normal, n_attack) / cluster_sizes
    purity_score = purity_per_cluster.mean()

    # --- 2. Vectorized Compactness Calculation ---
    # Convert X to DataFrame for easier grouping if it's not already
    if not isinstance(X, pd.DataFrame):
        X_df = pd.DataFrame(X)
    else:
        X_df = X
    
    X_df['label'] = labels
    X_df_no_noise = X_df[X_df['label'] != -1]
    
    # Calculate centroids for all clusters at once
    centroids = X_df_no_noise.groupby('label').transform('mean')
    
    # Calculate squared distances for all points at once
    squared_distances = ((X_df_no_noise.drop('label', axis=1) - centroids)**2).sum(axis=1)
    
    # Calculate mean compactness for each cluster, then average them
    compactness_score = squared_distances.groupby(X_df_no_noise['label']).mean().mean()

    # --- 3. Final surrogate score ---
    surrogate_score = purity_score / (1 + compactness_score + 1e-9)
    return surrogate_score