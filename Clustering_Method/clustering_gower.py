import pandas as pd
import numpy as np
import gower
from k_medoids_python import KMedoids

def get_categorical_feature_indices(X):
    """
    Identifies the indices of categorical features in a DataFrame.
    A feature is considered categorical if it's not numeric or has few unique values.
    """
    if not isinstance(X, pd.DataFrame):
        # If it's a numpy array, we can't safely determine categories. Assume all numeric.
        return []

    categorical_indices = []
    for i, col in enumerate(X.columns):
        # Check if the column is of a non-numeric type
        if not pd.api.types.is_numeric_dtype(X[col]):
            categorical_indices.append(i)
        # Also consider numeric columns with a small number of unique values as categorical
        elif len(X[col].unique()) < 20: # Heuristic: less than 20 unique values
             categorical_indices.append(i)
             
    return categorical_indices

def clustering_gower_kmedoids(X, n_clusters, random_state=42):
    """
    Performs K-Medoids clustering using the Gower distance metric.
    This is suitable for datasets with mixed data types (numerical and categorical).

    Args:
        X (pd.DataFrame or np.ndarray): Input data.
        n_clusters (int): The number of clusters to form.
        random_state (int): Seed for reproducibility.

    Returns:
        np.ndarray: Array of cluster labels for each data point, or None if clustering fails.
    """
    if X is None or X.shape[0] == 0:
        print("[WARN] Gower K-Medoids: Input data is empty.")
        return None

    try:
        print(f"[INFO] Gower K-Medoids: Calculating Gower distance matrix for data of shape {X.shape}...")
        
        # Determine which features are categorical for the distance calculation
        categorical_features = get_categorical_feature_indices(X)
        if categorical_features:
            print(f"[INFO] Gower K-Medoids: Identified {len(categorical_features)} categorical features.")
            cat_feature_flags = [True if i in categorical_features else False for i in range(X.shape[1])]
        else:
            print("[INFO] Gower K-Medoids: No categorical features identified. Treating all as numeric.")
            cat_feature_flags = None # gower library default

        # Calculate the Gower distance matrix
        distance_matrix = gower.gower_matrix(X, cat_features=cat_feature_flags)
        
        print(f"[INFO] Gower K-Medoids: Starting K-Medoids clustering with k={n_clusters}...")
        
        # Initialize and run K-Medoids
        # The 'precomputed' metric tells scikit-learn-extra that we are providing a distance matrix
        kmedoids_instance = KMedoids(n_clusters=n_clusters, method='pam', random_state=random_state)
        
        # fit_predict expects a distance matrix
        clusters = kmedoids_instance.fit_predict(distance_matrix)
        
        print(f"[INFO] Gower K-Medoids: Clustering complete. Found {len(np.unique(clusters))} clusters.")
        
        return clusters

    except Exception as e:
        print(f"[ERROR] Gower K-Medoids clustering failed: {e}")
        import traceback
        traceback.print_exc()
        return None
