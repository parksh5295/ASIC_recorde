# clustering_recommender.py

import argparse
import time
import os
import pandas as pd
from Dataset_Choose_Rule.choose_amount_dataset import file_path_line_nonnumber, file_cut
from definition.Anomal_Judgment import anomal_judgment_nonlabel, anomal_judgment_label
from utils.time_transfer import time_scalar_transfer
from Modules.Heterogeneous_module import choose_heterogeneous_method
from Heterogeneous_Method.separate_group_mapping import map_intervals_to_groups
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.metrics import silhouette_score
from scipy.stats import kurtosis
from sklearn.feature_selection import VarianceThreshold
from tqdm import tqdm

# --- Project-specific imports ---
try:
    from Dataset_Choose_Rule.choose_amount_dataset import file_path_line_nonnumber, file_cut
    from definition.Anomal_Judgment import anomal_judgment_nonlabel, anomal_judgment_label
    from utils.time_transfer import time_scalar_transfer
    from Modules.Heterogeneous_module import choose_heterogeneous_method
    from Modules.PCA import pca_func # Import the PCA function
    from utils.minmaxscaler import apply_minmax_scaling_and_save_scalers # Import the scaler function
except ImportError as e:
    print(f"Could not import a project module: {e}")


def recommend_clustering_by_distribution(X_df):
    X = X_df.values
    n_samples, n_features = X.shape
    stats = {'n_samples': n_samples, 'n_features': n_features}

    # GMM covariance ratio
    reg_value = 1e-6 # Add regularization for numerical stability
    try:
        gmm_diag = GaussianMixture(n_components=2, covariance_type='diag', random_state=0, reg_covar=reg_value).fit(X)
        gmm_full = GaussianMixture(n_components=2, covariance_type='full', random_state=0, reg_covar=reg_value).fit(X)
        diag_ratios = [np.sum(np.diag(f)) / np.sum(f) for f in gmm_full.covariances_]
        avg_diag_ratio = np.mean(diag_ratios)
    except np.linalg.LinAlgError as e:
        print(f"[Warning] GMM fitting failed due to LinAlgError: {e}. Using default avg_diag_ratio = 0.5")
        avg_diag_ratio = 0.5 # Assign a default value or handle differently
    except ValueError as e:
        print(f"[Warning] GMM fitting failed due to ValueError: {e}. Using default avg_diag_ratio = 0.5")
        avg_diag_ratio = 0.5 # Handle cases like insufficient samples

    stats['covariance_diagonal_ratio'] = avg_diag_ratio

    est_n_clusters = estimate_clusters(X)
    metrics = {'estimated_clusters': est_n_clusters}

        # Apply score-based ranking
    def score_algorithm(name, diag_ratio, n_samples, n_features, est_clusters):
        score = 0
        if name == "GMM":
            if diag_ratio > 0.7:
                score += 3
            elif diag_ratio > 0.6:
                score += 2
            else:
                score -= 1
            if n_features > 10:
                score += 1
        elif name == "KMeans":
            score += 2
            if diag_ratio > 0.6:
                score += 1
        elif name == "DBSCAN":
            if diag_ratio < 0.6:
                score += 3
            if n_samples > 10000:
                score += 1
            if n_features > 20:
                score -= 1
        elif name == "SGMM":
            if diag_ratio > 0.7:
                score += 2
            if n_features > 15:
                score += 2
        elif name == "GK":
            if diag_ratio > 0.65:
                score += 3
        elif name == "NeuralGas":
            score += 1
        elif name == "MeanShift":
            if diag_ratio < 0.6:
                score += 2
        elif name == "FCM":
            if diag_ratio < 0.65:
                score += 2
        elif name == "GMeans":
            if est_clusters >= 5:
                score += 2
        elif name == "Kmedians":
            score += 1
        return score

    recommendations = list(set(recommendations))  # Remove duplicates just in case
    recommendations.sort(key=lambda name: -score_algorithm(name, avg_diag_ratio, n_samples, n_features, est_n_clusters))


    # Recommendation algorithm limit
    recommendations = []

    if avg_diag_ratio > 0.7:
        recommendations += ['GMM', 'KMeans', 'SGMM']
    elif avg_diag_ratio > 0.65:
        recommendations += ['GK', 'GMM', 'KMeans']
    else:
        recommendations += ['DBSCAN', 'MeanShift', 'FCM']

    recommendations += ['NeuralGas']

    reason_summary = f"""
        [Distribution-Based Recommendation]
        - Covariance Diagonal Ratio: {avg_diag_ratio:.2f}
        - Estimated #Clusters: {est_n_clusters}
        - Sample/Cluster Ratio: {n_samples / est_n_clusters:.2f}
        """

    return recommendations, metrics, stats, reason_summary.strip()


def recommend_clustering_by_feature_types(X_df):
    X = X_df.values
    n_samples, n_features = X.shape
    stats = {'n_samples': n_samples, 'n_features': n_features}
    metrics = {'estimated_clusters': min(5, max(2, n_samples // 100))}

    # Recommend simple and stable clustering methods (within specified 11)
    recommendations = ['KMeans', 'Kmedians', 'GMeans', 'NeuralGas']

    if n_features > 20:
        recommendations += ['SGMM']

    reason_summary = f"""
        [Feature-Type-Based Recommendation]
        - Sample Size: {n_samples}, Feature Dim: {n_features}
        - Estimated #Clusters: {metrics['estimated_clusters']}
        - Using simple clustering methods for small or mixed feature data
        """

    return recommendations, metrics, stats, reason_summary.strip()


def estimate_clusters(X):
    # Simple cluster number estimation (silhouette optimization)
    best_score = -1
    best_k = 2
    for k in range(2, min(10, len(X))):
        try:
            labels = KMeans(n_clusters=k, random_state=0).fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except:
            continue
    return best_k



def smart_clustering_selector(X_df, raw_df, threshold_sample=500, threshold_dim=10):
    n_samples, n_features = X_df.shape
    print(f"\n[Info] Sample size: {n_samples}, Feature dim: {n_features}")

    if n_samples < threshold_sample or n_features < threshold_dim:
        print("[Strategy] Using feature-type-based recommendation (small data).")
        algorithms, metrics, stats, reason_summary = recommend_clustering_by_feature_types(X_df)
    else:
        print("[Strategy] Using distribution-based recommendation.")
        
        X_fit = X_df # Use the data as is, it's already a sample if the original was large

        try:
            # Fit GMM on the (potentially sampled) data
            gmm = GaussianMixture(n_components=2, random_state=42, n_init=1, reg_covar=1e-5)
            gmm.fit(X_fit)
            bic = gmm.bic(X_fit)
            avg_diag_ratio = np.mean(np.diag(gmm.covariances_[0])) / np.mean(gmm.covariances_[0])
            
            # Fit K-Means on the sampled data
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10) # MODIFIED: Suppress FutureWarning
            kmeans.fit(X_fit) # Use sampled data
            try:
                silhouette = silhouette_score(X_fit, kmeans.labels_) # Use sampled data
            except (ValueError, TypeError) as e:
                # This can happen if k-means produces only one cluster or other errors
                print(f"[Warning] Silhouette score calculation failed due to {e}. Defaulting to 0.")
                silhouette = 0
                
            # --- Recommendation Logic ---
            if bic < 10000 and silhouette > 0.5:
                algorithms = ["GMM", "KMeans", "DBSCAN"]
            elif avg_diag_ratio > 0.8:
                algorithms = ["KMeans", "Kmedians", "DBSCAN"]
            else:
                algorithms = ["GMM", "KMeans", "MShift"]

            metrics = {"BIC": f"{bic:.2f}", "Silhouette": f"{silhouette:.3f}"}
            stats = {"n_samples": n_samples, "n_features": n_features, "avg_diag_ratio": f"{avg_diag_ratio:.3f}"}
            reason_summary = (
                f"[Distribution-Based Recommendation]\n"
                f"- BIC Score: {bic:.2f} (lower is better)\n"
                f"- Silhouette Score: {silhouette:.3f} (higher is better)\n"
                f"- GMM Covariance Shape (avg_diag_ratio): {avg_diag_ratio:.3f} (closer to 1.0 suggests spherical clusters)"
            )

        except (ValueError, TypeError) as e:
            print(f"[Fallback] Distribution-based strategy failed due to: {e}")
            print("[Fallback] Switching to feature-type-based recommendation.")
            algorithms, metrics, stats, reason_summary = recommend_clustering_by_feature_types(X_df)
            
    return algorithms, metrics, stats, reason_summary


def explain_recommendation(algorithm, metrics, stats):
    explanation = f"[Recommendation] {algorithm}\n"

    if 'covariance_diagonal_ratio' in stats:
        diag_ratio = stats['covariance_diagonal_ratio']
        explanation += f"[Recommendation Reason] The average covariance matrix has a diagonal ratio of {diag_ratio:.2f}, which suggests that GMM(diagonal) can effectively describe the data distribution.\n"

    if 'estimated_clusters' in metrics:
        n_clusters = metrics['estimated_clusters']
        explanation += f"[Recommendation Reason] The estimated number of clusters is {n_clusters}.\n"

    if 'n_samples' in stats and 'estimated_clusters' in metrics:
        ratio = metrics['estimated_clusters'] / stats['n_samples']
        explanation += f"[Recommendation Reason] The ratio of the number of samples to the number of clusters is {ratio:.4f}.\n"

    return explanation


def save_recommendation_to_csv(file_type, file_number, algorithms, metrics, stats, reason_summary):
    output_dir = f"../Dataset/recommend/"
    os.makedirs(output_dir, exist_ok=True)

    file_type_dir = os.path.join(output_dir, file_type)
    os.makedirs(file_type_dir, exist_ok=True)

    save_path = os.path.join(file_type_dir, f"{file_type}_{file_number}_recommendation.csv")

    df = pd.DataFrame({
        "Recommended_Algorithms": algorithms,
        "Best_Algorithm": [algorithms[0]] * len(algorithms),
        "Estimated_Clusters": [metrics.get("estimated_clusters", None)] * len(algorithms),
        "Cov_Diagonal_Ratio": [stats.get("covariance_diagonal_ratio", None)] * len(algorithms),
        "n_Samples": [stats.get("n_samples", None)] * len(algorithms),
        "n_Features": [stats.get("n_features", None)] * len(algorithms),
        "Reason_Summary": [reason_summary] * len(algorithms),
    })

    df.to_csv(save_path, index=False)
    print(f"\nSaved recommendation result to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Recommend clustering method for given dataset")
    parser.add_argument('--file_type', type=str, default="MiraiBotnet")
    parser.add_argument('--file_number', type=int, default=1)
    args = parser.parse_args()

    file_type = args.file_type
    file_number = args.file_number

    total_start = time.time()

    # 1. Load data
    file_path, _ = file_path_line_nonnumber(file_type, file_number)
    cut_type = 'random' if file_type in ['DARPA98', 'DARPA', 'NSL-KDD', 'NSL_KDD'] else 'all'
    data = file_cut(file_type, file_path, cut_type)

    # --- NEW: Sample the data EARLY if it's too large, BEFORE heavy preprocessing ---
    SAMPLE_SIZE = 100000
    n_samples_initial = len(data)
    if n_samples_initial > SAMPLE_SIZE:
        print(f"[Info] Initial dataset is large ({n_samples_initial} rows). "
              f"Creating a random sample of {SAMPLE_SIZE} rows BEFORE preprocessing to improve performance.")
        data = data.sample(n=SAMPLE_SIZE, random_state=42)
    # --- END NEW ---

    # Clean column names by stripping leading/trailing whitespace
    data.columns = data.columns.str.strip()

    # 2. Apply label
    if file_type in ['MiraiBotnet', 'NSL-KDD']:
        data['label'], _ = anomal_judgment_nonlabel(file_type, data)
    elif file_type == 'netML':
        data['label'] = data['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    elif file_type == 'DARPA98':
        data['label'] = data['Class'].apply(lambda x: 0 if x == '-' else 1)
    else:
        data['label'] = anomal_judgment_label(data)

    # 3. Time + embedding
    data = time_scalar_transfer(data, file_type)
    embedded_df, feature_list, category_mapping, data_list = choose_heterogeneous_method(
        data, file_type, het_method="Interval_inverse", regul='N'
    )
    '''
    # X, _ = map_intervals_to_groups(embedded_df, category_mapping, data_list, regul='N') # REMOVED: This step is conceptually incorrect for distribution analysis.

    # USE THE DATA BEFORE MAPPING for a more accurate distribution analysis.
    X = embedded_df.copy()
    '''
    
    # --- NEW STRATEGY: Rebuild the dataframe from pure numerical parts ---
    # The 'embedded_df' contains Interval objects which cause errors in analysis.
    # The 'data_list' contains the raw numerical dataframes before they were converted.
    # We will combine these parts to get a clean, purely numerical dataframe that includes all features.
    
    # data_list contains: [categorical_df, time_df, packet_length_df, count_df, binary_df]
    # We concatenate all non-empty parts.
    X_parts = [df for df in data_list if not df.empty]
    if not X_parts:
        # Fallback if data_list is empty for some reason
        print("[Error] No valid data parts found after preprocessing. Defaulting to empty DataFrame.")
        X = pd.DataFrame()
    else:
        X = pd.concat(X_parts, axis=1)

    # --- NEW: Replace infinite values with NaN so they can be dropped ---
    # The preprocessing step can sometimes generate infinity values (e.g., division by zero).
    # We replace them with NaN here, so the subsequent dropna() call removes them.
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # --- FIX: Drop rows with NaN and log the changes ---
    if X.isnull().values.any():
        #X.fillna(0, inplace=True)
        original_rows = X.shape[0]
        print(f"[Warning] NaN values found in pre-processed data (shape: {X.shape}). Dropping rows with NaN.")
        X.dropna(inplace=True)
        new_rows = X.shape[0]
        print(f"[Info] Removed {original_rows - new_rows} rows containing NaN values.")
        print(f"[Info] Data shape changed from ({original_rows}, {X.shape[1]}) to ({new_rows}, {X.shape[1]}).")

    if X.empty:
        print("[Error] No data remaining after removing NaN values. Cannot proceed with clustering recommendation.")
        return
    # --- END FIX ---

    # --- Drop duplicate rows that can cause GMM to fail ---
    original_rows = X.shape[0]
    X.drop_duplicates(inplace=True)
    new_rows = X.shape[0]
    if original_rows > new_rows:
        print(f"[Info] Removed {original_rows - new_rows} duplicate rows.")
        print(f"[Info] Data shape changed from ({original_rows}, {X.shape[1]}) to ({new_rows}, {X.shape[1]}).")
    # --- END NEW ---

    # --- NEW: Remove low-variance (constant) features from NUMERICAL columns only ---
    
    original_shape = X.shape
    
    # Separate numerical and non-numerical columns
    numerical_cols = X.select_dtypes(include=np.number).columns
    non_numerical_cols = X.select_dtypes(exclude=np.number).columns
    
    if not numerical_cols.empty:
        # Apply VarianceThreshold only on the numerical part of the data
        selector = VarianceThreshold(threshold=0)
        
        try:
            # Fit on the numerical data
            numerical_data = X[numerical_cols]
            selector.fit(numerical_data)
            
            # Get the mask of features to keep
            features_kept_mask = selector.get_support()
            numerical_cols_kept = numerical_cols[features_kept_mask]
            
            # Check if any columns were removed
            if len(numerical_cols_kept) < len(numerical_cols):
                removed_cols = numerical_cols[~features_kept_mask]
                print(f"[Info] Removed {len(removed_cols)} constant-value numerical feature(s): {list(removed_cols)}")
                
                # Reconstruct X with only the kept numerical columns and all non-numerical columns
                all_kept_cols = list(numerical_cols_kept) + list(non_numerical_cols)
                X = X[all_kept_cols]
                print(f"[Info] Data shape changed from {original_shape} to {X.shape}.")
            else:
                print("[Info] No constant-value numerical features found to remove.")

        except Exception as e:
            print(f"[Warning] Could not apply variance threshold selection due to an error: {e}")
            print("[Warning] Proceeding with the original data, but clustering might fail.")
    else:
       print("[Info] No numerical features found to apply variance threshold.")
    # --- END NEW ---

    # --- NEW: Apply Scaling and PCA before model fitting ---
    X_final = X
    try:
        print("[Info] Applying Min-Max scaling before PCA...")
        # We need to provide dummy file_number and het_method as the scaler function expects them
        X_scaled, _ = apply_minmax_scaling_and_save_scalers(X, file_type, 1, "Interval_inverse")
        
        # Conditionally apply PCA, similar to Data_Labeling_Evaluate_Thresholds.py
        if file_type not in ['DARPA98', 'DARPA']:
            print("[Info] Applying PCA for dimensionality reduction...")
            X_reduced = pca_func(X_scaled)
            # pca_func returns a numpy array, convert it back to a DataFrame, preserving the index
            X_final = pd.DataFrame(X_reduced, index=X.index)
            print(f"[Info] Data shape after PCA: {X_final.shape}")
        else:
            print("[Info] Skipping PCA for this dataset type.")
            X_final = X_scaled
    except Exception as e:
        print(f"[Warning] Scaling or PCA failed due to an error: {e}")
        print("[Warning] Proceeding with pre-PCA data.")
        X_final = X # Fallback to data before scaling/PCA
    # --- END NEW ---

    # 3. Recommend clustering method based on data characteristics
    print(f"[Info] Sample size: {X_final.shape[0]}, Feature dim: {X_final.shape[1]}")

    # 4. Recommend clustering method
    recommendations, metrics, stats, recommendation_explanation = smart_clustering_selector(X_final, data)

    print("\nRecommended Clustering Algorithms:")
    for i, algo in enumerate(recommendations, 1):
        print(f"  {i}. {algo}")

    print(f"\nExplanation:\n{recommendation_explanation}")

    # 5. Save to CSV
    save_recommendation_to_csv(file_type, file_number, recommendations, metrics, stats, recommendation_explanation)

    print(f"\nDone in {time.time() - total_start:.2f} seconds.")


if __name__ == '__main__':
    main()
