import argparse
import time
import os
import pandas as pd
import numpy as np


# --- Project-specific imports from recommend_clustering_2.py ---
from Dataset_Choose_Rule.choose_amount_dataset import file_path_line_nonnumber, file_cut
from definition.Anomal_Judgment import anomal_judgment_nonlabel, anomal_judgment_label
from utils.time_transfer import time_scalar_transfer
from Modules.Heterogeneous_module import choose_heterogeneous_method
from utils.apply_labeling import apply_labeling_logic
try:
    from Modules.PCA import pca_func
    from utils.minmaxscaler import apply_minmax_scaling_and_save_scalers
except ImportError as e:
    print(f"Could not import a project module: {e}")
    pca_func = None
    apply_minmax_scaling_and_save_scalers = None
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score
from scipy.stats import kurtosis
from collections import Counter
from kneed import KneeLocator
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
# --- Import Surrogate Score and dependencies ---
from Tuning_hyperparameter.Surrogate_score import compute_surrogate_score_optimized


def print_and_prepare_data(families):
    """Prints the algorithm families to the console and prepares a DataFrame for CSV export."""
    print("="*70)
    print("Clustering Algorithm Families Guide")
    print("="*70)
    print("This guide starts with one fundamental question: 'What defines a cluster?'\n")

    data_for_csv = []
    
    for family_name, details in families.items():
        print(f"\n--- {family_name} ---\n")
        print(f"  Core Idea: {details['core_idea']}")
        print(f"  Algorithms: {', '.join(details['algorithms'])}")
        print(f"  Use When: {details['use_when']}\n")

        for algo in details['algorithms']:
            data_for_csv.append({
                "Family": family_name,
                "Algorithm": algo,
                "Core_Idea": details['core_idea'],
                "Use_When": details['use_when']
            })

    return pd.DataFrame(data_for_csv)

def save_recommendation_to_csv(df, file_type, file_number):
    """Saves the categorized algorithm data to a CSV file."""
    output_dir = f"../Dataset/recommend/"
    os.makedirs(output_dir, exist_ok=True)

    file_type_dir = os.path.join(output_dir, file_type)
    os.makedirs(file_type_dir, exist_ok=True)

    save_path = os.path.join(file_type_dir, f"{file_type}_{file_number}_recommendation_3.csv")
    
    df.to_csv(save_path, index=False)
    print(f"\nSaved algorithm families to: {save_path}")

# --- Analysis Functions for Decision Tree ---

def get_known_normal_indices(X_final, sample_size=50000):
    """
    Creates a sample of known normal indices for the surrogate score.
    In this unsupervised script, we treat the largest cluster from a preliminary
    run as the 'known normal' group for analysis purposes.
    """
    n_samples = X_final.shape[0]
    if n_samples < 20:
        return np.array([])

    sample = X_final
    if n_samples > sample_size:
        idx = np.random.choice(n_samples, sample_size, replace=False)
        sample = X_final[idx]
    
    # Preliminary clustering to find the largest cluster
    try:
        # Use a reasonable k based on data size
        n_clusters_prelim = max(2, min(10, int(np.sqrt(sample.shape[0]) / 10)))
        mbk = MiniBatchKMeans(n_clusters=n_clusters_prelim, n_init=3, batch_size=1024, random_state=42).fit(sample)
        labels = mbk.labels_
        
        # Find the label of the largest cluster
        largest_cluster_label = Counter(labels).most_common(1)[0][0]
        
        # Get indices of points in the largest cluster
        known_normal_idx = np.where(labels == largest_cluster_label)[0]
        
        # If original indices were sampled, map back (not needed here as we return indices of the sample)
        return known_normal_idx
    except Exception:
        return np.array([])

def estimate_n_clusters_surrogate(X, known_normal_idx, sample_size=50000, max_k=200):
    """
    Estimates k using the Elbow method on the Surrogate Score (assumes spherical clusters).
    """
    n_samples = X.shape[0]
    if n_samples < 10:
        return 2

    X_sample = X
    kn_idx_sample = known_normal_idx
    if n_samples > sample_size:
        # We need to sample both data and known_normal_indices consistently
        sample_indices = np.random.choice(n_samples, sample_size, replace=False)
        X_sample = X[sample_indices]
        
        # Filter known_normal_idx to only include those in the new sample
        kn_mask = np.isin(known_normal_idx, sample_indices)
        kn_original_indices = known_normal_idx[kn_mask]
        
        # Create a mapping from original index to new sample index
        mapper = {orig_idx: new_idx for new_idx, orig_idx in enumerate(sample_indices)}
        kn_idx_sample = np.array([mapper[i] for i in kn_original_indices])

    k_range = range(2, min(max_k + 1, X_sample.shape[0] // 10))
    surrogate_scores = []

    print(f"  > [K-Est 1/2] Testing k from {min(k_range)} to {max(k_range)} using Surrogate Score...")
    for k in tqdm(k_range, desc="Estimating k (Surrogate)"):
        try:
            mbk = MiniBatchKMeans(n_clusters=k, n_init=3, batch_size=1024, random_state=42)
            labels = mbk.fit_predict(X_sample)
            
            if len(set(labels)) < 2:
                surrogate_scores.append(np.nan)
                continue

            # Calculate the actual surrogate score
            score_dict = compute_surrogate_score_optimized(X_sample, labels, kn_idx_sample)
            surrogate_scores.append(score_dict['final'])

        except Exception:
            surrogate_scores.append(np.nan)
            
    valid_indices = ~np.isnan(surrogate_scores)
    if not np.any(valid_indices):
        return 3

    valid_k = np.array(list(k_range))[valid_indices]
    valid_scores = np.array(surrogate_scores)[valid_indices]

    if len(valid_k) < 3:
        return valid_k[np.argmax(valid_scores)] if len(valid_k) > 0 else 3

    try:
        kn = KneeLocator(valid_k, valid_scores, curve='concave', direction='increasing')
        if kn.knee:
            return kn.knee
        else:
            return valid_k[np.argmax(valid_scores)]
    except Exception:
        return valid_k[np.argmax(valid_scores)]

def estimate_n_clusters_bic(X, sample_size=50000, max_k=200):
    """
    Estimates k using the BIC score from Gaussian Mixture Models (assumes complex cluster shapes).
    Tests ~50 k-values spread across the range up to max_k.
    """
    n_samples = X.shape[0]
    if n_samples < 10:
        return 2

    X_sample = X
    if n_samples > sample_size:
        sample_indices = np.random.choice(n_samples, sample_size, replace=False)
        X_sample = X[sample_indices]
    
    # Define the upper bound for k, ensuring it's reasonable for the sample size
    upper_k_bound = min(max_k, X_sample.shape[0] // 20)
    
    # If the upper bound is too small to generate a meaningful range, use a simple range
    if upper_k_bound < 10:
        k_range = range(2, upper_k_bound + 1)
    else:
        # Generate ~50 k-values evenly spaced between 2 and the upper bound
        k_range = np.unique(np.linspace(2, upper_k_bound, 50, dtype=int))

    bics = []
    
    print(f"  > [K-Est 2/2] Testing {len(k_range)} k-values from {min(k_range)} to {max(k_range)} using GMM BIC...")
    for k in tqdm(k_range, desc="Estimating k (BIC)"):
        try:
            gmm = GaussianMixture(n_components=k, n_init=3, random_state=42, reg_covar=1e-5)
            gmm.fit(X_sample)
            bics.append(gmm.bic(X_sample))
        except Exception:
            bics.append(np.nan)

    valid_indices = ~np.isnan(bics)
    if not np.any(valid_indices):
        return 3

    valid_k = np.array(list(k_range))[valid_indices]
    valid_bics = np.array(bics)[valid_indices]

    if len(valid_k) == 0:
        return 3
    
    # The best k is the one with the lowest BIC
    best_k = valid_k[np.argmin(valid_bics)]
    return best_k


def analyze_data_for_q1(X, n_clusters, sample_size=50000):
    """
    Analyzes data for Q1 (Fuzzy-based) by checking for ambiguous cluster boundaries.
    Uses an optimized method with sampling and MiniBatchKMeans.
    """
    n_samples = X.shape[0]
    if n_samples < n_clusters * 2:
        return 0.0, "Not enough data for boundary analysis."

    X_sample = X
    if n_samples > sample_size:
        # Use a new random sample each time this function is called
        idx = np.random.choice(n_samples, sample_size, replace=False)
        X_sample = X[idx]
    
    try:
        # 1. Fast temporary clustering
        mbk = MiniBatchKMeans(n_clusters=n_clusters, n_init=3, random_state=None).fit(X_sample)
        labels = mbk.labels_
        
        # Ensure more than one cluster was found
        if len(set(labels)) < 2:
            return 0.0, "MiniBatchKMeans resulted in a single cluster."

        # 2. Lightweight silhouette score calculation on the sample
        sil_scores = silhouette_samples(X_sample, labels)
        
        # 3. Calculate ratio of points on the boundary
        boundary_points = np.sum((sil_scores > -0.1) & (sil_scores < 0.1))
        boundary_ratio = boundary_points / len(X_sample)
        
        reason = f"Boundary point ratio is {boundary_ratio:.2%} (silhouette scores between -0.1 and 0.1)."
        return boundary_ratio, reason
    except Exception as e:
        return 0.0, f"Boundary analysis failed: {e}"

def analyze_data_for_q2(X, sample_size=50000):
    """
    Analyzes data for Q2 (Density-based).
    Estimates the ratio of noise points using a DBSCAN simulation.
    """
    n_samples = X.shape[0]
    if n_samples < 10:
        return 0.0, "Not enough data for density analysis."

    X_sample = X
    if n_samples > sample_size:
        idx = np.random.choice(n_samples, sample_size, replace=False)
        X_sample = X[idx]

    try:
        # Automatically estimate a reasonable 'eps' for DBSCAN
        nn = NearestNeighbors(n_neighbors=5).fit(X_sample)
        distances, _ = nn.kneighbors(X_sample)
        eps_candidate = np.mean(distances[:, -1])

        # Simulate DBSCAN to find noise ratio
        db = DBSCAN(eps=eps_candidate, min_samples=5).fit(X_sample)
        noise_ratio = np.sum(db.labels_ == -1) / len(X_sample)
        
        reason = f"DBSCAN simulation with auto-eps ({eps_candidate:.3f}) found {noise_ratio:.2%} noise."
        return noise_ratio, reason
    except Exception as e:
        return 0.0, f"Density analysis failed: {e}"

def analyze_data_for_q3(X, n_clusters, sample_size=50000):
    """
    Analyzes data for Q3 (Centroid-based).
    Compares BIC of spherical vs. full GMM to check for spherical cluster shapes.
    """
    n_samples = X.shape[0]
    if n_samples < 10:
        return 1.0, "Not enough data for shape analysis."
        
    X_sample = X
    if n_samples > sample_size:
        idx = np.random.choice(n_samples, sample_size, replace=False)
        X_sample = X[idx]

    try:
        gmm_spherical = GaussianMixture(n_components=n_clusters, covariance_type='spherical', random_state=0, n_init=1, reg_covar=1e-5).fit(X_sample)
        gmm_full = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=0, n_init=1, reg_covar=1e-5).fit(X_sample)
        
        bic_spherical = gmm_spherical.bic(X_sample)
        bic_full = gmm_full.bic(X_sample)
        
        # Lower BIC is better. A ratio close to 1 or < 1 means spherical is a good fit.
        # Handle cases where bic_full could be zero or negative to avoid invalid ratios.
        if bic_full <= 0:
             # If full BIC is non-positive, it's an unusual case.
             # If spherical BIC is also non-positive and smaller, it's a good fit.
             # Otherwise, consider it a poor fit for spherical.
             spherical_fit_ratio = 0.9 if bic_spherical < bic_full else 2.0
        else:
             spherical_fit_ratio = bic_spherical / bic_full

        reason = f"Spherical GMM BIC / Full GMM BIC = {spherical_fit_ratio:.3f}. (Ratio <= 1.0 suggests spherical)."
        return spherical_fit_ratio, reason
    except Exception as e:
        return 10.0, f"Shape analysis failed: {e}" # Default to non-spherical

def analyze_data_for_q4(X):
    """
    Analyzes data for Q4 (Model-based vs. Neural Network-based).
    Uses Kurtosis to check similarity to a Gaussian distribution.
    """
    if X.shape[0] < 20:
        return None, "Not enough data for distribution analysis."
        
    try:
        # Kurtosis of a normal distribution is close to 3.
        # We check the average absolute deviation from 3 across all features.
        kurt = kurtosis(X, axis=0, fisher=False)
        avg_kurtosis_deviation = np.nanmean(np.abs(kurt - 3))
        
        reason = f"Avg. Kurtosis deviation from Gaussian (3.0) is {avg_kurtosis_deviation:.3f}."
        return avg_kurtosis_deviation, reason
    except Exception as e:
        return None, f"Distribution analysis failed: {e}"

def recommend_family_from_decision_tree(q1_ratio, q2_ratio, q3_ratio, q4_dev):
    """Recommends a family based on the analyzed metrics."""
    log = []

    # Q1 Check
    log.append(f"Q1 (Fuzzy): Boundary point ratio is {q1_ratio:.2%}.")
    if q1_ratio > 0.2: # If more than 20% of points are on a boundary
        log.append("  -> DECISION: Cluster boundaries are ambiguous. Recommending Fuzzy Clustering.")
        return "4. Fuzzy Clustering", log

    # Q2 Check
    log.append(f"Q2 (Density): Noise ratio is {q2_ratio:.2%}.")
    if q2_ratio > 0.1: # If more than 10% of data is considered noise
        log.append("  -> DECISION: Noise separation appears important. Recommending Density-based.")
        return "2. Density-based", log

    # Q3 Check
    log.append(f"Q3 (Centroid): Spherical fit score is {q3_ratio:.3f}.")
    if q3_ratio <= 1.05: # If spherical model is almost as good or better than full model
        log.append("  -> DECISION: Data fits spherical clusters well. Recommending Centroid-based.")
        return "1. Centroid-based", log

    # Q4 Check
    log.append(f"Q4 (Model/NN): Kurtosis deviation from Gaussian is {q4_dev:.3f}.")
    if q4_dev is not None and q4_dev <= 1.5: # If distribution is reasonably close to Gaussian
        log.append("  -> DECISION: Data appears to follow a statistical distribution. Recommending Model-based.")
        return "3. Model-based", log
    else:
        log.append("  -> DECISION: Data does not follow a simple distribution. Recommending Neural Network-based for complex pattern detection.")
        return "5. Neural Network-based", log

def calculate_family_fit_scores(q1_ratio, q2_ratio, q3_ratio, q4_dev):
    """Calculates a normalized 'fit score' (0-1) for each family based on metrics."""
    scores = {}
    
    # Fuzzy score: Directly proportional to boundary ratio. Capped at 1.0.
    scores['4. Fuzzy Clustering'] = min(1.0, q1_ratio / 0.3) # Normalize: 30% boundary = max score
    
    # Density score: Directly proportional to noise ratio. Capped at 1.0.
    scores['2. Density-based'] = min(1.0, q2_ratio / 0.2) # Normalize: 20% noise = max score
    
    # Centroid score: Inversely proportional to spherical fit ratio. Score is high when ratio is <= 1.
    # A ratio of 1 gives a score of 1. A ratio of 2 gives a score of 0.
    scores['1. Centroid-based'] = max(0, 1 - (q3_ratio - 1))
    
    # Model-based score: Inversely proportional to kurtosis deviation.
    # A deviation of 0 gives a score of 1. A deviation of 3 gives a score of 0.
    if q4_dev is not None:
        scores['3. Model-based'] = max(0, 1 - (q4_dev / 3.0))
        # Neural-Net score is opposite to model-based
        scores['5. Neural Network-based'] = min(1.0, q4_dev / 3.0)
    else:
        scores['3. Model-based'] = 0.0
        scores['5. Neural Network-based'] = 0.0
        
    return scores


def print_and_save_summary_report(run_results, est_k_surrogate, est_k_bic, file_type, file_number):
    """
    Calculates average fit scores, prints a summary, and saves the detailed log.
    The final recommendation is now based on the highest average fit score.
    """
    
    # --- Calculate average fit scores from all runs ---
    df_results = pd.DataFrame(run_results)
    avg_scores = {}
    score_cols = [col for col in df_results.columns if 'fit_score' in col]
    for col in score_cols:
        family_name = col.replace('_fit_score', '')
        avg_scores[family_name] = df_results[col].mean()
        
    # Determine the best family based on the highest average fit score
    if not avg_scores:
        most_common_rec = "Unable to determine"
    else:
        most_common_rec = max(avg_scores, key=avg_scores.get)

    # --- Print Terminal Summary ---
    print("\n" + "="*70)
    print(f"Final Recommendation Summary ({len(run_results)} runs)")
    print("="*70)

    print("\n[Estimated Optimal k]")
    print(f"- Based on Surrogate Score (spherical assumption): {est_k_surrogate}")
    print(f"- Based on GMM BIC (complex shape assumption): {est_k_bic}")

    print("\n[Average Family Fit Score (0-1 scale)]")
    sorted_scores = sorted(avg_scores.items(), key=lambda item: item[1], reverse=True)
    for family, score in sorted_scores:
        print(f"- {family}: {score:.3f}")
        
    print(f"\n[Overall Recommendation]")
    print(f"--> Based on the highest average fit score, the recommended family is: **{most_common_rec}**")

    # --- Save Detailed CSV Log ---
    # Reorder columns for clarity, now including fit scores
    cols_start = ['run', 'q1_ratio', 'q2_ratio', 'q3_ratio', 'q4_dev']
    # Dynamically get score columns in the same order as they are printed
    sorted_score_cols = [f"{fam}_fit_score" for fam, score in sorted_scores]
    cols = cols_start + sorted_score_cols
    
    # Ensure all columns exist before reordering
    final_cols = [c for c in cols if c in df_results.columns]
    df_results = df_results[final_cols]

    output_dir = f"../Dataset/recommend/"
    os.makedirs(output_dir, exist_ok=True)
    file_type_dir = os.path.join(output_dir, file_type)
    os.makedirs(file_type_dir, exist_ok=True)
    save_path = os.path.join(file_type_dir, f"{file_type}_{file_number}_recommendation_3_fit_score_report.csv")
    df_results.to_csv(save_path, index=False)
    print(f"\nSaved detailed multi-run report to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Recommends a clustering family based on data characteristics")
    parser.add_argument('--file_type', type=str, default="MiraiBotnet")
    parser.add_argument('--file_number', type=int, default=1)
    args = parser.parse_args()

    file_type = args.file_type
    file_number = args.file_number

    total_start = time.time()

    # --- Data Loading and Preprocessing (from recommend_clustering_2.py) ---
    print("[Step 1] Loading data...")
    file_path, _ = file_path_line_nonnumber(file_type, file_number)
    data = file_cut(file_type, file_path, 'all')

    data.columns = data.columns.str.strip()

    print("[Step 2] Applying labels...")
    data = apply_labeling_logic(data, args.file_type)

    print("[Step 3] Preprocessing features...")
    data = time_scalar_transfer(data, file_type)
    _, _, _, data_list = choose_heterogeneous_method(data, file_type, het_method="Interval_inverse", regul='N')
    
    X_parts = [df for df in data_list if not df.empty]
    if not X_parts:
        print("[Error] No valid data parts found after preprocessing.")
        return
    X = pd.concat(X_parts, axis=1)

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    if X.isnull().values.any():
        original_rows = X.shape[0]
        X.dropna(inplace=True)
        print(f"[Info] Removed {original_rows - X.shape[0]} rows containing NaN values.")

    if X.empty:
        print("[Error] No data remaining after cleaning. Cannot proceed.")
        return

    print("[Step 4] Applying Scaling and PCA...")
    X_final = X
    if apply_minmax_scaling_and_save_scalers and pca_func:
        X_scaled, _ = apply_minmax_scaling_and_save_scalers(X, file_type, 1, "Interval_inverse")
        if file_type not in ['DARPA98', 'DARPA', 'CICModbus23', 'CICModbus']:
            X_final = pca_func(X_scaled)
            if not isinstance(X_final, pd.DataFrame):
                X_final = pd.DataFrame(X_final, index=X_scaled.index[:len(X_final)])
        else:
            X_final = X_scaled
    
    print(f"\n[Info] Data processed successfully. Final shape for analysis: {X_final.shape}")
    
    # --- New Core Logic: Multi-run Analysis ---
    X_final_np = X_final.values if isinstance(X_final, pd.DataFrame) else X_final

    # --- Create a sample of 'known normal' indices for surrogate score calculation ---
    # This is done once and passed to the k estimation.
    print("\n[Step 5] Generating a sample of 'known normal' indices for analysis...")
    known_normal_idx_for_k = get_known_normal_indices(X_final_np)
    print(f" -> Found {len(known_normal_idx_for_k)} pseudo 'known normal' samples.")

    # --- Estimate optimal number of clusters before starting runs ---
    print("\n[Step 6] Estimating optimal number of clusters (k) using dual methods...")
    est_k_surrogate = estimate_n_clusters_surrogate(X_final_np, known_normal_idx_for_k)
    est_k_bic = estimate_n_clusters_bic(X_final_np)
    print(f" -> Surrogate Score based k (spherical assumption) = {est_k_surrogate}")
    print(f" -> GMM BIC based k (complex shape assumption) = {est_k_bic}")
    
    # Decide which k to use for the main analysis loop.
    # We will use the BIC-based k for analyses involving GMM (q3)
    # and the surrogate-based k for KMeans-based analyses (q1).
    k_for_q1 = est_k_surrogate
    k_for_q3 = est_k_bic

    N_RUNS = 15
    all_run_results = []

    print("\n" + "="*70)
    print(f"Starting Analysis with {N_RUNS} Independent Runs...")
    print("="*70)

    for i in range(N_RUNS):
        print(f"\n--- Run {i+1}/{N_RUNS} ---")
        
        q1_ratio, q1_reason = analyze_data_for_q1(X_final_np, n_clusters=k_for_q1)
        print(f" Q1 Metric (k={k_for_q1}): {q1_reason}")

        q2_ratio, q2_reason = analyze_data_for_q2(X_final_np)
        print(f" Q2 Metric: {q2_reason}")

        q3_ratio, q3_reason = analyze_data_for_q3(X_final_np, n_clusters=k_for_q3)
        print(f" Q3 Metric (k={k_for_q3}): {q3_reason}")

        q4_dev, q4_reason = analyze_data_for_q4(X_final_np)
        print(f" Q4 Metric: {q4_reason}")
        
        # REMOVED: Decision tree logic
        # NEW: Directly calculate and store fit scores
        fit_scores = calculate_family_fit_scores(q1_ratio, q2_ratio, q3_ratio, q4_dev)
        
        run_data = {
            'run': i + 1,
            'q1_ratio': q1_ratio,
            'q2_ratio': q2_ratio,
            'q3_ratio': q3_ratio,
            'q4_dev': q4_dev
        }
        # Add fit scores to the dictionary with a clear naming convention
        for fam, score in fit_scores.items():
            run_data[f"{fam}_fit_score"] = score
        
        all_run_results.append(run_data)


    # --- Print and Save Final Summary Report ---
    print_and_save_summary_report(all_run_results, est_k_surrogate, est_k_bic, file_type, file_number)

    print(f"\nTotal script time: {time.time() - total_start:.2f} seconds.")


if __name__ == '__main__':
    main()
