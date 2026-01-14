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

def estimate_n_clusters_robust(X, known_normal_idx, sample_size=50000, max_k=200):
    """
    Estimates the optimal number of clusters using the Elbow method on the
    project's standard Surrogate Score.
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

    print(f"  > Testing k from {min(k_range)} to {max(k_range)} using Surrogate Score...")
    # Add tqdm progress bar here
    for k in tqdm(k_range, desc="Estimating k"):
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
        spherical_fit_ratio = bic_spherical / bic_full if bic_full != 0 else 1.0
        
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

def print_and_save_results(recommended_family, families, metrics, log, file_type, file_number):
    """Prints the recommendation and saves it to a CSV."""
    print("\n" + "="*70)
    print("Clustering Recommendation Report")
    print("="*70)
    
    print("\n[Data Characteristics]")
    for key, value in metrics.items():
        print(f"- {key.replace('_', ' ').title()}: {value:.3f}")

    print("\n[Decision Path]")
    for line in log:
        print(line)
        
    print("\n[Primary Recommendation]")
    print(f"--> Based on the analysis, the recommended family is: **{recommended_family}**")
    details = families[recommended_family]
    print(f"    - Core Idea: {details['core_idea']}")
    print(f"    - Algorithms: {', '.join(details['algorithms'])}")
    print(f"    - Use When: {details['use_when']}")

    print("\n[Alternative Consideration - Requirement-based]")
    fuzzy_details = families["4. Fuzzy Clustering"]
    print(f"--> If your analysis requires a data point to belong to MULTIPLE clusters (e.g., membership scores), consider: **4. Fuzzy Clustering**")
    print(f"    - Algorithms: {', '.join(fuzzy_details['algorithms'])}")

    # --- Prepare for CSV ---
    recommended_algos = families[recommended_family]['algorithms']
    data_for_csv = []
    for algo in recommended_algos:
        data_for_csv.append({
            "Recommendation_Status": "Primary",
            "Family": recommended_family,
            "Algorithm": algo,
            "Reasoning": " | ".join(log)
        })
    
    fuzzy_algos = families["4. Fuzzy Clustering"]['algorithms']
    for algo in fuzzy_algos:
        data_for_csv.append({
            "Recommendation_Status": "Alternative (Requirement-based)",
            "Family": "4. Fuzzy Clustering",
            "Algorithm": algo,
            "Reasoning": "Required if data points need to have partial membership in multiple clusters."
        })
        
    df = pd.DataFrame(data_for_csv)
    
    # --- Save CSV ---
    output_dir = f"../Dataset/recommend/"
    os.makedirs(output_dir, exist_ok=True)
    file_type_dir = os.path.join(output_dir, file_type)
    os.makedirs(file_type_dir, exist_ok=True)
    save_path = os.path.join(file_type_dir, f"{file_type}_{file_number}_recommendation_3.csv")
    df.to_csv(save_path, index=False)
    print(f"\nSaved detailed recommendation to: {save_path}")

def print_and_save_summary_report(run_results, file_type, file_number):
    """Prints a summary of all runs and saves the detailed log to CSV."""
    
    # --- Print Terminal Summary ---
    print("\n" + "="*70)
    print(f"Final Recommendation Summary ({len(run_results)} runs)")
    print("="*70)

    recommendations = [r['recommendation'] for r in run_results]
    counts = Counter(recommendations)
    
    print("\n[Recommendation Frequency]")
    for family, count in counts.most_common():
        percentage = (count / len(run_results)) * 100
        print(f"- {family}: {count} times ({percentage:.1f}%)")
    
    most_common_rec = counts.most_common(1)[0][0]
    print(f"\n[Overall Recommendation]")
    print(f"--> The most frequent recommendation is: **{most_common_rec}**")

    # --- Save Detailed CSV Log ---
    df = pd.DataFrame(run_results)
    # Reorder columns for clarity
    cols = ['run', 'recommendation', 'q1_ratio', 'q2_ratio', 'q3_ratio', 'q4_dev', 'decision_log']
    df = df[cols]

    output_dir = f"../Dataset/recommend/"
    os.makedirs(output_dir, exist_ok=True)
    file_type_dir = os.path.join(output_dir, file_type)
    os.makedirs(file_type_dir, exist_ok=True)
    save_path = os.path.join(file_type_dir, f"{file_type}_{file_number}_recommendation_3_report.csv")
    df.to_csv(save_path, index=False)
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
    print("\n[Step 6] Estimating optimal number of clusters (k) using robust Surrogate Score method...")
    est_k = estimate_n_clusters_robust(X_final_np, known_normal_idx_for_k)
    print(f" -> Data-driven estimate for k = {est_k}")

    N_RUNS = 15
    all_run_results = []

    print("\n" + "="*70)
    print(f"Starting Analysis with {N_RUNS} Independent Runs")
    print("="*70)

    for i in range(N_RUNS):
        print(f"\n--- Run {i+1}/{N_RUNS} ---")
        
        # Each run performs its own sampling and analysis, using the estimated k
        q1_ratio, q1_reason = analyze_data_for_q1(X_final_np, n_clusters=est_k)
        print(f" Q1 Metric: {q1_reason}")

        q2_ratio, q2_reason = analyze_data_for_q2(X_final_np)
        print(f" Q2 Metric: {q2_reason}")

        q3_ratio, q3_reason = analyze_data_for_q3(X_final_np, n_clusters=est_k)
        print(f" Q3 Metric: {q3_reason}")

        q4_dev, q4_reason = analyze_data_for_q4(X_final_np)
        print(f" Q4 Metric: {q4_reason}")
        
        recommendation, log = recommend_family_from_decision_tree(
            q1_ratio, q2_ratio, q3_ratio, q4_dev
        )
        print(f" -> Run {i+1} Recommendation: {recommendation}")
        
        all_run_results.append({
            'run': i + 1,
            'recommendation': recommendation,
            'q1_ratio': q1_ratio,
            'q2_ratio': q2_ratio,
            'q3_ratio': q3_ratio,
            'q4_dev': q4_dev,
            'decision_log': " | ".join(log)
        })

    # --- Print and Save Final Summary Report ---
    print_and_save_summary_report(all_run_results, file_type, file_number)

    print(f"\nTotal script time: {time.time() - total_start:.2f} seconds.")


if __name__ == '__main__':
    main()
