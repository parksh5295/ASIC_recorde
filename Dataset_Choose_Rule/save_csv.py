# for save Clustering, Association row to csv

import os
import pandas as pd
from datetime import datetime


# Functions for creating folders if they don't exist
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def csv_compare_clustering(file_type, clusterint_method, file_number, data, GMM_type=None, optimal_cni_threshold=None):
    row_compare_df = data[['cluster', 'adjusted_cluster', 'label']].copy() # Use .copy() to avoid SettingWithCopyWarning
    if optimal_cni_threshold is not None:
        row_compare_df['Optimal_CNI_Threshold'] = optimal_cni_threshold
    
    save_path = f"../Dataset_Paral/save_dataset/{file_type}/"
    ensure_directory_exists(save_path)  # Verify and create the folder
    
    if clusterint_method == "GMM":
        file_path = f"{save_path}{file_type}_{clusterint_method}_{file_number}_clustering_Compare_gmm{GMM_type}.csv"
    else:
        file_path = f"{save_path}{file_type}_{clusterint_method}_{file_number}_clustering_Compare.csv"
    row_compare_df.to_csv(file_path, index=False)
    print(f"[INFO SaveCSV] Clustering comparison data saved to: {os.path.abspath(file_path)}")
    
    return row_compare_df

def csv_compare_matrix_clustering(file_type, file_number, clusterint_method, metrics_original, metrics_adjusted, GMM_type, optimal_cni_threshold=None):
    # Ensure metrics_original and metrics_adjusted are dictionaries before creating DataFrame
    # Create copies to modify them before DataFrame creation if needed
    metrics_original_with_thresh = metrics_original.copy() if metrics_original else {}
    metrics_adjusted_with_thresh = metrics_adjusted.copy() if metrics_adjusted else {}

    if optimal_cni_threshold is not None:
        metrics_original_with_thresh['Optimal_CNI_Threshold'] = optimal_cni_threshold
        metrics_adjusted_with_thresh['Optimal_CNI_Threshold'] = optimal_cni_threshold

    metrics_df = pd.DataFrame([metrics_original_with_thresh, metrics_adjusted_with_thresh], index=["Original", "Adjusted"])
    
    save_path = f"../Dataset_Paral/save_dataset/{file_type}/"
    ensure_directory_exists(save_path)  # Verify and create the folder
    
    if clusterint_method == "GMM":
        file_path = f"{save_path}{file_type}_{clusterint_method}_{file_number}_clustering_Compare_Metrics_gmm{GMM_type}.csv"
    else:
        file_path = f"{save_path}{file_type}_{clusterint_method}_{file_number}_clustering_Compare_Metrics.csv"
    metrics_df.to_csv(file_path, index=True)
    print(f"[INFO SaveCSV] Clustering metrics data saved to: {os.path.abspath(file_path)}")
    
    return metrics_df


#def csv_association(file_type, file_number, association_rule, association_result, association_metric, signature_ea):
    #df = pd.DataFrame([association_result])
def csv_association(file_type, file_number, Association_mathod, association_result, association_metric, signature_ea, loop_limit=None, signature_count=None):
    # Check OS and set path
    if os.name == 'nt':
        path = f"../Dataset_Paral/signature/{file_type}/"
    else:
        path = f"../Dataset_Paral/signature/{file_type}/"

    # Create the directory if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)

    # Define the full file path
    file_path = f"{path}{file_type}_{file_number}_{Association_mathod}_{association_metric}_{signature_ea}_association_result.csv"

    # Convert the list of signature dictionaries to a DataFrame
    df = pd.DataFrame(association_result['Verified_Signatures'])

    # --- NEW: Write summary as a commented header to the CSV file ---
    with open(file_path, 'w', newline='') as f:
        if loop_limit is not None and signature_count is not None:
            header = (
                f"# Automatically Generated Signature Report\n"
                f"# Generation Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"# Loop Limit Level for this Dataset: {loop_limit}\n"
                f"# Final Signature Count: {signature_count}\n\n"
            )
            f.write(header)
        
        # Write the DataFrame to the same file, after the header
        df.to_csv(f, index=False)

    print(f"Result with summary saved to: {file_path}")

    return


def csv_compare_clustering_ex(file_type, clusterint_method, file_number, data, GMM_type=None, optimal_cni_threshold=None):
    """Save clustering comparison data to Dataset_ex folder"""
    row_compare_df = data[['cluster', 'adjusted_cluster', 'label']].copy()
    if optimal_cni_threshold is not None:
        row_compare_df['Optimal_CNI_Threshold'] = optimal_cni_threshold
    
    save_path = f"../Dataset_ex/Data_Label/{file_type}/"
    ensure_directory_exists(save_path)
    
    if clusterint_method == "GMM":
        file_path = f"{save_path}{file_type}_{clusterint_method}_{file_number}_clustering_Compare_gmm{GMM_type}.csv"
    else:
        file_path = f"{save_path}{file_type}_{clusterint_method}_{file_number}_clustering_Compare.csv"
    row_compare_df.to_csv(file_path, index=False)
    print(f"[INFO SaveCSV] Clustering comparison data saved to: {os.path.abspath(file_path)}")
    
    return row_compare_df


def csv_compare_matrix_clustering_ex(file_type, file_number, clusterint_method, metrics_original, metrics_adjusted, GMM_type, optimal_cni_threshold=None):
    """Save clustering metrics data to Dataset_ex folder"""
    metrics_original_with_thresh = metrics_original.copy() if metrics_original else {}
    metrics_adjusted_with_thresh = metrics_adjusted.copy() if metrics_adjusted else {}

    if optimal_cni_threshold is not None:
        metrics_original_with_thresh['Optimal_CNI_Threshold'] = optimal_cni_threshold
        metrics_adjusted_with_thresh['Optimal_CNI_Threshold'] = optimal_cni_threshold

    metrics_df = pd.DataFrame([metrics_original_with_thresh, metrics_adjusted_with_thresh], index=["Original", "Adjusted"])
    
    save_path = f"../Dataset_ex/Data_Label/{file_type}/"
    ensure_directory_exists(save_path)
    
    if clusterint_method == "GMM":
        file_path = f"{save_path}{file_type}_{clusterint_method}_{file_number}_clustering_Compare_Metrics_gmm{GMM_type}.csv"
    else:
        file_path = f"{save_path}{file_type}_{clusterint_method}_{file_number}_clustering_Compare_Metrics.csv"
    metrics_df.to_csv(file_path, index=True)
    print(f"[INFO SaveCSV] Clustering metrics data saved to: {os.path.abspath(file_path)}")
    
    return metrics_df
    