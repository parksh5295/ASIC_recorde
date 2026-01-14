import pandas as pd
import os
import json
import numpy as np
import re
from datetime import datetime


def ensure_directory_exists(filepath):
    """If the directory of the specified file path does not exist, create it."""
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        print(f"Creating directory: {directory}")
        os.makedirs(directory)

# Helper function to convert numpy types
def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist()) # Convert arrays to lists first
    elif pd.isna(obj): # Handle pandas NA/NaN specifically for JSON
        return None # Represent NaN as null in JSON
    return obj

def save_validation_results(file_type, file_number, association_rule, association_metric, signature_ea,
                            recall_before, recall_after,
                            precision_before, precision_after,
                            num_initial_signatures, num_final_signatures, num_fp_flagged,
                            num_fake_signatures_generated, num_fake_signatures_detected_as_fp,
                            detected_anomalous_before, total_true_anomalies, detected_anomalous_after,
                            tp_before, fp_before, tp_after, fp_after,
                            fp_eval_method,
                            basic_eval=None, fp_results=None, overfit_results=None, filtered_eval=None):
    """
    Save all evaluation results, including recall, precision metrics and filtered eval, in a single CSV/JSON file
    """
    # Construct a more descriptive filename
    # Sanitize association_rule for filename
    safe_association_method = re.sub(r'[^\\w\\-_.]', '_', str(association_rule))
    base_filename = f"{file_type}_{file_number}_{safe_association_method}_{association_metric}_ea{signature_ea}_{fp_eval_method}_validation"
    
    save_path_dir = f"../Dataset_Paral/validation_results/{file_type}/"
    ensure_directory_exists(save_path_dir)

    # Main results dictionary
    validation_summary_data = {
        'FileType': file_type,
        'FileNumber': file_number,
        'AssociationMethod': association_rule,
        'AssociationMetric': association_metric,
        'SignatureEA': signature_ea,
        'FPEvalMethod': fp_eval_method,
        'RecallBeforeFPRemoval': recall_before,
        'DetectedAnomalousBefore': detected_anomalous_before,
        'TotalTrueAnomalies': total_true_anomalies,
        'PrecisionBeforeFPRemoval': precision_before,
        'TP_Before': tp_before,
        'FP_Before': fp_before,
        'NumInitialSignatures': num_initial_signatures,
        'NumFakeSignaturesGenerated': num_fake_signatures_generated,
        'RecallAfterFPRemoval': recall_after,
        'DetectedAnomalousAfter': detected_anomalous_after,
        'PrecisionAfterFPRemoval': precision_after,
        'TP_After': tp_after,
        'FP_After': fp_after,
        'NumFinalSignatures': num_final_signatures,
        'NumSignaturesFlaggedAsFP': num_fp_flagged,
        'NumFakeSignaturesDetectedAsFP': num_fake_signatures_detected_as_fp
    }

    # Detailed results (can be large, good for JSON)
    detailed_results_data = {
        'Basic_Evaluation_Details': basic_eval.to_dict('records') if isinstance(basic_eval, pd.DataFrame) else basic_eval,
        'False_Positive_Analysis_Details': fp_results.to_dict('records') if isinstance(fp_results, pd.DataFrame) else fp_results,
        'Overfitting_Analysis_Details': overfit_results,
        'Filtered_Signature_Evaluation_Details': filtered_eval.to_dict('records') if isinstance(filtered_eval, pd.DataFrame) else filtered_eval
    }

    # --- Convert numpy types before saving ---
    validation_summary_serializable = convert_numpy_types(validation_summary_data)
    detailed_results_serializable = convert_numpy_types(detailed_results_data)
    # -----------------------------------------

    # Save summary to CSV
    summary_csv_path = os.path.join(save_path_dir, f"{base_filename}_summary.csv")
    try:
        # Ensure all values in summary are scalar or simple lists for CSV
        # For any list-like values, convert to string representation
        csv_ready_summary = {}
        for k, v in validation_summary_serializable.items():
            if isinstance(v, list) or isinstance(v, dict) or isinstance(v, tuple):
                csv_ready_summary[k] = str(v)
            else:
                csv_ready_summary[k] = v
        
        summary_df = pd.DataFrame([csv_ready_summary]) # Create a single row DataFrame
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Validation summary results saved to CSV: {summary_csv_path}")
    except Exception as csv_e:
        print(f"Warning: Could not save validation summary to CSV: {csv_e}")

    # Save detailed results to JSON
    detailed_json_path = os.path.join(save_path_dir, f"{base_filename}_details.json")
    try:
        # Combine summary and details for a single comprehensive JSON file
        comprehensive_results = {
            'Summary': validation_summary_serializable,
            'Details': detailed_results_serializable,
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(detailed_json_path, 'w') as f:
            json.dump(comprehensive_results, f, indent=4)
        print(f"Comprehensive validation results (summary & details) saved as JSON: {detailed_json_path}")
    except Exception as json_e:
        print(f"Error during saving detailed results to JSON: {json_e}")
