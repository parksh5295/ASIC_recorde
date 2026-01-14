import csv
import os
from datetime import datetime
import pandas as pd

def time_save_csv_VL(file_type, file_number, clustering_algorithm, timing_info):
    """
    Save timing information for each step and the total execution time to a CSV file.

    Parameters:
    - file_type (str): Dataset type (e.g., MiraiBotnet)
    - file_number (int): File index or part number
    - clustering_algorithm (str): Name of clustering algorithm used
    - timing_info (dict): Dictionary containing time taken per step
    - save_dir (str): Directory to save timing CSVs
    """

    save_dir = f"../Dataset_Paral/time_log/virtual_labeling/{file_type}"

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Filename format: [filetype]_[filenumber]_[clustering]_[timestamp].csv
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{file_type}_{file_number}_{clustering_algorithm}_{timestamp}.csv"
    filepath = os.path.join(save_dir, filename)

    # Write CSV file
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Step', 'Time_Seconds'])
        for step, duration in timing_info.items():
            writer.writerow([step, round(duration, 4)])

    print(f"\n Timing log saved to: {filepath}")

    return


def time_save_csv_VL_eval(file_type, file_number, clustering_algorithm, timing_info):
    """
    Save timing information for Data_Labeling_Evaluate_Thresholds.py to a separate directory.
    """

    save_dir = f"../Dataset_Paral/time_log/virtual_labeling_eval/{file_type}"

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Filename format: [filetype]_[filenumber]_[clustering]_[timestamp].csv
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{file_type}_{file_number}_{clustering_algorithm}_{timestamp}.csv"
    filepath = os.path.join(save_dir, filename)

    # Write CSV file
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Step', 'Time_Seconds'])
        for step, duration in timing_info.items():
            writer.writerow([step, round(duration, 4)])

    print(f"\n Timing log for EVAL saved to: {filepath}")

    return


def time_save_csv_CS(file_type, file_number, Association_mathod, timing_info, best_confidence=None, min_support=None):
    """
    Save timing information for each step, total execution time, best_confidence, and min_support to a CSV file.
    """

    save_dir = f"../Dataset_Paral/time_log/condition_assocation/{file_type}"

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Filename format: [filetype]_[filenumber]_[clustering]_[timestamp].csv
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{file_type}_{file_number}_{Association_mathod}_{timestamp}.csv"
    filepath = os.path.join(save_dir, filename)

    # Write CSV file
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Step', 'Time_Seconds'])
        for step, duration in timing_info.items():
            writer.writerow([step, round(duration, 4)])
        
        # Add best_confidence and min_support if available
        if best_confidence is not None:
            writer.writerow(['Best_Confidence', best_confidence])
        if min_support is not None:
            writer.writerow(['Min_Support', min_support])

    print(f"\n Timing log saved to: {filepath}")

    return


# Save Validation signature time
def time_save_csv_VS(file_type, file_number, Association_mathod, timing_info):
    save_dir = f"../Dataset_Paral/time_log/validation_signature/{file_type}"

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Filename format: [filetype]_[filenumber]_[clustering]_[timestamp].csv
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{file_type}_{file_number}_{Association_mathod}_{timestamp}.csv"
    filepath = os.path.join(save_dir, filename)

    # Write CSV file
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Step', 'Time_Seconds'])
        for step, duration in timing_info.items():
            writer.writerow([step, round(duration, 4)])

    print(f"\n Timing log saved to: {filepath}")

    return


def time_save_csv_CS_ex(file_type, file_number, Association_mathod, timing_info, best_confidence=None, min_support=None):
    """
    Save timing information for Main_Association_Rule_ex.py to a separate directory.
    """

    save_dir = f"../Dataset_Paral/time_log/condition_assocation_ex/{file_type}"

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Filename format: [filetype]_[filenumber]_[association_method]_[timestamp].csv
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{file_type}_{file_number}_{Association_mathod}_{timestamp}.csv"
    filepath = os.path.join(save_dir, filename)

    # Write CSV file
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Step', 'Time_Seconds'])
        for step, duration in timing_info.items():
            writer.writerow([step, round(duration, 4)])
        
        # Add best_confidence and min_support if available
        if best_confidence is not None:
            writer.writerow(['Best_Confidence', best_confidence])
        if min_support is not None:
            writer.writerow(['Min_Support', min_support])

    print(f"\n Timing log for EX saved to: {filepath}")

    return


def time_save_csv_VL_ex(file_type, file_number, clustering_algorithm, timing_info):
    """Save timing information to Dataset_ex folder"""
    save_dir = f"../Dataset_ex/time_log/virtual_labeling_ex/{file_type}"
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{file_type}_{file_number}_{clustering_algorithm}_{timestamp}.csv"
    filepath = os.path.join(save_dir, filename)

    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Step', 'Time_Seconds'])
        for step, duration in timing_info.items():
            writer.writerow([step, round(duration, 4)])

    print(f"\n Timing log saved to: {filepath}")
    return


def time_save_csv_mapping_conditions(timing_info, file_type, file_number, association_method, 
                                    min_support, min_confidence, condition_values):
    """
    Save time logs for Evaluate_Mapping_Conditions.py to CSV.
    
    Args:
        timing_info: Dictionary containing timing information for each condition value
        file_type: Type of dataset file
        file_number: Number of dataset file
        association_method: Association rule mining method used
        min_support: Minimum support threshold
        min_confidence: Minimum confidence threshold
        condition_values: List of n_splits values tested
    """
    # Create directory if it doesn't exist
    output_dir = "../Dataset_Paral/time_log/mapping_conditions_eval/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with parameters
    filename = f"time_log_mapping_conditions_{file_type}_{file_number}_{association_method}_ms{min_support}_mc{min_confidence}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Prepare data for CSV
    data = []
    for condition_value, timing in timing_info.items():
        if isinstance(timing, dict):
            # Extract total time and individual stage times
            total_time = timing.get('total_time', 0)
            data.append({
                'n_splits': condition_value,
                'total_time': total_time,
                'stage1_time': timing.get('stage1_time', 0),
                'stage2_time': timing.get('stage2_time', 0),
                'stage3_time': timing.get('stage3_time', 0),
                'stage4_time': timing.get('stage4_time', 0)
            })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    
    print(f"Time log saved to: {filepath}")
    return filepath
