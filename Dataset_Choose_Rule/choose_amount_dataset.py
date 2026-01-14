# Agendas for importing partial or full files in a dataset
# Output is data

import pandas as pd
import random
from sklearn.model_selection import train_test_split
# from Dataset_Choose_Rule.CICIDS2017_csv_selector import select_csv_file # Commented out
from Dataset_Choose_Rule.dtype_optimize import infer_dtypes_safely, _post_process_specific_datasets


def file_path_line_nonnumber(file_type, file_number=1): # file_number is not used, but insert to prevent errors from occurring
    if file_type == 'MiraiBotnet':
        file_path = "../Dataset/load_dataset/MiraiBotnet/output-dataset_ESSlab.csv"
    elif file_type in ['ARP', 'MitM', 'Kitsune']:
        file_path = "../Dataset/load_dataset/ARP_MitM_Kitsune/ARP_MitM_dataset.csv/ARP_MitM_dataset_final.csv"
    elif file_type in ['CICIDS2017', 'CICIDS']:
        # file_path, file_number =  select_csv_file() # Original line
        # file_path = "~/asic/Dataset/load_dataset/CICIDS2017/CICIDS_all.csv" # Use unified CSV
        file_path = "../Dataset/load_dataset/CICIDS2017/CICIDS_all.csv"
        file_number = 1 # Default file_number, as select_csv_file used to return it
    elif file_type == 'netML' :
        file_path = "../Dataset/load_dataset/netML/netML_dataset.csv"
    elif file_type in ['NSL-KDD', 'NSL_KDD']:
        file_path = "../Dataset/load_dataset/NSL-KDD/train/train_payload.csv"
    elif file_type in ['DARPA', 'DARPA98']:
        file_path = "../Dataset/load_dataset/DARPA98/train/DARPA98.csv"
    elif file_type in ['CICModbus23', 'CICModbus']:
        file_path = "../Dataset/load_dataset/CICModbus23/CICModbus23_total.csv"
    elif file_type in ['IoTID20', 'IoTID']:
        file_path = "../Dataset/load_dataset/IoTID20/IoTID20.csv"
    elif file_type in ['CICIoT', 'CICIoT2023']:
        file_path = "../Dataset/load_dataset/CICIoT2023/training-flow.csv"
    else:
        print("No file information yet, please double-check the file type or provide new data!")
        file_path_line_nonnumber(file_type)
    return file_path, file_number

def file_path_line_withnumber(file_type, file_number=1):
    return # file_path

# After selecting the file path
# Functions for getting only part of a file as data
def file_cut(file_type, file_path, cut_type='random'):
    inferred_dtypes = infer_dtypes_safely(file_type, file_path)
    # Convert integer dtypes to nullable Int32 to allow NaNs with lower memory
    inferred_dtypes = {
        col: ('Int32' if isinstance(dt, str) and dt.startswith('int') else dt)
        for col, dt in inferred_dtypes.items()
    }

    df = None  # Initialize df to avoid UnboundLocalError

    if cut_type == 'random':
        # Get the total number of rows (excluding headers)
        total_rows = sum(1 for _ in open(file_path)) - 1  # excluding headers

        # Select row numbers to randomly sample
        # num_rows_to_sample = int(input("Enter the desired number of rows of data: "))
        '''
        if file_type in ['CICIoT2023']:
            num_rows_to_sample = 200000
        else:
            num_rows_to_sample = 10000
        '''
        num_rows_to_sample = 10000
        sampled_rows = sorted(random.sample(range(1, total_rows + 1), num_rows_to_sample))

        # Read only selected rows (but keep headers)
        df = pd.read_csv(
            file_path,
            dtype=inferred_dtypes,
            skiprows=lambda x: x > 0 and x not in sampled_rows
        )
        df = _post_process_specific_datasets(df, file_type)

    elif cut_type in ['in order', 'In order', 'In Order']:    # from n~m row
        n = int(input("Enter the row number to start with: "))  # Row number to start with (1-based index, i.e., first data is 1)
        m = int(input("Enter the row number to end with: "))  # Row number to end

        df = pd.read_csv(
            file_path,
            dtype=inferred_dtypes,
            skiprows=lambda x: x > 0 and x < n,
            nrows=m - n + 1
        )
        df = _post_process_specific_datasets(df, file_type)

    elif cut_type in ['all', 'All']:
        df = pd.read_csv(file_path, dtype=inferred_dtypes)
        df = _post_process_specific_datasets(df, file_type)

    return df   # return data


def file_cut_GEN(file_type, file_path, cut_type='all', header=0):
    """
    Loads a dataset from a CSV file, tailored for 'all' cut_type and explicit header handling.
    """
    # Include _infer_dtypes_safely and _post_process_specific_datasets because they are used in the ‘all’ case in the file_cut function.
    inferred_dtypes = infer_dtypes_safely(file_type, file_path)
    # Convert integer dtypes to nullable Int32 to allow NaNs with lower memory
    inferred_dtypes = {
        col: ('Int32' if isinstance(dt, str) and dt.startswith('int') else dt)
        for col, dt in inferred_dtypes.items()
    }
    df = None

    if cut_type in ['all', 'All']:
        # Pass the header argument when calling pd.read_csv.
        df = pd.read_csv(file_path, dtype=inferred_dtypes, header=header)
        df = _post_process_specific_datasets(df, file_type)
    else:
        # Validate_Signature_ex.py only uses the ‘all’ type, but leaves a message in case it is called with a different type.
        print(f"Warning: file_cut_GEN was called with cut_type='{cut_type}'. " +
              "This function is primarily designed for cut_type='all'. " +
              "Returning None or incompletely processed DataFrame.")
        # If needed, you can add logic for other cut_types here or raise an error.
        # Currently, df will return None.

    return df