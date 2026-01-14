import pandas as pd
import argparse
import os

# Import the necessary label judgment functions
from definition.Anomal_Judgment import anomal_judgment_label, anomal_judgment_nonlabel
# Import the dataset path chooser function
from Dataset_Choose_Rule.association_data_choose import file_path_line_association


def count_labels(file_path, file_type):
    """
    Reads a dataset file (CSV), creates a standardized 'label' column based on file_type,
    and counts the occurrences of benign (0) and attack (1) labels.

    Args:
        file_path (str): The path to the dataset file.
        file_type (str): The type of the dataset (e.g., 'CICIoT2023', 'Kitsune').
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    try:
        print(f"Reading dataset from: {file_path}")
        # Use low_memory=False for better performance with mixed data types
        data = pd.read_csv(file_path, low_memory=False)
        print("Dataset loaded successfully.")

        # --- NEW: Dynamically create the 'label' column based on file_type ---
        print(f"Standardizing labels for dataset type: '{file_type}'...")
        if file_type in ['MiraiBotnet', 'NSL-KDD', 'NSL_KDD']:
            data['label'], _ = anomal_judgment_nonlabel(file_type, data)
        elif file_type == 'netML':
            data['label'] = data['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
        elif file_type == 'DARPA98':
            data['label'] = data['Class'].apply(lambda x: 0 if str(x).strip() == '-' else 1)
        elif file_type in ['CICIDS2017', 'CICIDS']:
            data['label'] = data['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
        elif file_type in ['CICModbus23', 'CICModbus']:
            # Note: This is a specific condition for CICModbus23, might need adjustment
            data['label'] = data['Attack'].apply(lambda x: 0 if str(x).strip() == 'Baseline Replay: In position' else 1)
        elif file_type in ['IoTID20', 'IoTID']:
            data['label'] = data['Label'].apply(lambda x: 0 if str(x).strip() == 'Normal' else 1)
        elif file_type in ['CICIoT', 'CICIoT2023']:
            # Assuming 'attack_flag' is already 0 or 1
            data['label'] = data['attack_flag']
        elif file_type == 'Kitsune':
            # Assuming 'Label' is already 0 or 1
            data['label'] = data['Label']
        else:
            # Fallback for datasets with a standard 'label' column
            if 'label' in data.columns:
                 data['label'] = anomal_judgment_label(data)
            else:
                print(f"Error: Unknown file_type '{file_type}' and no 'label' column found.")
                print(f"Available columns are: {data.columns.tolist()}")
                return
        
        print("Label standardization complete.")

        label_counts = data['label'].value_counts(dropna=False)
        
        benign_count = label_counts.get(0, 0)
        attack_count = label_counts.get(1, 0)
        
        # Handle potential other labels if they exist
        other_labels = {k: v for k, v in label_counts.items() if k not in [0, 1]}
        
        total_rows = len(data)

        print("\n--- Label Distribution ---")
        print(f"Benign (label=0) rows: {benign_count}")
        print(f"Attack (label=1) rows: {attack_count}")
        print("--------------------------")
        if other_labels:
            print("Other labels found:")
            for label, count in other_labels.items():
                print(f"  - Label '{label}': {count} rows")
            print("--------------------------")
        print(f"Total rows: {total_rows}")

    except KeyError as e:
        print(f"\nError: A required column {e} was not found for file_type '{file_type}'.")
        print(f"Please check if the file_type is correct and the dataset has the expected columns.")
        print(f"Available columns are: {data.columns.tolist()}")
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")


def add(x, y):
    """Adds two numbers and returns the result."""
    return x + y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count benign and attack labels in a dataset file using file_type.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--file_type',
        type=str,
        required=True,
        help="Type of the dataset (e.g., CICIoT2023, Kitsune, NSL-KDD, etc.)"
    )
    parser.add_argument(
        '--file_path',
        type=str,
        default=None,
        help="Optional: Direct path to the CSV file. If not provided, the file path will be automatically located based on file_type."
    )

    args = parser.parse_args()
    
    # --- Get the file path: either from user input or automatically locate ---
    if args.file_path:
        # User provided a direct file path
        file_path = args.file_path
        print(f"Using provided file path: {file_path}")
    else:
        # Automatically locate the file based on file_type
        try:
            file_path, _ = file_path_line_association(args.file_type)
            print(f"Automatically located dataset file for '{args.file_type}': {file_path}")
        except FileNotFoundError as e:
            print(f"Error: Could not find the dataset file for file_type '{args.file_type}'.")
            print(f"Details: {e}")
            exit(1)
        except Exception as e:
            print(f"An unexpected error occurred while locating the dataset file: {e}")
            exit(1)

    count_labels(file_path, args.file_type)
