import pandas as pd
import os
import glob
from pathlib import Path

def get_day_sort_key(filename_path, days_order):
    """
    Generates a sort key for a filename based on the day of the week.
    Files are sorted by the index of the day in days_order, then by filename.
    """
    basename = os.path.basename(filename_path)
    for i, day in enumerate(days_order):
        if basename.upper().startswith(day.upper()):
            return (i, basename)  # Sort by day index, then by original filename
    return (len(days_order), basename) # Files not matching known days go last

def combine_cicids_csvs(input_dir_relative_to_script, output_file_path_str):
    """
    Combines CICIDS2017 CSV files from a specified directory into a single CSV file.

    Args:
        input_dir_relative_to_script (str): Path to the input CSV directory,
                                             relative to this script's location.
        output_file_path_str (str): Path for the output combined CSV file.
                                     Can use '~' for user's home directory.
    """
    try:
        # Determine the script's own directory to resolve relative paths
        script_dir = Path(__file__).parent.resolve()
    except NameError:
        # Fallback for environments where __file__ might not be defined (e.g. interactive)
        script_dir = Path.cwd()


    # Construct absolute input directory path
    input_dir = (script_dir / input_dir_relative_to_script).resolve()

    # Expand user for output path and resolve it
    output_file_path = Path(output_file_path_str).expanduser().resolve()

    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file_path}")

    if not input_dir.is_dir():
        print(f"Error: Input directory not found: {input_dir}")
        return

    # Glob for .csv files, case-insensitive for extension
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    csv_files += glob.glob(os.path.join(input_dir, "*.CSV"))
    csv_files = sorted(list(set(csv_files))) # Remove duplicates and sort initially

    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    # CICIDS2017 typically involves these days in order
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

    # Sort files: first by day of the week, then by original filename
    csv_files.sort(key=lambda f: get_day_sort_key(f, days_order))

    print("\nFiles to be merged (in order):")
    for f_path in csv_files:
        print(f"  - {os.path.basename(f_path)}")

    all_data_list = []
    for file_path in csv_files:
        print(f"Processing: {os.path.basename(file_path)}...")
        try:
            # Explicitly handle potential mixed types and low_memory issues common with large CSVs
            df = pd.read_csv(file_path, skip_blank_lines=True, low_memory=False)
            
            # Clean column names (remove leading/trailing spaces)
            df.columns = df.columns.str.strip()
            
            # Drop rows that are entirely empty if any (though skip_blank_lines should handle some)
            df.dropna(how='all', inplace=True)

            if not df.empty:
                all_data_list.append(df)
                print(f"  Successfully read and added {os.path.basename(file_path)}. Shape: {df.shape}")
            else:
                print(f"  Warning: {os.path.basename(file_path)} is empty or became empty after cleaning, and will be skipped.")
        
        except pd.errors.EmptyDataError:
            print(f"  Warning: {os.path.basename(file_path)} is empty and will be skipped.")
        except Exception as e:
            print(f"  Error reading {os.path.basename(file_path)}: {e}")
            # Depending on desired behavior, you might want to re-raise or exit
            # For now, we'll skip the problematic file and continue
            print(f"  Skipping this file due to error.")

    if not all_data_list:
        print("\nNo data to merge after attempting to read CSV files.")
        return

    print(f"\nConcatenating {len(all_data_list)} DataFrame(s)...")
    try:
        combined_df = pd.concat(all_data_list, ignore_index=True)
    except Exception as e:
        print(f"Error during concatenation: {e}")
        return
        
    print(f"Concatenation complete. Total rows in combined DataFrame: {len(combined_df)}")

    # Ensure output directory exists
    try:
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Ensured output directory exists: {output_file_path.parent}")
    except Exception as e:
        print(f"Error creating output directory {output_file_path.parent}: {e}")
        return

    try:
        print(f"Saving combined data to {output_file_path}...")
        combined_df.to_csv(output_file_path, index=False)
        print(f"Successfully combined {len(all_data_list)} CSV files into {output_file_path}")
    except Exception as e:
        print(f"Error writing combined CSV to {output_file_path}: {e}")

if __name__ == '__main__':
    # The script CICIDS_file_sum.py will be located in the ASIC_code directory.
    # The CSV files are in a directory relative to ASIC_code:
    # ../Dataset/load_dataset/CICIDS2017/MachineLearningCSV
    
    # This path is relative to the location of this script.
    relative_input_csv_dir = "../Dataset/load_dataset/CICIDS2017/MachineLearningCSV"
    
    # Output file path as specified by the user.
    # '~' will be expanded to the user's home directory.
    output_csv_path = "~/asic/Dataset/load_dataset/CICIDS2017/CICIDS_all.csv"
    
    combine_cicids_csvs(relative_input_csv_dir, output_csv_path)
