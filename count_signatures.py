import os
import pandas as pd
import argparse
import re


def parse_summary_from_header(file_path):
    """
    Parses Loop Limit and Signature Count from the commented header of a CSV file.
    """
    loop_limit = 'N/A'
    sig_count_header = 'N/A'
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    if 'Loop Limit Level' in line:
                        match = re.search(r':\s*(\S+)', line)
                        if match:
                            loop_limit = match.group(1)
                    elif 'Final Signature Count' in line:
                        match = re.search(r':\s*(\d+)', line)
                        if match:
                            sig_count_header = match.group(1)
                else:
                    # Stop reading after the header comments
                    break
    except Exception:
        pass # Ignore errors if header can't be read
    return loop_limit, sig_count_header


def count_signatures_in_files(directory):
    """
    Scans a directory for association result CSVs, reads summary info from headers,
    and counts the actual signatures in each file.
    """
    print(f"Scanning directory: {directory}\n")
    
    results = []
    
    # Walk through the directory and its subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith("_association_result.csv"):
                file_path = os.path.join(root, file)
                try:
                    # Get summary from header
                    loop_limit, sig_count_header = parse_summary_from_header(file_path)
                    
                    # Get actual count from CSV content
                    df = pd.read_csv(file_path, comment='#')
                    actual_sig_count = len(df)
                    
                    results.append({
                        'Filename': file,
                        'Loop Limit (from header)': loop_limit,
                        'Signature Count (from header)': sig_count_header,
                        'Actual Signature Count (rows)': actual_sig_count
                    })
                except Exception as e:
                    results.append({
                        'Filename': file,
                        'Loop Limit (from header)': 'Error',
                        'Signature Count (from header)': 'Error',
                        'Actual Signature Count (rows)': f"Could not process: {e}"
                    })

    if not results:
        print("No '_association_result.csv' files found.")
        return

    # Print results in a formatted table
    results_df = pd.DataFrame(results)
    print("--- Signature Count Summary ---")
    print(results_df.to_string(index=False))
    
    # Optionally, save the summary to a CSV
    summary_save_path = os.path.join(directory, "z_signature_counts_summary.csv")
    results_df.to_csv(summary_save_path, index=False)
    print(f"\nSummary saved to: {summary_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count signatures in previously generated association rule CSV files.")
    parser.add_argument(
        '--scan_dir', 
        type=str, 
        default="../Dataset_Paral/signature/",
        help="The root directory to scan for signature CSV files."
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.scan_dir):
        print(f"Error: Directory not found at '{args.scan_dir}'")
    else:
        count_signatures_in_files(args.scan_dir)
