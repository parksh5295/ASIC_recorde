import pandas as pd
import argparse
import ast

def cut_signatures(input_file, output_file, top_n=40):
    """
    Reads a signature CSV file, extracts the top N signatures by Precision,
    and saves them to a new CSV file in the same format.
    """
    try:
        main_df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except Exception as e:
        print(f"Error reading CSV '{input_file}': {e}")
        return

    if main_df.empty:
        print(f"Error: Input file '{input_file}' is empty.")
        return

    if not all(col in main_df.columns for col in ['Verified_Signatures', 'Recall', 'Best_confidence']):
        print("Error: Input CSV must contain 'Verified_Signatures', 'Recall', and 'Best_confidence' columns.")
        return
        
    original_recall = main_df.loc[0, 'Recall']
    original_best_confidence = main_df.loc[0, 'Best_confidence']

    try:
        signatures_str = main_df.loc[0, 'Verified_Signatures']
        
        signatures_list = []
        if pd.isna(signatures_str): # Handles numpy.nan, None
            signatures_list = []
        elif isinstance(signatures_str, str):
            if signatures_str.lower() == 'nan' or not signatures_str.strip(): # Handles "nan" string or empty string
                signatures_list = []
            else:
                signatures_list = ast.literal_eval(signatures_str)
        else: # Should be a list already if not string or NaN (though CSV would make it string)
             print(f"Warning: 'Verified_Signatures' was not a string or NaN. Attempting to use as is: {type(signatures_str)}")
             signatures_list = signatures_str


        if not isinstance(signatures_list, list):
            print(f"Error: 'Verified_Signatures' column (value: {signatures_str}) does not evaluate to a valid list.")
            return
        if not all(isinstance(item, dict) for item in signatures_list):
            if signatures_list: # Only error if list is not empty and items are not dicts
                 print("Error: Not all items in 'Verified_Signatures' are dictionaries.")
                 return

    except (SyntaxError, ValueError) as e:
        print(f"Error parsing 'Verified_Signatures' from '{input_file}': {e}")
        print(f"Content of 'Verified_Signatures': {main_df.loc[0, 'Verified_Signatures']}")
        print("The 'Verified_Signatures' column should contain a string representation of a list of dictionaries, or be empty/NaN.")
        return
    except KeyError:
        # This should be caught by the check above, but as a safeguard.
        print("Error: 'Verified_Signatures', 'Recall', or 'Best_confidence' column not found in input CSV.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while processing 'Verified_Signatures': {e}")
        return

    top_n_signatures_list = []
    if not signatures_list:
        print("No signatures found in the input file. The 'Verified_Signatures' field in the output will be an empty list.")
        top_n_signatures_list = []
    else:
        signatures_df = pd.DataFrame(signatures_list)

        if 'Precision' not in signatures_df.columns:
            print("Error: 'Precision' column not found in the signature data within 'Verified_Signatures'.")
            print("The 'Verified_Signatures' field in the output will be an empty list.")
            top_n_signatures_list = []
        else:
            signatures_df_sorted = signatures_df.sort_values(by='Precision', ascending=False, kind='mergesort') # Added kind for stable sort
            top_n_df = signatures_df_sorted.head(top_n)
            top_n_signatures_list = top_n_df.to_dict(orient='records')

    output_data_for_df = {
        'Verified_Signatures': str(top_n_signatures_list), # Convert list of dicts back to string for CSV
        'Recall': original_recall,
        'Best_confidence': original_best_confidence
    }
    output_df = pd.DataFrame([output_data_for_df])

    try:
        output_df.to_csv(output_file, index=False)
        print(f"Successfully saved top {len(top_n_signatures_list)} signatures to '{output_file}'")
    except Exception as e:
        print(f"Error writing CSV to '{output_file}': {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Filters a signature CSV file to keep only the top N signatures by Precision. "
                    "The input CSV is expected to have one row, with columns 'Verified_Signatures', "
                    "'Recall', and 'Best_confidence'. 'Verified_Signatures' should be a "
                    "string representation of a list of dictionaries, where each dictionary is a signature "
                    "and contains a 'Precision' key."
    )
    parser.add_argument("input_file", help="Path to the input signature CSV file.")
    parser.add_argument("output_file", help="Path to save the filtered signature CSV file.")
    parser.add_argument("--top_n", type=int, default=40, help="Number of top signatures to keep (default: 40).")

    args = parser.parse_args()

    cut_signatures(args.input_file, args.output_file, args.top_n)
