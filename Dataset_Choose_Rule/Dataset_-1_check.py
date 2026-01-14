import pandas as pd
import sys


def main(csv_path):
    try:
        csv_path = "C:\\ASIC_excute\\Dataset\\load_dataset\\training-flow.csv"
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error while reading CSV file: {e}")
        return

    if "attack_flag" not in df.columns:
        print("Column 'attack_flag' does not exist in the CSV file.")
        return

    filtered = df[df["attack_flag"] == -1]

    if filtered.empty:
        print("No rows found where attack_flag == -1.")
        return

    # Take up to the first 5 matching rows from the top
    result = filtered.head(5)
    print(result)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python print_attack_rows.py <csv_file_path>")
    else:
        main(sys.argv[1])
