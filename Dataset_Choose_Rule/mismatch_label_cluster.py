import pandas as pd
import argparse
import os


def main(input_csv, output_csv):
    # Load CSV
    df = pd.read_csv(input_csv)

    # Sanity check
    required_cols = {"label", "cluster"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # Ensure values are comparable (in case they are strings)
    df["label"] = df["label"].astype(int)
    df["cluster"] = df["cluster"].astype(int)

    # Filter rows where label != cluster
    mismatch_df = df[df["label"] != df["cluster"]]

    # Save to new CSV (order preserved automatically)
    mismatch_df.to_csv(output_csv, index=False)

    print(f"Saved {len(mismatch_df)} mismatched rows to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract rows where label != cluster")
    parser.add_argument(
        "--input",
        default="C:\\ASIC_excute\\Dataset\\Dataset_ex\\best_clustering_MiraiBotnet_1.csv",
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output",
        default="C:\\ASIC_excute\\Dataset\\Dataset_ex\\mismatch_label_cluster_MiraiBotnet.csv",
        help="Path to output CSV file"
    )

    args = parser.parse_args()
    main(args.input, args.output)
