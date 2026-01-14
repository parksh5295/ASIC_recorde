import pandas as pd
import argparse

# Modules
from Dataset_Choose_Rule.association_data_choose import get_clustered_data_path
from utils.time_transfer import time_scalar_transfer


def compute_dice_from_csv(
    file_type,
    file_number,
    label_col="label",
    cluster_col="cluster",
    chunksize=100_000
):
    tp = 0
    pred_pos = 0
    true_pos = 0

    # --- Data Loading and Initial Mapping (ALWAYS RUNS) ---
    file_path, total_rows = get_clustered_data_path(file_type, file_number)

    for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):
        #processed_chunk = time_scalar_transfer(chunk, file_type)

        # If it's a string, cast it to an integer
        #labels = chunk[label_col].astype(int)
        #clusters = chunk[cluster_col].astype(int)

        labels = chunk[label_col]
        clusters = chunk[cluster_col]

        labels = labels[labels.isin([0, 1])]
        clusters = clusters.loc[labels.index]

        clusters = clusters.isin([1]).astype(int)
        labels = labels.astype(int)

        tp += ((labels == 1) & (clusters == 1)).sum()
        pred_pos += (clusters == 1).sum()
        true_pos += (labels == 1).sum()

    if pred_pos + true_pos == 0:
        return 0.0

    dice = 2 * tp / (pred_pos + true_pos)
    return dice


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk-wise Dice coefficient computation")
    parser.add_argument('--file_type', type=str, default="MiraiBotnet", help="Type of the dataset file.")
    parser.add_argument('--file_number', type=int, default=1, help="Number of the dataset file.")
    parser.add_argument("--label_col", default="label", help="Ground-truth label column")
    parser.add_argument("--cluster_col", default="cluster", help="Predicted/virtual label column")
    parser.add_argument("--chunksize", type=int, default=100_000, help="Chunk size for streaming")

    args = parser.parse_args()

    dice = compute_dice_from_csv(
        args.file_type,
        args.file_number,
        args.label_col,
        args.cluster_col,
        args.chunksize
    )

    print(f"[DICE] {dice:.6f}")
