import pandas as pd


def cluster_col_check(csv_path: str, column_name: str = "cluster"):
    print(f"Checking column: {column_name} in {csv_path}")
    df = pd.read_csv(csv_path)
    counts = df[column_name].value_counts().sort_index()
    print("Value counts for column:", column_name)
    print("0 count:", counts.get(0, 0))
    print("1 count:", counts.get(1, 0))

if __name__ == "__main__":
    csv_path = "../Dataset_ex/load_dataset/Kitsune/best_clustering_Kitsune_1.csv"
    cluster_col_check(csv_path)
    print("Done")