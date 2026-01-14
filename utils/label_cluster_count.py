import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Count 0/1 (and other values) in label/cluster columns of a CSV."
    )
    #parser.add_argument("--csv_path", type=str, default="C:\\ASIC_excute\\Dataset\\Dataset_ex\\best_clustering_MiraiBotnet_1.csv", help="Path to the CSV file.")
    parser.add_argument("--csv_path", type=str, default="C:\\Users\\SAMSUNG\\Desktop\\CSSLab\\ASIC\\cpfolder\\Dataset_ex\\load_dataset\\netML\\best_clustering_netML_1_AAP_scoring.csv", help="Path to the CSV file.")
    parser.add_argument("--label-col", default="label", help="Name of the label column (default: label).")
    parser.add_argument("--cluster-col", default="cluster", help="Name of the cluster column (default: cluster).")
    parser.add_argument("--sep", default=",", help="CSV separator (default: ',').")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path, sep=args.sep)

    def summarize(col_name):
        if col_name not in df.columns:
            print(f"[WARN] Column '{col_name}' not found.")
            return None
        vc = df[col_name].value_counts(dropna=False).sort_index()
        print(f"\nColumn '{col_name}':")
        for k, v in vc.items():
            print(f"  {k}: {v}")
        return vc

    label_vc = summarize(args.label_col)
    cluster_vc = summarize(args.cluster_col)

    if label_vc is not None and cluster_vc is not None:
        # Basic counts and overlap
        n_label1 = (df[args.label_col] == 1).sum()
        n_cluster1 = (df[args.cluster_col] == 1).sum()
        n_both = ((df[args.label_col] == 1) & (df[args.cluster_col] == 1)).sum()
        print(f"\nCounts -> label==1: {n_label1}, cluster==1: {n_cluster1}, overlap: {n_both}")

        print("\nDifferences (cluster - label) for keys present in either:")
        all_keys = set(label_vc.index).union(set(cluster_vc.index))
        for k in sorted(all_keys):
            lv = label_vc.get(k, 0)
            cv = cluster_vc.get(k, 0)
            print(f"  {k}: {cv - lv} (cluster {cv} vs label {lv})")


if __name__ == "__main__":
    main()

