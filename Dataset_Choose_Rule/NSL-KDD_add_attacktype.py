import pandas as pd


# =========================
BASE_CSV_PATH = r"C:/ASIC_excute/Dataset/NSL-KDD_tool/best_clustering_NSL-KDD_1.csv"      # CSV with existing columns
NSL_RAW_PATH  = r"C:/ASIC_excute/Dataset/NSL-KDD_tool/KDDTrain_txt.csv"     # Newly received NSL-KDD original
OUTPUT_CSV_PATH = r"C:/ASIC_excute/Dataset/NSL-KDD_tool/best_clustering_NSL-KDD_1_class.csv"
CLASS_COL_NAME = "class"
# =========================


def attach_class_column():
    # 1. Existing CSV (with header)
    base_df = pd.read_csv(BASE_CSV_PATH)

    # 2. NSL-KDD original (no header)
    raw_df = pd.read_csv(NSL_RAW_PATH, header=None)

    # 3. row count verification
    if len(base_df) != len(raw_df):
        raise ValueError(
            f"Row count mismatch: base={len(base_df)}, raw={len(raw_df)}"
        )

    # 4. Last column = attack type
    class_series = raw_df.iloc[:, -2]

    # 5. Add class column to existing DF (maintain order)
    base_df[CLASS_COL_NAME] = class_series.values

    # 6. Save
    base_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print("[OK] Class column successfully attached")
    print(f" - rows: {len(base_df)}")
    print(f" - output: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    attach_class_column()
