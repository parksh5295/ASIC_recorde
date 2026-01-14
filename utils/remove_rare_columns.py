import pandas as pd
from tqdm import tqdm

# Constants for optimization
HIGH_CARDINALITY_THRESHOLD = 10000  # Columns with more unique values than this will be skipped

def remove_rare_columns(df, min_support_ratio, file_type, min_distinct_frequent_values=2, cols_to_protect=None):
    """
    Removes columns from a DataFrame where no single value meets a specified support threshold.
    Optimized for performance on large datasets with high cardinality columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        min_support_ratio (float): The minimum support ratio a value must have to be considered frequent.
        file_type (str): The type of the dataset file (used for logging/context).
        min_distinct_frequent_values (int): The minimum number of distinct values that must meet the
                                            support threshold for a column to be kept.
        cols_to_protect (list, optional): A list of column names that should not be removed, regardless of their statistics. Defaults to None.
    Returns:
        pd.DataFrame: A new DataFrame with rare columns removed.
    """
    if df.empty:
        return df

    if cols_to_protect is None:
        cols_to_protect = []

    support_count_threshold = len(df) * min_support_ratio
    cols_to_drop = []
    
    # Use tqdm to show progress over the columns
    for col in tqdm(df.columns, desc="[Filter] Analyzing rare columns"):
        if col in cols_to_protect:
            continue

        try:
            # --- Optimization 1: Skip high cardinality columns ---
            # Columns with a huge number of unique values (like IPs or IDs) are unlikely
            # to have any single value meeting the support threshold and are slow to process.
            num_unique_values = df[col].nunique()
            if num_unique_values > HIGH_CARDINALITY_THRESHOLD:
                # print(f"  [Info] Skipping high-cardinality column '{col}' ({num_unique_values} unique values).")
                cols_to_drop.append(col)
                continue

            # --- Optimization 2: Use efficient value_counts ---
            value_counts = df[col].value_counts(dropna=True)
            
            # Find values that meet the support threshold
            frequent_values = value_counts[value_counts >= support_count_threshold]
            
            # Check if the number of distinct frequent values is sufficient
            if len(frequent_values) < min_distinct_frequent_values:
                cols_to_drop.append(col)
                
        except Exception as e:
            print(f"  [Warning] Could not process column '{col}'. It may be dropped. Error: {e}")
            cols_to_drop.append(col)
            
    if cols_to_drop:
        print(f"  [Info] Dropping {len(cols_to_drop)} columns that don't meet the frequency criteria.")
        # print(f"  Columns to drop: {cols_to_drop}")
        df_filtered = df.drop(columns=cols_to_drop)
    else:
        print("  [Info] No columns needed to be dropped based on frequency criteria.")
        df_filtered = df

    return df_filtered

