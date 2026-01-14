# for dealing with support statistics
import pandas as pd


def get_dominant_columns(df: pd.DataFrame, freq_threshold: float = 0.99):
    """
    Returns a mapping of columns whose most frequent value occupies at least `freq_threshold`
    of the rows. These columns are effectively constant for support counting.
    """
    dominant_map = {}
    if df is None or df.empty:
        return dominant_map

    for col in df.columns:
        vc = df[col].value_counts(dropna=False)
        if vc.empty:
            continue
        top_count = vc.iloc[0]
        ratio = top_count / len(df)
        if ratio >= freq_threshold:
            dominant_map[col] = vc.index[0]
    return dominant_map


def build_rules_from_dominant(dominant_cols: dict):
    """
    If no rules were generated after masking, create a minimal rule using only
    the dominant columns/values so that downstream stages still have a rule to evaluate.
    """
    if not dominant_cols:
        return []
    return [dict(dominant_cols)]
