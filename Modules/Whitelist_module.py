import pandas as pd
from tqdm import tqdm

def calculate_single_signature_performance(signature, test_df, regrouping_map=None):
    """Calculates precision and recall for a single signature."""
    if not signature or test_df.empty:
        return 0.0, 0.0

    test_anomalous = test_df[test_df['label'] == 1]
    test_normal = test_df[test_df['label'] == 0]

    if test_anomalous.empty:
        return 0.0, 0.0 # Cannot calculate recall if there are no anomalies

    # Calculate True Positives (TP)
    mask_tp = pd.Series([True] * len(test_anomalous), index=test_anomalous.index)
    for key, value in signature.items():
        if key in test_anomalous.columns:
            if regrouping_map and key in regrouping_map:
                r_map = regrouping_map[key]
                sig_group = value
                mask_tp &= (test_anomalous[key].map(r_map).fillna(-2) == sig_group)
            else:
                mask_tp &= (test_anomalous[key] == value)
        else: # If a key from the signature isn't in the data, it can't match.
            mask_tp = pd.Series([False] * len(test_anomalous), index=test_anomalous.index)
            break
    tp_count = mask_tp.sum()

    # Calculate False Positives (FP)
    mask_fp = pd.Series([True] * len(test_normal), index=test_normal.index)
    for key, value in signature.items():
        if key in test_normal.columns:
            if regrouping_map and key in regrouping_map:
                r_map = regrouping_map[key]
                sig_group = value
                mask_fp &= (test_normal[key].map(r_map).fillna(-2) == sig_group)
            else:
                mask_fp &= (test_normal[key] == value)
        else:
             mask_fp = pd.Series([False] * len(test_normal), index=test_normal.index)
             break
    fp_count = mask_fp.sum()

    # Calculate Precision and Recall
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
    recall = tp_count / len(test_anomalous) if not test_anomalous.empty else 0.0
    
    return precision, recall

def create_whitelist(signatures, test_df, recall_threshold=0.5, precision_threshold=0.5, regrouping_map=None):
    """
    Creates a whitelist of high-performing signatures.
    Returns a set of frozensets for efficient lookup.
    """
    whitelist = set()
    
    print(f"  -> Evaluating {len(signatures)} individual signatures for whitelist creation...")
    for sig in tqdm(signatures, desc="Whitelist Progress"):
        p, r = calculate_single_signature_performance(sig, test_df, regrouping_map)
        
        # Only check for recall, precision is ignored for the whitelist.
        if r >= recall_threshold: # and p >= precision_threshold:
            whitelist.add(frozenset(sig.items()))
            
    return whitelist 