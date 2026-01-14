import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import multiprocessing
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def fpc(df, min_support=0.5, min_confidence=0.8, num_processes=None, **kwargs):
    """
    Performs frequent pattern mining using FP-Growth and then prunes the results 
    to find only closed patterns before generating association rules. This significantly
    reduces the number of redundant rules.

    Args:
        df (pd.DataFrame): Input data.
        min_support (float): Minimum support threshold.
        min_confidence (float): Minimum confidence threshold.
        num_processes (int): Number of processes to use (Note: mlxtend's fpgrowth is single-threaded).
                             This argument is kept for signature consistency but not used by fpgrowth.

    Returns:
        list: A list of dictionaries, where each dictionary represents a generated association rule.
        int: The maximum level/size of the itemsets from which rules were generated.
    """
    if df.empty:
        return [], 0

    logger.info(f"    [FPC Init] Starting FP-Closed algorithm. Support={min_support}, Confidence={min_confidence}")

    # 1. Convert DataFrame to transaction list format required by mlxtend
    logger.info("    [FPC] Converting data to transaction format...")
    transactions = []
    for row in tqdm(df.itertuples(index=False, name=None), total=len(df), desc="    [FPC] Transactionizing"):
        transactions.append(set(f"{col}={val}" for col, val in zip(df.columns, row) if pd.notna(val)))

    if not transactions:
        return [], 0
        
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_onehot = pd.DataFrame(te_ary, columns=te.columns_)

    # 2. Run FP-Growth to get all frequent itemsets
    logger.info("    [FPC] Running FP-Growth to find all frequent itemsets...")
    frequent_itemsets = fpgrowth(df_onehot, min_support=min_support, use_colnames=True)
    
    if frequent_itemsets.empty:
        logger.warning("    [FPC] No frequent itemsets found with the given support. No rules will be generated.")
        return [], 0

    logger.info(f"    [FPC] Found {len(frequent_itemsets)} frequent itemsets. Pruning to find closed patterns...")

    # 3. Prune to find "closed" itemsets
    # An itemset is closed if no superset has the same support.
    itemset_supports = {frozenset(itemset): support for itemset, support in zip(frequent_itemsets['itemsets'], frequent_itemsets['support'])}
    
    closed_itemsets = []
    # Sort by length descending to optimize the check
    sorted_itemsets = sorted(itemset_supports.keys(), key=len, reverse=True)

    for itemset in tqdm(sorted_itemsets, desc="    [FPC] Pruning to Closed"):
        is_closed = True
        current_support = itemset_supports[itemset]
        
        # Check against already confirmed closed itemsets (which are supersets or unrelated)
        for closed_set in closed_itemsets:
            if itemset.issubset(closed_set) and current_support == itemset_supports[closed_set]:
                is_closed = False
                break
        
        if is_closed:
            closed_itemsets.append(itemset)

    if not closed_itemsets:
        logger.warning("    [FPC] No closed itemsets found after pruning.")
        return [], 0

    logger.info(f"    [FPC] Pruning complete. {len(closed_itemsets)} closed itemsets remain.")
    
    # Re-create DataFrame for rule generation
    closed_df_data = []
    for itemset in closed_itemsets:
        closed_df_data.append({'support': itemset_supports[itemset], 'itemsets': itemset})
    
    frequent_closed_itemsets_df = pd.DataFrame(closed_df_data)

    max_level = frequent_closed_itemsets_df['itemsets'].apply(len).max() if not frequent_closed_itemsets_df.empty else 0

    # 4. Generate association rules from the closed itemsets
    logger.info(f"    [FPC] Generating association rules from {len(frequent_closed_itemsets_df)} closed itemsets...")
    rules_df = association_rules(frequent_closed_itemsets_df, metric="confidence", min_threshold=min_confidence)

    # 5. Convert rules to the required dictionary format
    final_rules = []
    for _, row in rules_df.iterrows():
        full_itemset = row['antecedents'].union(row['consequents'])
        rule_dict = {}
        for item_str in full_itemset:
            try:
                key, value_str = item_str.split('=', 1)
                # Attempt to convert value back to numeric if possible
                val_float = float(value_str)
                rule_dict[key] = int(val_float) if val_float.is_integer() else val_float
            except (ValueError, IndexError):
                rule_dict[key] = value_str # Keep as string if conversion fails
        final_rules.append(rule_dict)

    logger.info(f"    [FPC] Finished. Generated {len(final_rules)} rules. Max itemset size: {max_level}")
    return final_rules, max_level
