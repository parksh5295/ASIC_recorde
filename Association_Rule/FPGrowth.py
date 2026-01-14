# FP-Growth(Frequent Pattern Growth) Algorithm
# Better than Apriori for processing large amounts of data
# Output: Association rules Dictionary List; [{'feature1': value1, 'feature2': value2, ...}, {...}, ...]

'''
import pandas as pd
from fim import fpgrowth

def FPGrowth_rule(df, min_support=0.5, min_confidence=0.8):
    """
    Fast FP-Growth using pyfim
    Input:
        df: pandas DataFrame (categorical or binary)
        min_support: float (0.0~1.0, proportion)
        min_confidence: float (0.0~1.0, proportion)
    Output:
        List of rule dictionaries [{feature1: value1, feature2: value2, ...}, ...]
    """

    # Convert DataFrame to transactions: list of lists like ['proto=tcp', 'flag=1', ...]
    transactions = df.astype(str).apply(lambda row: [f"{col}={val}" for col, val in row.items()], axis=1).tolist()

    abs_support = int(len(transactions) * min_support * 1000 / 1000)  # pyfim uses absolute integer support
    abs_conf = int(min_confidence * 100)  # pyfim uses percent confidence (0–100)

    # Run FP-Growth in rule generation mode ('r') and report antecedents + consequents ('aC')
    raw_rules = fpgrowth(transactions, supp=abs_support, conf=abs_conf, report='aC', target='r')

    rule_dicts = []
    for rule in raw_rules:
        antecedents = rule[0]
        consequents = rule[1]

        # Parse antecedents and consequents back into dictionary form
        antecedent_dict = {kv.split('=')[0]: int(kv.split('=')[1]) for kv in antecedents}
        consequent_dict = {kv.split('=')[0]: int(kv.split('=')[1]) for kv in consequents}

        combined_rule = {**antecedent_dict, **consequent_dict}
        sorted_rule = {k: combined_rule[k] for k in sorted(combined_rule)}

        if sorted_rule not in rule_dicts:
            rule_dicts.append(sorted_rule)

    return rule_dicts
'''


import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

def FPGrowth_rule(df, min_support=0.5, min_confidence=0.8, association_metric='confidence', num_processes=None, file_type_for_limit=None, max_level_limit=None, itemset_limit=None, turn_counter=None, params_str=None,
                  enable_dynamic_support=False, dynamic_support_threshold=500000, support_increment_factor=1.2, **kwargs):
    # Decide on a metrics method
    metric = association_metric

    # One-Hot Encoding with sparse matrix for memory efficiency
    df_encoded = pd.get_dummies(df.astype(str), prefix_sep="=", sparse=True)

    # Apply FP-Growth
    frequent_itemsets = fpgrowth(df_encoded, 
                                min_support=min_support, 
                                use_colnames=True)

    # Check if frequent itemsets are empty
    if frequent_itemsets.empty:
        return [], 0

    # --- ITEMSET_LIMIT CHECK (Safety Valve) ---
    if itemset_limit is not None and len(frequent_itemsets) > itemset_limit:
        print(f"    [WARN] Safety valve triggered: Number of frequent itemsets ({len(frequent_itemsets)}) exceeds the limit ({itemset_limit}).")
        print("    Stopping rule generation to prevent memory overflow.")
        print("    Consider increasing min_support or min_confidence.")
        # Calculate max level from frequent itemsets before stopping
        max_level_reached = max(len(itemset) for itemset in frequent_itemsets['itemsets']) if not frequent_itemsets.empty else 0
        return [], max_level_reached

    # Generate rules
    rules = association_rules(frequent_itemsets, 
                            metric=metric, 
                            min_threshold=min_confidence, 
                            num_itemsets=len(frequent_itemsets))

    # Check if rules are empty
    if rules.empty:
        # Calculate max level from frequent itemsets if no rules but itemsets exist
        max_level_reached = max(len(itemset) for itemset in frequent_itemsets['itemsets']) if not frequent_itemsets.empty else 0
        return [], max_level_reached

    # Pre-split column names for faster processing
    column_map = {col: col.split("=") for col in df_encoded.columns}
    
    # Use set for faster duplicate removal
    unique_rules = set()
    max_items_in_rule = 0
    
    # Use to_dict('records') instead of iterrows
    for rule in rules[['antecedents', 'consequents']].to_dict('records'):
        # Add directly to single dictionary
        combined_rule = {}
        
        # Process antecedents and consequents together
        for items in (rule['antecedents'], rule['consequents']):
            for item in items:
                key, value = column_map[item]
                combined_rule[key] = int(value)
        
        # Track maximum items in rule for max_level calculation
        num_items = len(combined_rule)
        if num_items > max_items_in_rule:
            max_items_in_rule = num_items
        
        # Convert to sorted tuple for faster set addition
        rule_tuple = tuple(sorted(combined_rule.items()))
        unique_rules.add(rule_tuple)

    # Convert set to final result format
    max_level_reached = max_items_in_rule
    return [dict(rule) for rule in unique_rules], max_level_reached
