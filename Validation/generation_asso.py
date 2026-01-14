import pandas as pd
from collections import defaultdict
from itertools import combinations
import logging

logger = logging.getLogger(__name__)

def calculate_actual_support(df, itemset_dict):
    """
    Calculates the actual support of a given itemset (rule condition dictionary) in the dataframe.
    """
    if not itemset_dict or df.empty:
        return 0.0
    
    query_parts = []
    for key, value in itemset_dict.items():
        # Ensure column names with spaces or special characters are quoted
        safe_key = f"`{str(key).replace('`', '``')}`"
        if isinstance(value, str):
            # Escape single quotes within string values for the query
            safe_value = str(value).replace("'", "\\'")
            query_parts.append(f"{safe_key} == '{safe_value}'")
        elif pd.isna(value): # Handle potential NaN values if they are part of an itemset (though usually filtered)
             query_parts.append(f"{safe_key}.isnull()")
        else: # Numeric or boolean
            query_parts.append(f"{safe_key} == {value}")
    
    query_string = " & ".join(query_parts)
    
    try:
        matches = df.query(query_string)
        return len(matches) / len(df) if len(df) > 0 else 0.0
    except Exception as e:
        logger.debug(f"Query error for itemset {itemset_dict} (query: '{query_string}'): {e}")
        return 0.0

def temp_rarm_for_fake_fp(df, min_support_threshold, num_rules_to_generate=5, itemset_size=2, fixed_confidence=0.9):
    """
    Temporary RARM logic for generating fake FP signatures.
    Generates itemsets of a specified size that meet a minimum actual support, 
    and assigns a fixed confidence.

    Args:
        df (pd.DataFrame): The mapped input dataframe.
        min_support_threshold (float): Minimum actual support for an itemset to be considered.
        num_rules_to_generate (int): Max number of rules to generate.
        itemset_size (int): The number of conditions (items) in each generated rule.
        fixed_confidence (float): The confidence value to assign to all generated rules.

    Returns:
        list: A list of rule dictionaries. Each dictionary contains the rule conditions
              (feature-value pairs) and the keys 'confidence' and 'support'.
    """
    if df.empty:
        logger.warning("temp_rarm_for_fake_fp: Input DataFrame is empty. Returning empty list.")
        return []

    all_columns = df.columns.tolist()
    generated_rules = []
    
    if itemset_size <= 0:
        logger.warning(f"temp_rarm_for_fake_fp: itemset_size must be positive. Received {itemset_size}. Returning empty list.")
        return []
    if itemset_size > len(all_columns):
        logger.warning(f"temp_rarm_for_fake_fp: itemset_size ({itemset_size}) is greater than number of available columns ({len(all_columns)}). "
                       f"Adjusting itemset_size to {len(all_columns)}.")
        itemset_size = len(all_columns)
        if itemset_size == 0: # No columns to form itemsets
             return []

    logger.info(f"temp_rarm_for_fake_fp: Generating up to {num_rules_to_generate} rules with itemset_size={itemset_size}, "
                f"min_support={min_support_threshold:.3f}, fixed_confidence={fixed_confidence:.2f}")

    # Iterate over combinations of column names
    possible_column_combinations = combinations(all_columns, itemset_size)

    for col_combo in possible_column_combinations:
        if len(generated_rules) >= num_rules_to_generate:
            break

        # Step 1: Create an itemset with original data types for support calculation.
        itemset_for_calc = {}
        valid_condition_set = True
        for col_name in col_combo:
            mode_result = df[col_name].mode() 
            if not mode_result.empty:
                value = mode_result.iloc[0]
                itemset_for_calc[col_name] = value # Keep original type
            else:
                valid_condition_set = False
                break
        
        if not valid_condition_set or not itemset_for_calc:
            continue

        # Step 2: Calculate support using the itemset with original types.
        actual_support = calculate_actual_support(df, itemset_for_calc)

        if actual_support >= min_support_threshold:
            # Step 3: If support is sufficient, create the final rule dictionary
            # using the original data types. Do NOT convert values to strings.
            final_rule_dict = dict(itemset_for_calc)
            
            final_rule_dict['confidence'] = fixed_confidence
            final_rule_dict['support'] = actual_support
            
            generated_rules.append(final_rule_dict)
            logger.info(f"temp_rarm_for_fake_fp: Added rule {final_rule_dict} with support {actual_support:.4f}")

    if not generated_rules:
        logger.warning(f"temp_rarm_for_fake_fp: No rules met the support threshold of {min_support_threshold:.3f} with itemset_size={itemset_size}.")
    
    logger.info(f"temp_rarm_for_fake_fp: Finished. Generated {len(generated_rules)} rules.")
    return generated_rules
