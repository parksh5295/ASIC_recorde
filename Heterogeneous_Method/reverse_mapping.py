"""
Reverse mapping utilities for converting group numbers back to original interval/categorical/binary values.
"""

import pandas as pd


def build_reverse_mapping(category_mapping):
    """
    Build reverse mapping dictionaries from group numbers to interval/categorical/binary values.
    Returns a dictionary: {column_name: {group_number: original_value_string}}
    
    Args:
        category_mapping: Dictionary containing 'interval', 'categorical', and/or 'binary' mappings
        
    Returns:
        Dictionary mapping column names to {group_number: original_value_string} dictionaries
    """
    reverse_mapping = {}
    
    # Process interval mappings
    if 'interval' in category_mapping and isinstance(category_mapping['interval'], pd.DataFrame):
        interval_rules = category_mapping['interval']
        for col in interval_rules.columns:
            col_mapping = {}
            for rule_str in interval_rules[col].dropna():
                try:
                    # Parse "interval=group" format, keeping the original interval string exactly as is
                    if '=' in rule_str:
                        interval_part, group_str = rule_str.split('=', 1)  # Split only on first '=' to handle edge cases
                        group = int(group_str)
                        # Store the original interval string without modification
                        col_mapping[group] = interval_part
                except (ValueError, IndexError):
                    continue
            if col_mapping:
                reverse_mapping[col] = col_mapping
    
    # Process categorical mappings
    if 'categorical' in category_mapping:
        categorical_mapping = category_mapping['categorical']
        if isinstance(categorical_mapping, dict):
            for feature, mapping_dict in categorical_mapping.items():
                if isinstance(mapping_dict, dict):
                    # mapping_dict is {value: group}
                    col_mapping = {group: str(value) for value, group in mapping_dict.items()}
                    if col_mapping:
                        reverse_mapping[feature] = col_mapping
                elif isinstance(categorical_mapping, pd.DataFrame) and feature in categorical_mapping.columns:
                    # Handle DataFrame format if needed
                    pass
        elif isinstance(categorical_mapping, pd.DataFrame):
            for col in categorical_mapping.columns:
                # For DataFrame format, assume each row is "value=group"
                col_mapping = {}
                for rule_str in categorical_mapping[col].dropna():
                    try:
                        if '=' in rule_str:
                            value_part, group_str = rule_str.split('=', 1)
                            group = int(group_str)
                            col_mapping[group] = value_part
                    except (ValueError, IndexError):
                        continue
                if col_mapping:
                    reverse_mapping[col] = col_mapping
    
    # Process binary mappings (similar to categorical)
    if 'binary' in category_mapping:
        binary_mapping = category_mapping['binary']
        if isinstance(binary_mapping, dict):
            for feature, mapping_dict in binary_mapping.items():
                if isinstance(mapping_dict, dict):
                    col_mapping = {group: str(value) for value, group in mapping_dict.items()}
                    if col_mapping:
                        reverse_mapping[feature] = col_mapping
        elif isinstance(binary_mapping, pd.DataFrame):
            for col in binary_mapping.columns:
                # For DataFrame format, assume each row is "value=group"
                col_mapping = {}
                for rule_str in binary_mapping[col].dropna():
                    try:
                        if '=' in rule_str:
                            value_part, group_str = rule_str.split('=', 1)
                            group = int(group_str)
                            col_mapping[group] = value_part
                    except (ValueError, IndexError):
                        continue
                if col_mapping:
                    reverse_mapping[col] = col_mapping
    
    return reverse_mapping


def reverse_map_rule(rule, reverse_mapping):
    """
    Convert a rule from group numbers to original interval/categorical/binary values.
    Preserves the exact format of interval strings without any modification.
    
    Args:
        rule: Dictionary with {feature: group_number}
        reverse_mapping: Dictionary from build_reverse_mapping: {column_name: {group_number: original_value_string}}
    
    Returns:
        Dictionary with {feature: original_value_string}
    """
    reversed_rule = {}
    for feature, group_num in rule.items():
        if feature in reverse_mapping:
            if group_num in reverse_mapping[feature]:
                # Use the exact original value string (especially important for intervals)
                reversed_rule[feature] = reverse_mapping[feature][group_num]
            else:
                # Group number not found in mapping, keep as-is with a marker
                reversed_rule[feature] = f"<UNMAPPED_GROUP_{group_num}>"
        else:
            # Feature not in reverse mapping (might be label column or other), keep as-is
            reversed_rule[feature] = group_num
    return reversed_rule
