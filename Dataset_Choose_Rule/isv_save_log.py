import pandas as pd
import os
import logging


logger = logging.getLogger(__name__)

def get_params_str(args, max_level=None):
    """
    Generates a standardized string from argparse arguments for directory/file naming.
    Includes an optional level limit.
    """
    level_str = f"_l{max_level}" if max_level is not None else "_l_inf"
    # base = f"{args.association_method}_s{args.min_support}_c{args.min_confidence}_cs{args.chunk_size}{level_str}"
    chunk_label = getattr(args, 'chunk_size_label', None)
    chunk_val = chunk_label if chunk_label is not None else args.chunk_size
    base = f"{args.association_method}_s{args.min_support}_c{args.min_confidence}_cs{chunk_val}{level_str}"
    normal_min_support = getattr(args, 'normal_min_support', None)
    if normal_min_support is not None:
        base += f"_ns{normal_min_support}"
    return base


def save_association_artifacts(itemsets_or_rules, file_type: str, turn_counter: int, level: int, artifact_type: str, params_str: str):
    """
    Saves intermediate artifacts from association rule mining (itemsets or rules) to a structured directory.

    Args:
        data_to_save (list or DataFrame): The list of itemsets or rules to save.
        file_type (str): The type of the dataset being processed.
        turn_counter (int): The current turn/chunk number in the incremental process.
        level (int): The current level (itemset size) in the association mining algorithm.
        artifact_type (str): The type of artifact, e.g., 'frequent_itemsets' or 'generated_rules'.
        params_str (str): A string representing the key parameters for unique folder names.
    """
    if not itemsets_or_rules:
        return
        
    try:
        # Define directory structure
        base_dir = "../Dataset_ISV"
        run_dir = os.path.join(base_dir, file_type, params_str)
        turn_dir = os.path.join(run_dir, f"turn_{turn_counter}")
        os.makedirs(turn_dir, exist_ok=True)

        # Define filename and save mode based on artifact_type
        is_append_mode = 'log' in artifact_type
        
        if is_append_mode:
            # For logging, use a single file per turn and append.
            filename = f"{artifact_type}.csv"
        else:
            # DEPRECATED: This path is no longer used but kept for safety.
            # All artifacts should now use the '.log' suffix to append.
            filename = f"level_{level}_{artifact_type}.csv"
        
        filepath = os.path.join(turn_dir, filename)

        # Convert to DataFrame if it's a list of dictionaries
        if isinstance(itemsets_or_rules, list):
            df = pd.DataFrame(itemsets_or_rules)
        elif isinstance(itemsets_or_rules, pd.DataFrame):
            df = itemsets_or_rules
        else:
            logger.warning(f"Unsupported data type for saving artifacts: {type(itemsets_or_rules)}")
            return
            
        if df.empty:
            return

        # Add level information for context, especially useful for appended logs
        df['level'] = level

        if is_append_mode:
            # Append to the file, including header only if the file doesn't exist
            header = not os.path.exists(filepath)
            df.to_csv(filepath, mode='a', header=header, index=False)
            logger.info(f"Appended {len(df)} new {artifact_type} for turn {turn_counter}, level {level} to {filepath}")
        else:
            # Write a new file for itemsets
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {len(df)} {artifact_type} for turn {turn_counter}, level {level} to {filepath}")

    except Exception as e:
        # Use exc_info=True to log the full traceback for better debugging
        logger.error(f"Failed to save association artifact for turn {turn_counter}, level {level}: {e}", exc_info=True)
