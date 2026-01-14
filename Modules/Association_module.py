# Modules to help you determine association rules
# Return: A list containing dictionaries with feature names and corresponding group numbers (each dictionary will be a signature).

from Association_Rule.Conditional_Probability import conditional_probability
from Association_Rule.Apriori import Apriori_rule
from Association_Rule.FPGrowth import FPGrowth_rule
from Association_Rule.Eclat import eclat
from Association_Rule.RARM import rarm
from Association_Rule.H_mine import h_mine
from Association_Rule.OPUS import opus
from Association_Rule.SaM import sam
from Association_Rule.FPC import fpc
import time


# The hunter association rule is generated.
def association_module(df, association_rule_choose, min_support=0.5, min_confidence=0.8, association_metric='confidence', 
                         num_processes=None, file_type_for_limit=None, max_level_limit=None, itemset_limit=None, 
                         turn_counter=None, params_str=None, enable_dynamic_support=False, 
                         dynamic_support_threshold=500000, support_increment_factor=1.2, **kwargs):
    """
    Acts as a router to call the chosen association rule mining algorithm.
    It accepts all possible parameters and passes them down.
    """
    start_time = time.time()
    
    # Consolidate all parameters into a single dictionary to pass to the algorithms
    all_params = {
        'min_support': min_support,
        'min_confidence': min_confidence,
        'association_metric': association_metric,
        'num_processes': num_processes,
        'file_type_for_limit': file_type_for_limit,
        'max_level_limit': max_level_limit,
        'itemset_limit': itemset_limit,
        'turn_counter': turn_counter,
        'params_str': params_str,
        'enable_dynamic_support': enable_dynamic_support,
        'dynamic_support_threshold': dynamic_support_threshold,
        'support_increment_factor': support_increment_factor
    }
    # Add any other kwargs that might have been passed
    all_params.update(kwargs)

    print(f"  [Debug] Association Module Start: Algorithm='{association_rule_choose}', Shape={df.shape}, Params={all_params}")

    algorithm_map = {
        'rarm': rarm,
        'sam': sam,
        'opus': opus,
        'h_mine': h_mine,
        'eclat': eclat,
        'fpc': fpc
    }

    chosen_algorithm = algorithm_map.get(association_rule_choose.lower())

    association_list, max_level_reached = [], 0
    if chosen_algorithm:
        print(f"  [Debug]     Calling {association_rule_choose} function...")
        
        # --- FIX: The custom algorithms don't use 'association_metric'. ---
        # We pop it from the params dict before unpacking to avoid a TypeError.
        all_params.pop('association_metric', None)
        
        # The chosen algorithm will only use the parameters it recognizes from all_params
        association_list, max_level_reached = chosen_algorithm(df, **all_params)
    else:
        print("The name of the association rule appears to be incorrect.")
        
    end_time = time.time()
    print(f"  [Debug] Association Module End: Found {len(association_list)} rules in {end_time - start_time:.2f} seconds.")

    return association_list, max_level_reached # dictionary
