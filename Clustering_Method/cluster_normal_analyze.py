import numpy as np


def create_ratio_summary_10_bins(ratio_distribution):
    """Create summary statistics for ratio distribution in 10 bins (0.0-0.1, 0.1-0.2, ..., 0.9-1.0)"""
    
    # Initialize 10 bins
    bins = {
        'ratio_0.0-0.1': 0, 'ratio_0.1-0.2': 0, 'ratio_0.2-0.3': 0, 'ratio_0.3-0.4': 0, 'ratio_0.4-0.5': 0,
        'ratio_0.5-0.6': 0, 'ratio_0.6-0.7': 0, 'ratio_0.7-0.8': 0, 'ratio_0.8-0.9': 0, 'ratio_0.9-1.0': 0
    }
    
    if not ratio_distribution:
        return bins
    
    # Distribute ratios into bins
    for ratio, count in ratio_distribution.items():
        if ratio < 0.1:
            bins['ratio_0.0-0.1'] += count
        elif ratio < 0.2:
            bins['ratio_0.1-0.2'] += count
        elif ratio < 0.3:
            bins['ratio_0.2-0.3'] += count
        elif ratio < 0.4:
            bins['ratio_0.3-0.4'] += count
        elif ratio < 0.5:
            bins['ratio_0.4-0.5'] += count
        elif ratio < 0.6:
            bins['ratio_0.5-0.6'] += count
        elif ratio < 0.7:
            bins['ratio_0.6-0.7'] += count
        elif ratio < 0.8:
            bins['ratio_0.7-0.8'] += count
        elif ratio < 0.9:
            bins['ratio_0.8-0.9'] += count
        else:
            bins['ratio_0.9-1.0'] += count
    
    return bins