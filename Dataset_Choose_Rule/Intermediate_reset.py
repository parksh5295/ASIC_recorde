import os
import glob


def reset_intermediate_files(file_type, file_number):
    """Delete all intermediate progress files for a fresh start"""
    
    # Generate data hash for this specific dataset
    from utils.generate_data_hash import generate_stable_data_hash
    # We need to get X.shape, but we don't have it yet, so use get_existing_hash if available
    from utils.generate_data_hash import get_existing_hash_for_file_type
    
    try:
        data_hash = get_existing_hash_for_file_type(file_type, file_number)
    except:
        # Fallback: delete all progress files (old behavior)
        data_hash = "*"
    
    # Patterns for different types of progress files (now hash-specific)
    patterns_to_delete = [
        # Elbow method progress files
        f"../Dataset_ex/progress_tracking/elbow_{data_hash}_*_progress.csv",
        # Grid search progress files  
        f"../Dataset_ex/progress_tracking/grid_search_{data_hash}_*_progress.csv",
        # Jaccard Elbow method progress files (hash-specific)
        f"../Dataset_ex/progress_tracking/jaccard_elbow_{data_hash}_*_progress.csv",
        f"../Dataset_ex/progress_tracking/jaccard_elbow_{data_hash}_*_Grid_progress.csv",
        # Threshold optimization files
        f"../Dataset_ex/progress_tracking/*_threshold_optimization.csv",
        # Algorithm results files
        f"../Dataset_ex/progress_tracking/*_results.csv",
        # Summary files
        f"../Dataset_ex/progress_tracking/*_summary.txt",
        # Mapping files (to force regeneration with new feature configuration)
        f"../Dataset_ex/mapping_info/*{file_type}*_Interval_inverse_mapping.pkl",
        f"../Dataset_ex/mapping_info/*{file_type}*_Interval_scalers.pkl"
    ]
    
    deleted_files = []
    
    for pattern in patterns_to_delete:
        matching_files = glob.glob(pattern)
        for file_path in matching_files:
            try:
                os.remove(file_path)
                deleted_files.append(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Warning: Could not delete {file_path}: {e}")
    
    if deleted_files:
        print(f"Successfully deleted {len(deleted_files)} intermediate files")
    else:
        print("No intermediate files found to delete")