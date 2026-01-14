import os
import glob
import logging

# Setup logger, so messages are formatted nicely if this is run.
# Using a basic config in case this script is ever run directly.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_chunk_temp_files():
    """
    Finds and deletes temporary files generated ONLY during the chunked virtual labeling process.
    These files are identified by the 'temp_chunk' keyword in their names, ensuring that
    main algorithm selection cache files are not affected.
    """
    logger.info("Starting cleanup of temporary chunk processing files...")
    
    # Path is relative to the main script's execution directory (ASIC-#/)
    progress_dir = os.path.join("..", "Dataset_ex", "progress_tracking")
    
    if not os.path.isdir(progress_dir):
        logger.warning(f"Progress directory not found at '{progress_dir}'. Nothing to clean.")
        return

    # Glob patterns specifically targeting the temporary chunk files
    patterns_to_delete = [
        os.path.join(progress_dir, "jaccard_elbow_temp_chunk_*.csv"),
        os.path.join(progress_dir, "chunk_diagnostics_temp_chunk_*.csv")
    ]
    
    deleted_count = 0
    total_found = 0
    
    for pattern in patterns_to_delete:
        files_found = glob.glob(pattern)
        total_found += len(files_found)
        
        for file_path in files_found:
            try:
                os.remove(file_path)
                deleted_count += 1
                # Use a simpler print statement for command-line clarity
                print(f"  Removed: {os.path.basename(file_path)}")
            except OSError as e:
                logger.warning(f"  Could not remove {os.path.basename(file_path)}: {e}")
    
    if total_found == 0:
        print("No temporary chunk files found to delete.")
    else:
        print(f"Cleanup complete. Found {total_found} files and successfully removed {deleted_count} of them.")
