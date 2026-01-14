import numpy as np
import time
import sys
import gc
import fcntl
import multiprocessing
from tqdm import tqdm
import importlib
import csv
import os
import os
import hashlib
from datetime import datetime
import pandas as pd
import joblib # Added for KMeans parallel backend
import itertools # Added for itertools.product
from kneed import KneeLocator # Added for KneeLocator
import logging # Added for logging

# Module imports
from Clustering_Method.cluster_normal_analyze import create_ratio_summary_10_bins

# Platform-specific import for file locking
import sys
IS_LINUX = sys.platform == "linux"
if IS_LINUX:
    try:
        import fcntl
    except ImportError:
        IS_LINUX = False
        print("[WARN] fcntl module not found, file locking in parallel writes will be disabled. This is expected on non-Linux systems.")


BASE_PROGRESS_DIR = os.environ.get("JACCARD_PROGRESS_DIR", os.path.join("..", "Dataset_ex", "progress_tracking"))

# === save path ===

def get_jaccard_elbow_progress_file_path(algorithm, data_hash):
    """Get the progress file path for Jaccard Elbow method."""
    #import os
    # Use relative path to go up one level to the parent directory
    #progress_dir = os.path.join("..", "Dataset_ex", "progress_tracking")
    """Get the progress file path for Jaccard Elbow method (respects JACCARD_PROGRESS_DIR env)."""
    progress_dir = BASE_PROGRESS_DIR
    os.makedirs(progress_dir, exist_ok=True)
    
    progress_file = os.path.join(progress_dir, f"jaccard_elbow_{data_hash}_{algorithm}_progress.csv")
    print(f"[DEBUG] Current working directory: {os.getcwd()}")
    print(f"[DEBUG] Progress directory: {os.path.abspath(progress_dir)}")
    print(f"[DEBUG] Progress file: {os.path.abspath(progress_file)}")
    return progress_file

def get_grid_search_progress_file_path(algorithm, data_hash):
    """Get the progress file path for Grid Search method."""
    #progress_dir = os.path.join("..", "Dataset_ex", "progress_tracking")
    """Get the progress file path for Grid Search method (respects JACCARD_PROGRESS_DIR env)."""
    progress_dir = BASE_PROGRESS_DIR
    os.makedirs(progress_dir, exist_ok=True)
    return os.path.join(progress_dir, f"jaccard_elbow_{data_hash}_{algorithm}_Grid_progress.csv")

def load_jaccard_elbow_progress(algorithm, data_hash):
    """Load completed parameter values from progress file with optimized I/O."""
    progress_file = get_jaccard_elbow_progress_file_path(algorithm, data_hash)
    completed_values = set()
    existing_scores = {}
    
    if os.path.exists(progress_file):
        try:
            # Read entire file at once for better performance
            with open(progress_file, 'r', newline='', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Skip header and process all lines efficiently
            for line in lines[1:]:  # Skip header
                row = line.strip().split(',')
                if len(row) >= 3: # Check for at least param, surrogate_score, jaccard_score
                    try:
                        param_value = float(row[0])
                        completed_values.add(param_value)
                        # Load scores from the row
                        surrogate_score_val = float(row[1])
                        jaccard_score_val = float(row[2])
                        existing_scores[param_value] = {'surrogate_score': surrogate_score_val, 'jaccard_score': jaccard_score_val}
                    except ValueError:
                        continue  # Skip invalid rows
            
            print(f"[Jaccard Elbow Progress] Loaded {len(completed_values)} completed parameter values from {progress_file}")
        except Exception as e:
            print(f"[Jaccard Elbow Progress] Error loading progress file {progress_file}: {e}")
    
    return completed_values, existing_scores


# === save progress ===

def save_jaccard_elbow_progress_parallel(algorithm, data_hash, k, surrogate_score, jaccard_score, ratio_distribution, nan_count=0, max_retries=3):
    """
    Safely append progress to a CSV file from multiple processes, 
    including ratio distribution in 10 bins, timestamp, and retry logic.
    This version is platform-aware and works on non-Linux systems.
    """
    progress_file = get_jaccard_elbow_progress_file_path(algorithm, data_hash)
    
    # Create fixed 10-bin summary for ratio distribution
    ratio_summary = create_ratio_summary_10_bins(ratio_distribution)
    
    # Define fixed header with 10 ratio bins and nan_count
    header = ['param_value', 'surrogate_score', 'jaccard_score', 'nan_count'] + list(ratio_summary.keys()) + ['timestamp']
    
    row_data = [k, surrogate_score, jaccard_score, nan_count] + list(ratio_summary.values()) + [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]

    for attempt in range(max_retries):
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(progress_file), exist_ok=True)
            
            with open(progress_file, 'a', newline='', encoding='utf-8') as f:
                if IS_LINUX: fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    # Check if file is empty to write header
                    f.seek(0, os.SEEK_END)
                    file_is_empty = f.tell() == 0
                    
                    writer = csv.writer(f)
                    
                    if file_is_empty:
                        writer.writerow(header)
                    
                    writer.writerow(row_data)
                    f.flush() # Ensure data is written to disk
                finally:
                    # Always release the lock
                    if IS_LINUX: fcntl.flock(f, fcntl.LOCK_UN)
            
            return # Success, exit the loop and function

        except Exception as e:
            if attempt < max_retries - 1:
                # Wait before retrying
                time.sleep(0.1 * (attempt + 1)) 
            else:
                # Log final failure
                print(f"Error saving progress for {algorithm} k={k} after {max_retries} attempts: {e}")

# In _evaluate_single_elbow_k, the call to save_jaccard_elbow_progress_parallel does not need to be changed
# as max_retries has a default value.

def safe_save_grid_progress(progress_file, param_str, jaccard_score, max_retries=3):
    """Safely save Grid Search progress with file locking (Linux)"""
    for attempt in range(max_retries):
        try:
            # Check if file exists to determine if we need to write header
            file_exists = os.path.exists(progress_file)
            
            with open(progress_file, 'a') as f:
                # Apply exclusive lock (Linux)
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    # Write header if file doesn't exist
                    if not file_exists:
                        f.write('param_str,jaccard_score,timestamp\n')
                    
                    # Write progress data
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f'"{param_str}",{jaccard_score},{timestamp}\n')
                    f.flush()  # Force write to disk
                finally:
                    # Release the lock
                    #pass
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return True
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                print(f"[Grid Search] Retry {attempt + 1}/{max_retries} for saving progress: {e}")
            else:
                print(f"[Grid Search] Failed to save progress after {max_retries} attempts: {e}")
                return False
    return False


def save_jaccard_elbow_progress(algorithm, param_value, jaccard_score, data_hash):
    """Save a completed parameter value to progress file."""
    progress_file = get_jaccard_elbow_progress_file_path(algorithm, data_hash)
    
    try:
        # Check if file exists to determine if we need to write header
        file_exists = os.path.exists(progress_file)
        
        with open(progress_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['param_value', 'jaccard_score', 'timestamp'])
            writer.writerow([param_value, jaccard_score, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            
    except Exception as e:
        print(f"[Jaccard Elbow Progress] Error saving progress to {progress_file}: {e}")
        import traceback
        traceback.print_exc()

def save_grid_search_progress(progress_file, param_str, score):
    """Safely save Grid Search progress with file locking (Linux)"""
    for attempt in range(3): # Increased retries for grid search
        try:
            # Check if file exists to determine if we need to write header
            file_exists = os.path.exists(progress_file)
            
            with open(progress_file, 'a') as f:
                # Apply exclusive lock (Linux)
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    writer = csv.writer(f)
                    # Write header if file doesn't exist or is empty
                    if not file_exists or os.path.getsize(progress_file) == 0:
                        writer.writerow(['param_str', 'score', 'timestamp'])
                    
                    # Write progress data
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    writer.writerow([param_str, score, timestamp])
                    f.flush()  # Force write to disk
                finally:
                    # Release the lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return True
            
        except Exception as e:
            if attempt < 2: # Retry up to 2 times
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                print(f"[Grid Search] Retry {attempt + 1}/3 for saving progress: {e}")
            else:
                print(f"[Grid Search] Failed to save progress after 3 attempts: {e}")
                return False
    return False