from tqdm import tqdm
import importlib
import numpy as np
import time
import sys
import gc
import fcntl
import csv
import os
import hashlib
from datetime import datetime
import pandas as pd
import joblib # Added for KMeans parallel backend
import itertools # Added for itertools.product
from kneed import KneeLocator # Added for KneeLocator
import logging # Added for logging


# Progress tracking functions for Jaccard Elbow Method
def generate_data_hash(X):
    """Generate a unique hash for the dataset to distinguish different datasets."""
    try:
        # Create a hash based on dataset characteristics (more stable)
        # Use only shape and dtype which are most stable across runs
        data_info = f"{X.shape}_{X.dtype}"
        return hashlib.md5(data_info.encode()).hexdigest()[:8]
    except Exception:
        # Fallback to a simple hash if there's any issue
        return hashlib.md5(str(X.shape).encode()).hexdigest()[:8]

def generate_stable_data_hash(file_type, file_number, X_shape=None):
    """Generate a stable hash based on file information instead of data content."""
    try:
        # Use file information that doesn't change between runs
        # Convert file_number to string to avoid concatenation error
        if X_shape is not None:
            file_info = f"{file_type}_{str(file_number)}_{X_shape}"
        else:
            file_info = f"{file_type}_{str(file_number)}"
        return hashlib.md5(file_info.encode()).hexdigest()[:8]
    except Exception:
        # Fallback to a simple hash if there's any issue
        return hashlib.md5(f"{file_type}_{str(file_number)}".encode()).hexdigest()[:8]

# Temp chunk hash generation for chunk processing
def generate_temp_chunk_hash(file_type, chunk_file_number):
    """
    Generates a unique and clearly identifiable hash for temporary chunk processing files.
    These files are intended to be deleted after the chunked processing is complete.
    The hash does NOT include shape, as it's for temporary files.
    """
    file_info = f"{file_type}_{str(chunk_file_number)}"
    # Prepending 'temp_chunk_' makes these files easy to identify and safely delete.
    return f"temp_chunk_{hashlib.md5(file_info.encode()).hexdigest()[:8]}"


def get_existing_hash_for_file_type(file_type, file_number):
    """Get existing hash for specific file_type and file_number to reuse progress files."""
    # Map common file types to their existing hashes
    # NOTE: These hashes are generated WITH X_shape, so they are specific to data dimensions
    # Format: MD5("{file_type}_{file_number}_{X_shape}")[:8]
    hash_mapping = {
        "Kitsune_1": "6fd980d7",  # Actual hash from progress files (verified)
        "DARPA98_1": "d35ca016",  # Actual hash (needs verification)
        "MiraiBotnet_1": "4e9f44a4",  # MiraiBotnet hash (verified)
        "CICIoT2023_1": "797880c8",  # CICIoT2023 hash (needs verification)
        "CICIDS2017_1": "28cf32ab",  # CICIDS2017 hash (verified)
        "NSL-KDD_1": "2e5011c4",  # NSL-KDD hash (verified)
        "netML_1": "af12983c",  # netML hash (verified)
        "CICModbus23_1": "b3f6c8d4",  # MD5("CICModbus23_1")[:8] - pre-calculated
        # Add more mappings as needed
    }
    
    # Convert file_number to string to avoid concatenation error
    try:
        key = f"{file_type}_{str(file_number)}"
        if key in hash_mapping:
            hash_value = hash_mapping[key]
            if hash_value is not None:
                return hash_value
            else:
                # Hash not yet determined, fallback to generation
                return generate_stable_data_hash(file_type, file_number)
        else:
            # Fallback to new hash generation
            return generate_stable_data_hash(file_type, file_number)
    except Exception as e:
        print(f"[DEBUG] Error in get_existing_hash_for_file_type: {e}")
        print(f"[DEBUG] file_type: {file_type}, file_number: {file_number}, type: {type(file_number)}")
        # Fallback to new hash generation
        return generate_stable_data_hash(file_type, file_number)