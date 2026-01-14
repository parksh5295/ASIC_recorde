#!/usr/bin/env python3
"""
Test script for Elbow method progress tracking
"""

import numpy as np
import os
from Tuning_hyperparameter.Elbow_method import Elbow_method

def test_elbow_progress():
    """Test Elbow method with progress tracking"""
    
    # Create test data
    np.random.seed(42)
    X = np.random.randn(1000, 5)  # 1000 samples, 5 features
    
    print("=== Testing Elbow Method Progress Tracking ===")
    print(f"Data shape: {X.shape}")
    
    # Test 1: First run
    print("\n--- Test 1: First run ---")
    result1 = Elbow_method(X, X, 'Kmeans', 10, num_processes_for_algo=2)
    print(f"Optimal k: {result1['optimal_cluster_n']}")
    
    # Test 2: Second run (should load progress)
    print("\n--- Test 2: Second run (should load progress) ---")
    result2 = Elbow_method(X, X, 'Kmeans', 10, num_processes_for_algo=2)
    print(f"Optimal k: {result2['optimal_cluster_n']}")
    
    # Test 3: Different data (should create new progress file)
    print("\n--- Test 3: Different data ---")
    X2 = np.random.randn(1000, 5)  # Different data
    result3 = Elbow_method(X2, X2, 'Kmeans', 10, num_processes_for_algo=2)
    print(f"Optimal k: {result3['optimal_cluster_n']}")
    
    # Check progress files
    print("\n--- Progress Files Created ---")
    progress_dir = "Dataset_ex/progress_tracking"
    if os.path.exists(progress_dir):
        files = [f for f in os.listdir(progress_dir) if f.startswith('elbow_')]
        for file in files:
            print(f"  {file}")
    else:
        print("  No progress directory found")

if __name__ == "__main__":
    test_elbow_progress()
