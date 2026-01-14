#!/bin/bash

###############################################################################
# Server Environment Diagnostic Script
# Purpose: Compare ML/NumPy configurations across different servers
# Usage: ./diagnose_server_environment.sh > server_report.txt
###############################################################################

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                Server Environment Diagnostic Report                   ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo
echo "Server Hostname: $(hostname)"
echo "Report Date: $(date)"
echo "User: $(whoami)"
echo

###############################################################################
# 1. System Information
###############################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. SYSTEM INFORMATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
echo "Operating System:"
uname -a
echo
echo "CPU Information:"
if command -v lscpu &> /dev/null; then
    lscpu | grep -E "Model name|Architecture|Thread|Core|Socket|CPU\(s\):"
else
    echo "lscpu not available"
fi
echo
echo "Memory Information:"
if command -v free &> /dev/null; then
    free -h
else
    echo "free command not available"
fi
echo

###############################################################################
# 2. Python Environment
###############################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. PYTHON ENVIRONMENT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
echo "Python Executable:"
which python3
echo
echo "Python Version:"
python3 --version
echo
echo "Python Path:"
python3 -c "import sys; print('\n'.join(sys.path))"
echo

###############################################################################
# 3. Key Library Versions
###############################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. KEY LIBRARY VERSIONS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
python3 << 'EOF'
import sys
print(f"Python:        {sys.version.split()[0]}")

try:
    import numpy as np
    print(f"NumPy:         {np.__version__}")
except ImportError:
    print("NumPy:         NOT INSTALLED")

try:
    import scipy
    print(f"SciPy:         {scipy.__version__}")
except ImportError:
    print("SciPy:         NOT INSTALLED")

try:
    import sklearn
    print(f"Scikit-learn:  {sklearn.__version__}")
except ImportError:
    print("Scikit-learn:  NOT INSTALLED")

try:
    import pandas as pd
    print(f"Pandas:        {pd.__version__}")
except ImportError:
    print("Pandas:        NOT INSTALLED")

try:
    import joblib
    print(f"Joblib:        {joblib.__version__}")
except ImportError:
    print("Joblib:        NOT INSTALLED")
EOF
echo

###############################################################################
# 4. NumPy BLAS/LAPACK Configuration (CRITICAL!)
###############################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. NumPy BLAS/LAPACK CONFIGURATION ⭐ CRITICAL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
python3 << 'EOF'
import numpy as np
print("NumPy Installation Path:")
print(f"  {np.__file__}")
print()
print("BLAS/LAPACK Configuration:")
print("─" * 70)
np.show_config()
EOF
echo

###############################################################################
# 5. Environment Variables
###############################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5. RELEVANT ENVIRONMENT VARIABLES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
echo "Threading & BLAS Variables:"
env | grep -E "OMP_NUM_THREADS|MKL_NUM_THREADS|OPENBLAS_NUM_THREADS|BLIS_NUM_THREADS|VECLIB_MAXIMUM_THREADS|NUMEXPR" | sort || echo "  None set"
echo
echo "CUDA/GPU Variables:"
env | grep -E "CUDA|GPU" | sort || echo "  None set"
echo
echo "LD_LIBRARY_PATH:"
echo "${LD_LIBRARY_PATH:-Not set}"
echo

###############################################################################
# 6. GPU/CUDA Status
###############################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "6. GPU/CUDA STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU Detected:"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
    echo
    echo "CUDA Version:"
    nvidia-smi | grep "CUDA Version"
else
    echo "No NVIDIA GPU detected or nvidia-smi not available"
fi
echo
python3 << 'EOF'
try:
    import torch
    print(f"PyTorch Installed: Yes (version {torch.__version__})")
    print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"PyTorch CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
except ImportError:
    print("PyTorch: Not installed")
except Exception as e:
    print(f"PyTorch check error: {e}")
EOF
echo

###############################################################################
# 7. OpenMP Thread Configuration
###############################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "7. OpenMP THREAD CONFIGURATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
python3 << 'EOF'
try:
    from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
    print(f"Scikit-learn effective OpenMP threads: {_openmp_effective_n_threads()}")
except Exception as e:
    print(f"Cannot determine OpenMP threads: {e}")

import os
print(f"OMP_NUM_THREADS env var: {os.environ.get('OMP_NUM_THREADS', 'Not set')}")
print(f"MKL_NUM_THREADS env var: {os.environ.get('MKL_NUM_THREADS', 'Not set')}")
EOF
echo

###############################################################################
# 8. GMM Reproducibility Test (MOST IMPORTANT!)
###############################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "8. GMM REPRODUCIBILITY TEST ⭐⭐⭐ MOST IMPORTANT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
echo "Testing if GMM produces identical results with same random seed..."
echo "This directly tests the behavior causing your server differences."
echo
python3 << 'EOF'
import numpy as np
from sklearn.mixture import GaussianMixture

# Create test data
np.random.seed(42)
X = np.random.randn(1000, 10)

print("Running GMM 5 times with identical random_state=42:")
print("─" * 70)

results = []
for run in range(5):
    np.random.seed(42)  # Reset seed each time
    gmm = GaussianMixture(
        n_components=5, 
        random_state=42, 
        covariance_type='diag', 
        reg_covar=1e-5
    )
    gmm.fit(X)
    score = gmm.score(X)
    results.append(score)
    print(f"  Run {run+1}: Log-likelihood = {score:.15f}")

print()
print("Analysis:")
print("─" * 70)

# Check if all results are identical
unique_results = set([f"{r:.12f}" for r in results])
if len(unique_results) == 1:
    print("✅ REPRODUCIBLE: All runs produced IDENTICAL results")
    print("   This server should give consistent results.")
else:
    print("❌ NOT REPRODUCIBLE: Results differ across runs!")
    print(f"   Number of unique results: {len(unique_results)}")
    print(f"   Maximum difference: {max(results) - min(results):.15e}")
    print(f"   Standard deviation: {np.std(results):.15e}")
    print()
    print("   ⚠️  This is the likely cause of server differences!")
    print("   Possible causes:")
    print("   - Different BLAS libraries (MKL vs OpenBLAS)")
    print("   - Multi-threaded BLAS causing non-determinism")
    print("   - Different CPU architectures")
    print("   - CUDA/GPU interference")

print()
print("Detailed Results:")
for i, r in enumerate(results, 1):
    print(f"  Run {i}: {r:.18f}")
EOF
echo

###############################################################################
# 9. NumPy Random Number Generator Test
###############################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "9. NumPy RANDOM NUMBER GENERATOR TEST"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
python3 << 'EOF'
import numpy as np

print("Testing NumPy random number generation reproducibility:")
print("─" * 70)

results = []
for run in range(3):
    np.random.seed(42)
    arr = np.random.randn(10)
    results.append(arr.sum())
    print(f"Run {run+1}: Sum = {arr.sum():.15f}, First 3 values = {arr[:3]}")

print()
if len(set([f"{r:.12f}" for r in results])) == 1:
    print("✅ NumPy RNG is reproducible")
else:
    print("❌ NumPy RNG is NOT reproducible (very rare!)")
EOF
echo

###############################################################################
# 10. BLAS Detection Detail
###############################################################################
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "10. DETAILED BLAS DETECTION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
python3 << 'EOF'
import numpy as np

print("Attempting to detect BLAS backend:")
print("─" * 70)

# Method 1: Check __config__
try:
    from numpy.__config__ import show
    show()
except:
    print("Cannot access numpy.__config__.show()")

print()

# Method 2: Check for linked libraries
try:
    import numpy.core._multiarray_umath as mu
    print(f"NumPy core module file: {mu.__file__}")
except Exception as e:
    print(f"Cannot access NumPy core module: {e}")

print()

# Method 3: Try to detect from file
try:
    import subprocess
    numpy_lib = np.__file__.replace('__init__.py', 'core/_multiarray_umath.cpython-*.so')
    result = subprocess.run(['ldd', numpy_lib], capture_output=True, text=True, shell=True)
    if result.returncode == 0:
        print("Linked libraries (ldd output):")
        for line in result.stdout.split('\n'):
            if any(x in line.lower() for x in ['blas', 'lapack', 'mkl', 'openblas', 'atlas']):
                print(f"  {line.strip()}")
except Exception as e:
    print(f"Cannot run ldd: {e}")
EOF
echo

###############################################################################
# Summary
###############################################################################
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                           DIAGNOSIS COMPLETE                          ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo
echo "NEXT STEPS:"
echo "1. Run this script on BOTH servers"
echo "2. Compare Section 8 (GMM Reproducibility Test) results"
echo "3. Compare Section 4 (BLAS/LAPACK Configuration)"
echo "4. If GMM is NOT reproducible, the server has non-deterministic behavior"
echo
echo "To save this report:"
echo "  ./diagnose_server_environment.sh > server_$(hostname)_report.txt"
echo

