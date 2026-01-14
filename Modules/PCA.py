# For PCA before main clustering
# Input data is usally X; after that Hstack processing
# Output data is usally 'X_reduced'

import os
from sklearn.decomposition import PCA


def pca_func(data, number_of_components=10, state=42):  # default; n_components=10, state=42
    # Force single-threaded execution for PCA to ensure reproducibility across servers
    # This prevents BLAS threading from causing different SVD results
    old_omp = os.environ.get('OMP_NUM_THREADS')
    old_mkl = os.environ.get('MKL_NUM_THREADS')
    old_openblas = os.environ.get('OPENBLAS_NUM_THREADS')
    
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    try:
        # Use 'full' SVD solver for better reproducibility across different CPU architectures
        # 'randomized' solver can give slightly different results on different hardware
        pca = PCA(n_components=number_of_components, random_state=state, svd_solver='full')
        data_reduced = pca.fit_transform(data)
    finally:
        # Restore original environment variables
        if old_omp is not None:
            os.environ['OMP_NUM_THREADS'] = old_omp
        else:
            os.environ.pop('OMP_NUM_THREADS', None)
            
        if old_mkl is not None:
            os.environ['MKL_NUM_THREADS'] = old_mkl
        else:
            os.environ.pop('MKL_NUM_THREADS', None)
            
        if old_openblas is not None:
            os.environ['OPENBLAS_NUM_THREADS'] = old_openblas
        else:
            os.environ.pop('OPENBLAS_NUM_THREADS', None)
    
    return data_reduced