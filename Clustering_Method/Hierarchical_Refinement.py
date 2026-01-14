# In best_clustering_selector_parallel.py

# ... (기존 코드) ...

from sklearn.cluster import KMeans # Added import
from scipy.stats import hypergeom # import
import numpy as np # import

def local_recluster(X, labels, cluster_to_refine, n_sub=3):
    """Refines a specific cluster by reclustering it into subclusters."""
    idx = np.where(labels == cluster_to_refine)[0]
    if len(idx) < 10:
        return labels # Skip if too few points in the cluster
    sub = X[idx]
    try:
        kmeans = KMeans(n_clusters=n_sub, random_state=42, n_init=10)
        sub_labels = kmeans.fit_predict(sub)
    except Exception as e:
        print(f"[local_recluster] KMeans failed: {e}")
        return labels # Return original labels on failure

    max_label = labels.max()
    new_labels = labels.copy()
    for k in range(n_sub):
        max_label += 1
        new_labels[idx[sub_labels==k]] = max_label
    return new_labels

def compute_cluster_purity(labels, known_normal_idx):
    """Computes the purity of each cluster based on known normals."""
    cluster_ids = np.unique(labels)
    purity = {}
    for cluster_id in cluster_ids:
        cluster_indices = np.where(labels == cluster_id)[0]
        num_known_normal = np.sum(np.isin(cluster_indices, known_normal_idx))
        purity[cluster_id] = num_known_normal / len(cluster_indices) if len(cluster_indices) > 0 else 0.0
    return purity

# ... (기존 코드) ...

# Step 10: Evaluating CNI thresholds for '{best_algorithm_name}' on the full dataset...
# (기존 CNI 평가 코드)

'''
# --- Local Reclustering ---
print("\n[Local Reclustering] Starting local reclustering refinement...")
cluster_purity = compute_cluster_purity(raw_cluster_labels, consistent_known_normal_indices)
# print(f"[Local Reclustering] Cluster purities: {cluster_purity}")

# Select impure clusters (example: 0.3 < purity < 0.7)
impure_clusters = [c for c, purity in cluster_purity.items() if 0.3 < purity < 0.7]
print(f"[Local Reclustering] Impure clusters to refine: {impure_clusters}")

# Apply local reclustering to each impure cluster
for cluster_to_refine in impure_clusters:
    raw_cluster_labels = local_recluster(data_for_clustering, raw_cluster_labels, cluster_to_refine, n_sub=3)
    print(f"[Local Reclustering] Refined cluster: {cluster_to_refine}")

print("[Local Reclustering] Local reclustering refinement complete.")
# --- End Local Reclustering ---

# (기존 CNI 평가 및 후처리 코드)
'''