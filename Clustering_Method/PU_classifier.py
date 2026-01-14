# In best_clustering_selector_parallel.py

# ... (기존 코드) ...

from sklearn.linear_model import LogisticRegression # Add import
from sklearn.ensemble import RandomForestClassifier # Add import
from sklearn.calibration import CalibratedClassifierCV # Add import

def train_pu_classifier(X, known_normal_idx):
    """Trains a calibrated PU classifier."""
    # 1. Prepare data
    y_pu = np.zeros(len(X))  # All unlabeled
    y_pu[known_normal_idx] = 1 # Positives are 1
    X_pu = X

    # 2. Train classifier
    base_classifier = LogisticRegression(solver='liblinear', random_state=42) # Or RandomForestClassifier
    calibrated_classifier = CalibratedClassifierCV(base_classifier, method='isotonic', cv=5)
    calibrated_classifier.fit(X_pu, y_pu)

    return calibrated_classifier

def apply_pu_threshold(X, labels, known_normal_idx, calibrated_classifier):
    """Applies a threshold to PU classifier outputs to determine final labels."""
    # 1. Predict probabilities
    predicted_probs = calibrated_classifier.predict_proba(X)[:, 1] # Probability of being positive

    # 2. Use a simple threshold (e.g., 0.5) - Or use Elkan & Noto
    threshold = 0.5
    final_labels = (predicted_probs >= threshold).astype(int) # 1 for positive, 0 for negative
    return final_labels


# ... (기존 코드) ...


'''
# Step 10: Evaluating CNI thresholds for '{best_algorithm_name}' on the full dataset...
# (기존 CNI 평가 코드)

# --- PU Classifier Refinement ---
print("\n[PU Classifier] Starting PU classifier refinement...")
pu_classifier = train_pu_classifier(data_for_clustering, consistent_known_normal_indices)
final_cluster_labels = apply_pu_threshold(data_for_clustering, raw_cluster_labels, consistent_known_normal_indices, pu_classifier)
print("[PU Classifier] PU classifier refinement complete.")
# --- End PU Classifier Refinement ---

# (기존 CNI 평가 및 후처리 코드)
'''