from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, silhouette_score
import numpy as np
import multiprocessing # Added for parallel processing

# Helper function to call the metric functions, useful for starmap
def _calculate_metric_wrapper(metric_func, *args):
    return metric_func(*args)

def accuracy_basic(t, p):
    metric = accuracy_score(t, p)
    return metric

def precision_basic(t, p, average):
    metric = precision_score(t, p, average=average, zero_division=0)
    return metric

def recall_basic(t, p, average):
    metric = recall_score(t, p, average=average, zero_division=0)
    return metric

def f1_basic(t, p, average):
    metric = f1_score(t, p, average=average, zero_division=0)
    return metric

def jaccard_basic(t, p, average):
    try:
        # print(f"  [jaccard_basic DEBUG] y_true (shape {t.shape}, uniques {np.unique(t, return_counts=True)}), y_pred (shape {p.shape}, uniques {np.unique(p, return_counts=True)}), average: {average}") # Detailed print
        metric = jaccard_score(t, p, average=average, zero_division=0)
        # print(f"  [jaccard_basic DEBUG] Score: {metric}")
        return metric
    except ValueError as ve:
        print(f"  [jaccard_basic ERROR] ValueError calculating Jaccard score (average={average}): {ve}")
        print(f"    y_true first 10: {t[:10]}, y_pred first 10: {p[:10]}")
        return -1.0 # Return a distinct error indicator
    except Exception as e:
        print(f"  [jaccard_basic ERROR] Unexpected error calculating Jaccard score (average={average}): {e}")
        print(f"    y_true first 10: {t[:10]}, y_pred first 10: {p[:10]}")
        return -1.0 # Return a distinct error indicator

def silhouette_basic(x_data, p):
    metric = silhouette_score(x_data, p) if len(set(p)) > 1 else np.nan
    return metric


def average_combination(t, p, average, x_data):
    all_metrics = {
        "accuracy" : accuracy_basic(t, p),
        "precision" : precision_basic(t, p, average),
        "recall" : recall_basic(t, p, average),
        "f1" : f1_basic(t, p, average),
        "jaccard" : jaccard_basic(t, p, average),
        "silhouette" : silhouette_basic(x_data, p)
    }
    return all_metrics

def average_combination_wos(t, p, average):
    all_metrics = {
        "accuracy" : accuracy_basic(t, p),
        "precision" : precision_basic(t, p, average),
        "recall" : recall_basic(t, p, average),
        "f1" : f1_basic(t, p, average),
        "jaccard" : jaccard_basic(t, p, average)
    }
    return all_metrics


def evaluate_clustering(y_true, y_pred, X_data):
    if y_true.size > 0:
        # Define tasks for parallel execution
        # Each task is (function_to_call, y_true, y_pred, average_method, X_data)
        # For average_combination_wos, X_data might be None or an empty placeholder if not used.
        tasks = [
            (average_combination, y_true, y_pred, 'macro', X_data),
            (average_combination, y_true, y_pred, 'micro', X_data),
            (average_combination, y_true, y_pred, 'weighted', X_data)
        ]
        
        results_dict = {}
        try:
            # Use a Pool for parallel execution. 
            # Limit processes if this function itself might be called in a parallelized loop.
            # For now, let's use a small number or cpu_count().
            # If evaluate_clustering is called many times, creating a Pool each time can be inefficient.
            # Consider passing a Pool object if this is part of a larger parallel structure.
            num_processes = min(len(tasks), multiprocessing.cpu_count()) # Avoid over-subscribing

            with multiprocessing.Pool(processes=num_processes) as pool:
                # starmap will unpack the arguments for each task
                # results will be a list of dictionaries returned by average_combination
                parallel_results = pool.starmap(_calculate_metric_wrapper, tasks)
            
            # Reconstruct the results dictionary
            results_dict["average=macro"] = parallel_results[0]
            results_dict["average=micro"] = parallel_results[1]
            results_dict["average=weighted"] = parallel_results[2]
            
        except Exception as e:
            print(f"Error during parallel clustering evaluation: {e}. Falling back to sequential.")
            # Fallback to sequential execution in case of error
            results_dict["average=macro"] = average_combination(y_true, y_pred, 'macro', X_data)
            results_dict["average=micro"] = average_combination(y_true, y_pred, 'micro', X_data)
            results_dict["average=weighted"] = average_combination(y_true, y_pred, 'weighted', X_data)
            
        return results_dict
    return {}

def evaluate_clustering_wos(y_true, y_pred):
    if y_true.size > 0:
        tasks = [
            (average_combination_wos, y_true, y_pred, 'macro'),
            (average_combination_wos, y_true, y_pred, 'micro'),
            (average_combination_wos, y_true, y_pred, 'weighted')
        ]
        results_dict = {}
        try:
            num_processes = min(len(tasks), multiprocessing.cpu_count())
            with multiprocessing.Pool(processes=num_processes) as pool:
                parallel_results = pool.starmap(_calculate_metric_wrapper, tasks)

            results_dict["average=macro"] = parallel_results[0]
            results_dict["average=micro"] = parallel_results[1]
            results_dict["average=weighted"] = parallel_results[2]

        except Exception as e:
            print(f"Error during parallel clustering evaluation (wos): {e}. Falling back to sequential.")
            results_dict["average=macro"] = average_combination_wos(y_true, y_pred, 'macro')
            results_dict["average=micro"] = average_combination_wos(y_true, y_pred, 'micro')
            results_dict["average=weighted"] = average_combination_wos(y_true, y_pred, 'weighted')
            
        return results_dict
    return {}