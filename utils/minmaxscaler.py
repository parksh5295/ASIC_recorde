import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import pickle


def apply_minmax_scaling_and_save_scalers(group_mapped_df, file_type, file_number, heterogeneous_method, base_output_dir="results"):
    """
    Applies MinMaxScaler to each column of the input DataFrame, saves the scalers,
    and returns the scaled DataFrame.

    Args:
        group_mapped_df (pd.DataFrame): DataFrame with group IDs to be scaled.
        file_type (str): Type of the data file (e.g., 'MiraiBotnet').
        file_number (int): Number of the data file.
        heterogeneous_method (str): Heterogeneous method used (e.g., 'Interval_inverse').
        base_output_dir (str): Base directory to save scaler files.

    Returns:
        tuple: (pd.DataFrame, str)
               - Scaled DataFrame.
               - Path where the scalers were saved, or None if saving failed or df was empty.
    """
    if group_mapped_df.empty:
        print("Warning: Input DataFrame for scaling is empty. Returning empty DataFrame.")
        return pd.DataFrame(index=group_mapped_df.index), None

    # Prepare paths for saving scalers
    try:
        file_type_dir = os.path.join(base_output_dir, file_type)
        os.makedirs(file_type_dir, exist_ok=True) # Ensure directory exists
        scaler_filename = f"{file_type}_{file_number}_{heterogeneous_method}_scalers.pkl"
        scaler_save_path = os.path.join(file_type_dir, scaler_filename)
    except Exception as e:
        print(f"Error creating scaler save path: {e}. Scalers will not be saved.")
        scaler_save_path = None

    scaled_features_list = []
    feature_scalers = {}
    original_index = group_mapped_df.index

    for column in group_mapped_df.columns:
        scaler = MinMaxScaler()
        feature_values = group_mapped_df[column].values.reshape(-1, 1)
        scaled_values = scaler.fit_transform(feature_values)
        scaled_feature_series = pd.Series(scaled_values.flatten(), name=column, index=original_index)
        scaled_features_list.append(scaled_feature_series)
        feature_scalers[column] = scaler

    X_scaled_df = pd.concat(scaled_features_list, axis=1)

    if scaler_save_path and feature_scalers:
        try:
            with open(scaler_save_path, 'wb') as f:
                pickle.dump(feature_scalers, f)
            print(f"Feature scalers saved to {scaler_save_path}")
        except Exception as e:
            print(f"Error saving scalers to {scaler_save_path}: {e}")
            # Attempt fallback saving
            try:
                fallback_path = f"{file_type}_{file_number}_{heterogeneous_method}_scalers_fallback.pkl"
                with open(fallback_path, 'wb') as f:
                    pickle.dump(feature_scalers, f)
                print(f"Feature scalers saved to fallback path: {fallback_path}")
                scaler_save_path = fallback_path # Update path to fallback if successful
            except Exception as fe:
                print(f"Error saving scalers to fallback path {fallback_path}: {fe}")
                scaler_save_path = None # Indicate saving failed
    elif not feature_scalers:
        print("No features were scaled, so no scalers to save.")
        scaler_save_path = None
    else: # scaler_save_path was None
        print("Scaler save path was not properly configured. Scalers were not saved.")
        scaler_save_path = None


    return X_scaled_df, scaler_save_path
