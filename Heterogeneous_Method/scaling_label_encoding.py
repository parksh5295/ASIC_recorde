import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from Heterogeneous_Method.Feature_Encoding import Heterogeneous_Feature_named_featrues
import numpy as np

def scaling_label_encoding(data, file_type):
    """
    Applies Label Encoding to categorical features and then MinMax Scaling to ALL features
    (both original numerical and encoded categorical) to ensure a consistent scale.
    """
    
    feature_dict = Heterogeneous_Feature_named_featrues(file_type)
    
    categorical_features = [col for col in feature_dict.get('categorical_features', []) if col in data.columns]
    time_features = [col for col in feature_dict.get('time_features', []) if col in data.columns]
    packet_length_features = [col for col in feature_dict.get('packet_length_features', []) if col in data.columns]
    count_features = [col for col in feature_dict.get('count_features', []) if col in data.columns]
    binary_features = [col for col in feature_dict.get('binary_features', []) if col in data.columns]

    numerical_features = time_features + packet_length_features + count_features
    
    processed_data = data.copy()

    # 1. Handle Categorical Features: Apply LabelEncoder
    if categorical_features:
        for col in categorical_features:
            le = LabelEncoder()
            processed_data[col] = le.fit_transform(processed_data[col].astype(str))

    # 2. Combine all features to be scaled
    # Now, all categorical features are numerical. We scale them along with the original numerical features.
    all_features_to_scale = numerical_features + categorical_features + binary_features
    
    # Ensure the list contains only unique column names that exist in the dataframe
    all_features_to_scale = sorted(list(set(col for col in all_features_to_scale if col in processed_data.columns)))

    # 3. Handle All Features: Apply MinMaxScaler to everything
    if all_features_to_scale:
        scaler = MinMaxScaler()
        # Fit and transform the data, then create a new DataFrame to preserve columns and index
        scaled_data = scaler.fit_transform(processed_data[all_features_to_scale])
        processed_data = pd.DataFrame(scaled_data, columns=all_features_to_scale, index=processed_data.index)
    
    # Legacy return values to match the expected signature from choose_heterogeneous_method
    feature_list = [categorical_features, time_features, packet_length_features, count_features, binary_features]
    category_mapping = {}
    data_list = []

    return processed_data, feature_list, category_mapping, data_list
