# Modules for bundling and using Heterogeneous_Methods
# Return: (embedded) data, feature_list

from Heterogeneous_Method.Non_act import Heterogeneous_Non_OneHotEncoder, Heterogeneous_Non_StandardScaler
from Heterogeneous_Method.Interval_Same import Heterogeneous_Interval_Same
from Heterogeneous_Method.Interval_normalized import Heterogeneous_Interval_Inverse
from Heterogeneous_Method.scaling_label_encoding import scaling_label_encoding


def choose_heterogeneous_method(data, file_type, het_method='Interval_inverse', regul='N', n_splits_override=None, existing_mapping=None):
    """
    Selects and applies a heterogeneous data processing method.
    This version combines the legacy methods with the new scaling_label_encoding method.
    """
    data_list = None
    category_mapping = None
    feature_list = None
    embedded_data = data

    # --- proposed method for clustering preprocessing ---
    if het_method == 'scaling_label_encoding':
        # This function returns 4 values, matching the expected signature.
        embedded_data, feature_list, category_mapping, data_list = scaling_label_encoding(data, file_type)

    # --- RESTORED: Legacy methods for association rule mining etc. ---
    elif het_method == 'Non_act':
        if Heterogeneous_Non_OneHotEncoder is None:
            raise NotImplementedError("Non_act methods are not available due to import error.")
        
        # This logic is interactive and not ideal, but preserved for compatibility.
        het_method_Non_act = str(input("Choose between OneHotEncoder and StandardScaler: "))
        if het_method_Non_act == 'OneHotEncoder':
            embedded_data, feature_list = Heterogeneous_Non_OneHotEncoder(data)
        elif het_method_Non_act == 'StandardScaler':
            embedded_data, feature_list = Heterogeneous_Non_StandardScaler(data)
        else:
            print("There are two choices: OneHotEncoder and StandardScaler. Please try again.")
            # Avoid infinite recursion by returning default values
            return embedded_data, feature_list, category_mapping, data_list

    elif het_method == 'Interval_same':
        if Heterogeneous_Interval_Same is None:
             raise NotImplementedError("Interval_same method is not available due to import error.")
        embedded_data, feature_list = Heterogeneous_Interval_Same(data, file_type)
        
    elif het_method in ['Interval_inverse', 'Interval_Inverse']:
        # Pass the existing_mapping argument down to the core function
        # print(f"[DEBUG] Input data NaN count before Heterogeneous_Interval_Inverse: {data.isnull().sum().to_dict()}")
        embedded_data, feature_list, category_mapping, data_list = Heterogeneous_Interval_Inverse(
            data, file_type, regul, 
            n_splits_override=n_splits_override, 
            existing_mapping=existing_mapping
        )
        # print(f"[DEBUG] Output embedded_data NaN count after Heterogeneous_Interval_Inverse: {embedded_data.isnull().sum().to_dict()}")
        
    # --- IMPROVED: Better error handling for unknown methods ---
    else:
        # Replaced the original recursive call with a clear error.
        raise ValueError(f"Unknown heterogeneous method specified: '{het_method}'")

    return embedded_data, feature_list, category_mapping, data_list
    # category_mapping: [0]->categorical features mapping, [1]->flag features mapping (Non Standard)
