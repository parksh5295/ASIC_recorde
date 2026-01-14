from definition.Anomal_Judgment import anomal_judgment_nonlabel


def get_label_columns_to_exclude(file_type):
    """
    Returns a list of label-related columns that should be excluded from signature generation.
    These columns are ONLY used for evaluation, NEVER for rule generation.
    Common columns like 'label' and 'cluster' are always excluded.
    Dataset-specific label columns are also included.
    
    Args:
        file_type (str): The dataset type (e.g., 'MiraiBotnet', 'CICIDS2017', etc.)
    
    Returns:
        list: List of column names that should be excluded from signature generation
    """
    # Common columns that should always be excluded
    label_cols = ['label', 'cluster', 'adjusted_cluster']
    
    # Dataset-specific label columns (NEVER include these in signatures)
    if file_type == 'MiraiBotnet':
        label_cols.extend(['reconnaissance', 'infection', 'action'])
    elif file_type in ['DARPA98', 'DARPA']:
        label_cols.extend(['Flag', 'Class'])
    elif file_type in ['CICIDS2017', 'CICIDS']:
        label_cols.append('Label')
    elif file_type in ['CICIoT2023', 'CICIoT']:
        label_cols.extend(['attack_name', 'attack_flag', 'attack_step'])
    elif file_type == 'netML':
        label_cols.append('Label')
    elif file_type in ['NSL-KDD', 'NSL_KDD']:
        # NSL-KDD has 'class' column as a label column
        label_cols.extend(['class', 'flag'])
    elif file_type in ['CICModbus23', 'CICModbus']:
        label_cols.append('Attack')
    elif file_type in ['IoTID20', 'IoTID']:
        label_cols.append('Label')
    elif file_type == 'Kitsune':
        label_cols.append('Label')
    
    return label_cols


def anomal_class_data(data):
    anomal_rows = data[data['label'] == 1]
    return anomal_rows

def nomal_class_data(data):
    nomal_rows = data[data['label'] == 0]
    return nomal_rows

def without_label(data):
    data = data.drop(columns='label')
    return data

'''
def without_labelmaking_out(data_type, data):
    r, data_line = anomal_judgment_nonlabel(data_type, data)    # data_line: 'Column' name to determine label
    # r is the output of the anomalous judgment function, an argument to receive the value of data[label]. It is not used in this function.
    data = without_label(data)
    data = data.drop(columns=data_line)
    return data
'''

def without_labelmaking_out(arg1, arg2):
    """
    Historical helper that exists in two calling styles:
      1) without_labelmaking_out(file_type: str, df: DataFrame)
         - Uses anomal_judgment_nonlabel(...) to determine which column(s) to drop.
      2) without_labelmaking_out(df: DataFrame, columns_to_drop: list-like)
         - Directly drops the provided columns. This style is used by newer batch scripts.
    """
    # Style 1: (file_type, dataframe)
    if isinstance(arg1, str):
        data_type, data = arg1, arg2
        _, data_line = anomal_judgment_nonlabel(data_type, data)
        data = without_label(data)
        if isinstance(data_line, (list, tuple)):
            drop_cols = list(data_line)
        else:
            drop_cols = [data_line]
        return data.drop(columns=drop_cols, errors='ignore')

    # Style 2: (dataframe, columns_to_drop)
    data, columns_to_drop = arg1, arg2
    if columns_to_drop is None:
        columns_to_drop = []
    elif not isinstance(columns_to_drop, (list, tuple, set)):
        columns_to_drop = [columns_to_drop]
    return data.drop(columns=['label', *columns_to_drop], errors='ignore')