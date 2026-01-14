import pandas as pd

from definition.Anomal_Judgment import anomal_judgment_nonlabel, anomal_judgment_label


def apply_labeling_logic(data, file_type):
    """Apply labeling logic based on file type - reusable function"""
    if file_type in ['MiraiBotnet', 'NSL-KDD', 'NSL_KDD']:
        data['label'], _ = anomal_judgment_nonlabel(file_type, data)
    elif file_type == 'netML':
        data['label'] = data['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
    elif file_type == 'DARPA98':
        data['label'] = data['Class'].apply(lambda x: 0 if str(x).strip() == '-' else 1)
    elif file_type in ['CICIDS2017', 'CICIDS']:
        if 'Label' in data.columns:
            data['label'] = data['Label'].apply(lambda x: 0 if str(x).strip() == 'BENIGN' else 1)
        else:
            logger.error("'Label' column not found in data")
            data['label'] = 0
    elif file_type in ['CICModbus23', 'CICModbus']:
        data['label'] = data['Attack'].apply(lambda x: 0 if str(x).strip() == 'Baseline Replay: In position' else 1)
    elif file_type in ['IoTID20', 'IoTID']:
        data['label'] = data['Label'].apply(lambda x: 0 if str(x).strip() == 'Normal' else 1)
    elif file_type == 'Kitsune':
        data['label'] = data['Label']
    elif file_type in ['CICIoT', 'CICIoT2023']:
        data['label'] = data['attack_flag']
    else:
        logger.warning(f"Using generic anomal_judgment_label for {file_type}")
        data['label'] = anomal_judgment_label(data)
    
    return data