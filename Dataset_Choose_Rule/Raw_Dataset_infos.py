# Dataset_Choose_Rule/Raw_Dataset_infos.py

Dataset_infos = {
    "CICIDS2017": {
        "has_header": True,
        "description": "CICIDS2017 dataset specific information"
        # Add more info as needed (e.g., default paths, column details)
    },
    "MiraiBotnet": {
        "has_header": True,
        "description": "Mirai Botnet dataset"
    },
    "NSL-KDD": {
        "has_header": False, # Example: NSL-KDD might not have a header (verify this)
        "description": "NSL-KDD dataset"
    },
    "DARPA98": {
        "has_header": True,
        "description": "DARPA98 dataset"
    },
    "CICModbus23": {
        "has_header": True,
        "description": "CICModbus23 dataset"
    },
    "IoTID20": {
        "has_header": True,
        "description": "IoTID20 dataset"
    }
    # Add other file_type entries as needed
}

def get_dataset_info(file_type):
    """
    Retrieves information for a given file_type.
    Returns a default dictionary if the file_type is not found.
    """
    return Dataset_infos.get(file_type, {"has_header": True, "description": "Unknown dataset type"})

# Example usage:
if __name__ == '__main__':
    print("CICIDS2017 info:", get_dataset_info("CICIDS2017"))
    print("Unknown dataset info:", get_dataset_info("UnknownDataset")) 