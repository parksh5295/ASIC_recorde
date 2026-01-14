import json
import os
import logging

logger = logging.getLogger(__name__)

def save_to_json(data, file_path, indent=4):
    """
    Saves the given data to a JSON file.

    Args:
        data: The data to save (must be JSON serializable).
        file_path (str): The path to the file where the data will be saved.
        indent (int, optional): Indentation level for pretty-printing. Defaults to 4.
    """
    try:
        # Ensure the directory exists
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        logger.info(f"Successfully saved data to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        return False

def load_from_json(file_path):
    """
    Loads data from a JSON file.

    Args:
        file_path (str): The path to the JSON file to load.

    Returns:
        The loaded data, or None if an error occurs.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded data from {file_path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return None

if __name__ == '__main__':
    # Example usage (optional, for testing the module directly)
    test_data_dir = "test_data"
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)
    
    example_file = os.path.join(test_data_dir, "example.json")
    
    # Test saving
    my_data = {"name": "Test Data", "version": 1.0, "items": [1, 2, 3]}
    if save_to_json(my_data, example_file):
        print(f"Data saved to {example_file}")

    # Test loading
    loaded_data = load_from_json(example_file)
    if loaded_data:
        print(f"Data loaded from {example_file}: {loaded_data}")
        # Verify data
        assert loaded_data["name"] == "Test Data"
        assert loaded_data["version"] == 1.0
        
    # Test loading non-existent file
    non_existent_file = os.path.join(test_data_dir, "nothing.json")
    load_from_json(non_existent_file)

    # Test loading invalid json
    invalid_json_file = os.path.join(test_data_dir, "invalid.json")
    with open(invalid_json_file, 'w') as f:
        f.write("{'name': 'test',}") # Invalid JSON (single quotes, trailing comma)
    load_from_json(invalid_json_file)

    print("Test complete. Check log for details.") 