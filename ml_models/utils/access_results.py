import os
import json


def write_to_json(data, filename):
    """
    This function is write the data to a json file as a temporary storage.

    Args:
        data (dictionary): containing the model prediction results
        filename (string): filename of the json file.
    """
    
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Assuming this is in a module

    # Move up two levels from the current module's directory
    base_dir_up_two_levels = os.path.abspath(os.path.join(base_dir, os.pardir, os.pardir))

    relative_path = os.path.join(base_dir_up_two_levels, 'media', 'model_results', filename)
    # Construct the absolute path
    absolute_path = os.path.join(base_dir, relative_path)

    # Ensure that the directory structure exists, creating directories if necessary
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)

    # Write to the JSON file
    with open(absolute_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def read_prediction_from_json(filename):
    """_summary_

    Args:
        filename (string): the json filename where the model resutls is saved.

    Returns:
        dictionary: the data contained in the json file.
    """
    
    levels_to_move_up = 2
    
    current_file_path = os.path.dirname(os.path.abspath(__file__))  # Assuming this is in a module

    # Move up the specified number of levels
    for _ in range(levels_to_move_up):
        current_file_path = os.path.dirname(current_file_path)

    relative_path = os.path.join( 'media','model_results', filename)
    # Construct the absolute path
    absolute_path = os.path.join(current_file_path, relative_path)
    print("absolute_path>>>>>",absolute_path)
    
    with open(absolute_path, 'r') as file:
        data = json.load(file)

    return data

