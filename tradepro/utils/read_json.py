import os
import json

def read_ticker_list():
    """
    Read a json file contain the ticker list and defintions.

    Returns:
        dictionary: the data contained in the json file.
    """
    filename = "ticker_list.json"
    levels_to_move_up = 2
    
    current_file_path = os.path.dirname(os.path.abspath(__file__))  # Assuming this is in a module

    # Move up the specified number of levels
    for _ in range(levels_to_move_up):
        current_file_path = os.path.dirname(current_file_path)

    relative_path = os.path.join( 'media', filename)
    absolute_path = os.path.join(current_file_path, relative_path)
    
    with open(absolute_path, 'r') as file:
        data = json.load(file)

    return data

