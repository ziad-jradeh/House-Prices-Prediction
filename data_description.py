### A function to get the description of a column name or a value

import json

from pathlib import Path
data_folder = Path("data/")
file_to_open = data_folder / "data_description.json"

with open(file_to_open, 'r') as file:
    description_dict = json.load(file)
    
def _get_des(name):
    if name in description_dict:
        return description_dict[name]['description']
    else:
        for col in description_dict:
            if name in description_dict[col]['values']:
                return description_dict[col]['values'][name]
    return name


def get_description(*args):
    '''A function to get the description of a column name or a value.
    
    Can take a single string or a list of strings.'''
    
    descriptions = [_get_des(arg) for arg in args]
    if len(descriptions) == 1:
        return descriptions[0]
    else:
        return descriptions