from itertools import product
import os
import json
from copy import deepcopy


def dict_sort(dict_list:list) -> list:
    """Sorts a list of dictionaries by their string versions

    Args:
        dict_list (list): list of dictionaries

    Returns:
        list: sorted list of dictionaries
    """
    return sorted(dict_list, key=lambda d: json.dumps(d))
    

def unfold_dict(input_dict:dict) -> list:
    """
    Recursively unfolds a dictionary into multiple dictionaries if values are lists.
    For nested dictionaries, the unfolding continues recursively.

    Args:
        input_dict (dict): The input dictionary to unfold.

    Returns:
        list: A list of dictionaries representing all combinations of the input dictionary's values.
    """
    # Base case: if the dictionary is empty, return a single empty dictionary
    if not input_dict:
        return [{}]

    # Resultant list to store the unfolded dictionaries
    result = [{}]

    for key, value in input_dict.items():
        # If the value is a list, create combinations by unfolding
        if isinstance(value, list):
            temp_result = []
            for item in value:
                for partial_dict in result:
                    temp_dict = partial_dict.copy()
                    temp_dict[key] = item
                    temp_result.append(temp_dict)
            result = temp_result

        # If the value is a dictionary, recursively unfold
        elif isinstance(value, dict):
            temp_result = []
            nested_unfolded = unfold_dict(value)
            for nested_dict in nested_unfolded:
                for partial_dict in result:
                    temp_dict = partial_dict.copy()
                    temp_dict[key] = nested_dict
                    temp_result.append(temp_dict)
            result = temp_result

        # For non-list, non-dict values, just copy them to the result
        else:
            for partial_dict in result:
                partial_dict[key] = value

    return dict_sort(result)