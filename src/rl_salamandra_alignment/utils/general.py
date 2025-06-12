from itertools import product
import os
import json
from copy import deepcopy
from rl_salamandra_alignment import logger
import yaml


def dict_sort(dict_list: list) -> list:
    """Sorts a list of dictionaries by their string versions

    Args:
        dict_list (list): list of dictionaries

    Returns:
        list: sorted list of dictionaries
    """
    return sorted(dict_list, key=lambda d: json.dumps(d))

def drop_none_values(data):
    """
    Recursively drops all None values from dictionaries and lists of dictionaries.
    
    Args:
        data: Input data (dict, list, or other types)
        
    Returns:
        Cleaned data with all None values removed
    """
    if isinstance(data, dict):
        # Process dictionary
        return {
            key: drop_none_values(value)
            for key, value in data.items()
            if value is not None
        }
    elif isinstance(data, list):
        # Process list (filter None and recurse into elements)
        return [
            drop_none_values(item)
            for item in data
            if item is not None
        ]
    else:
        # Return non-dict/list values as-is
        return data

def unfold_dict(input_dict: dict) -> list:
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
    # Do not unfold "evaluation"
    evaluation_config = input_dict.pop("evaluation", None)

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
    result = [deepcopy(d) for d in result]
    
    # drop all None values:
    result = [drop_none_values(d) for d in result]

    # Make sure all configs share the same evaluation config.
    if evaluation_config:
        for d in result:
            d["evaluation"] = deepcopy(evaluation_config)
    
    return dict_sort(result)


def try_load_config(config_file: str) -> dict:
    """
    Load a YAML configuration file as a dictionary.

    Parameters:
    config_file (str): Path to the configuration file.

    Returns:
    dict: Configuration dictionary.
    """
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

    except FileNotFoundError:
        logger.warning(f"Configuration file {config_file} not found.")
        config = {}
    except yaml.YAMLError as exc:
        logger.warning(f"Error in configuration file: {exc}")
        config = {}
    logger.debug("Using the following configuration:")
    logger.debug(
        json.dumps(config, indent=2)
    )
    return config

