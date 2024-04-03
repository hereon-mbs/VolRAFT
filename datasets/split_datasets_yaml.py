import argparse
import yaml
import os
import re
from collections import OrderedDict


class InlineList(list):
    pass

def inline_list_presenter(dumper, data):
    return dumper.represent_sequence(u'tag:yaml.org,2002:seq', data, flow_style=True)

def load_yaml(path):
    """
    Load the content from a YAML file specified by `path`.
    
    Parameters:
    path (str): The path to the file to load the YAML content from.
    
    Returns:
    dict: The data loaded from the YAML file, or None if the file doesn't exist.
    """
    if os.path.exists(path):
        with open(path, 'r') as file:
            data = yaml.safe_load(file)
        return data
    else:
        print("File does not exist:", path)
        return None
    
def sort_top_level_keys(input_dict, key_order, verbose=False):
    """
    Sorts the top-level keys of a dictionary based on a defined key order.

    Parameters:
    input_dict (dict): The dictionary whose top-level keys are to be sorted.
    key_order (list): A list of strings representing the desired order of top-level keys.
    verbose (bool, optional): If True, the function will print intermediate messages. Default is False.

    Returns:
    dict: The input dictionary with its top-level keys sorted as per the key_order.
    """

    # Create an OrderedDict to store the sorted items
    sorted_dict = OrderedDict()

    # Iterate over the key_order and add the corresponding items from the input_dict
    for key in key_order:
        if key in input_dict:
            sorted_dict[key] = input_dict[key]
            if verbose:
                print(f"Added '{key}' to sorted dictionary.")
        elif verbose:
            print(f"'{key}' not found in the input dictionary.")

    # Add any remaining items from the input_dict that were not in key_order
    for key in input_dict:
        if key not in sorted_dict:
            sorted_dict[key] = input_dict[key]
            if verbose:
                print(f"Added '{key}' to sorted dictionary (not in specified order).")

    return sorted_dict

def sort_list_in_dict(input_dict, list_name, key_order, verbose=False):
    """
    Sorts dictionaries within a specified list in a given dictionary based on a defined key order.

    This function targets a specific list within the input dictionary, where each element
    of the list is expected to be a dictionary. The keys of these dictionaries are then
    sorted according to the specified order.

    Parameters:
    input_dict (dict): The dictionary containing the list to be sorted.
    list_name (str): The name of the list in the dictionary to be sorted.
    order (list): A list of strings representing the desired order of keys.
    verbose (bool, optional): If True, the function will print intermediate messages. Default is False.

    Returns:
    dict: The input dictionary with the specified list sorted as per the order.
    """

    if list_name in input_dict and isinstance(input_dict[list_name], list):
        if verbose:
            print(f'found {len(input_dict[list_name])} items from input dictionary')

        sorted_list = []

        # Iterate over each element in the list
        for item in input_dict[list_name]:
            if isinstance(item, dict):
                # Create an OrderedDict with keys in the desired order
                sorted_item = OrderedDict()
                for key in key_order:
                    if key in item:
                        sorted_item[key] = item[key]

                sorted_list.append(sorted_item)

        # Replace the original list with the sorted list
        input_dict[list_name] = sorted_list

        if verbose:
            print(f'Sorted {len(input_dict[list_name])} items')

    elif verbose:
        print(f"'{list_name}' not found or is not a list in the provided dictionary.")

    return input_dict

def save_yaml_from_ordered(data, path) -> None:
    """
    Save an ordered dictionary `data` to a YAML file specified by `path`.
    
    Parameters:
    data (dict): The data to be saved as YAML.
    path (str): The path to the file to save the YAML content to.
    """
    # Custom representer for OrderedDict to avoid !!python/object/apply in YAML output
    def represent_ordereddict(dumper, data):
        return dumper.represent_dict(data.items())
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Write the updated data to a new YAML file
    with open(path, 'w') as file:
        yaml.add_representer(OrderedDict, represent_ordereddict)
        yaml.dump(data, file, default_flow_style=False)

    return

def contains_pattern(name):
    # Define the regex pattern
    # '.*' matches any characters (including none) followed by '_fs402'
    pattern_re = ".*_syn0151_.*|.*_syn0154_.*|.*_syn0155_.*"

    # Use re.search to check if the pattern is in the string
    return re.search(pattern_re, name)

def split_datasets_yaml(args):
    yaml.add_representer(InlineList, inline_list_presenter)

    content_src = load_yaml(args.src)

    # Sort the content based on the order defined
    key_order = ['enable', 'name', 'vol0_path', 'vol1_path', 'mask_path', 'flow_path']

    content_dst = dict()
    content_dst["datasets_train"] = []
    content_dst["datasets_test"] = []

    for idx, dataset_src in enumerate(content_src["datasets"]):
        dataset_dst = dict()
        dataset_dst["enable"] = dataset_src["enable"]
        dataset_dst["name"] = dataset_src["name"]
        dataset_dst["vol0_path"] = dataset_src["vol0_path"]
        dataset_dst["vol1_path"] = dataset_src["vol1_path"]
        dataset_dst["mask_path"] = dataset_src["mask_path"]
        dataset_dst["flow_path"] = dataset_src["flow_path"]

        if contains_pattern(dataset_src["name"]):
            content_dst["datasets_test"].append(dataset_dst)
        else:
            content_dst["datasets_train"].append(dataset_dst)


    # Sort the content
    content_dst_sorted = content_dst
    content_dst_sorted = sort_top_level_keys(input_dict = content_dst_sorted, 
                                            key_order = ["datasets_train", "datasets_test"], 
                                            verbose = True)
    content_dst_sorted = sort_list_in_dict(input_dict = content_dst_sorted, 
                                        list_name = "datasets_train", 
                                        key_order = key_order, 
                                        verbose = True)
    content_dst_sorted = sort_list_in_dict(input_dict = content_dst_sorted, 
                                        list_name = "datasets_test", 
                                        key_order = key_order, 
                                        verbose = True)

    # Save to YAML file
    save_yaml_from_ordered(content_dst_sorted, path = args.dst)

if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser(description='Split Datasets YAML')
    parser.add_argument('--src', type=str, default='./datasets_all.yaml', help='File path to the source YAML files')
    parser.add_argument('--dst', type=str, default='./datasets.yaml', help='File path to the destination YAML file')
    parser.add_argument('--sort', action='store_true', help="Enable sorting the datasets according to its name")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    args = parser.parse_args()

    # Task
    split_datasets_yaml(args)
