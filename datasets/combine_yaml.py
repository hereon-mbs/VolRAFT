# Combine YAML files into one single dataset
import argparse
import yaml
import os
import glob
from collections import OrderedDict

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

def save_yaml(data, path) -> None:
    """
    Save a dictionary `data` to a YAML file specified by `path`.
    
    Parameters:
    data (dict): The data to be saved as YAML.
    path (str): The path to the file to save the YAML content to.
    """
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

    return

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

def sort_yaml_list(input_file, output_file, order, list_name, verbose = True):
    """
    Sorts a specified list within a YAML file based on a defined key order and 
    writes the sorted content to a new YAML file.

    This function targets a specific list within the YAML file, where each element
    of the list is expected to be a dictionary. The keys of these dictionaries are
    then sorted according to the specified order.

    Parameters:
    input_file (str): Path to the input YAML file.
    output_file (str): Path to the output YAML file where the sorted content will be saved.
    order (list): A list of strings representing the desired order of keys.
    list_name (str): The name of the list in the YAML file to be sorted.
    verbose (bool, optional): If True, the function will print intermediate messages. Default is False.

    Returns:
    None: The function writes the sorted content to a new YAML file and does not return anything.
    """
    # Custom representer for OrderedDict to avoid !!python/object/apply in YAML output
    def represent_ordereddict(dumper, data):
        return dumper.represent_dict(data.items())

    # Load YAML file
    with open(input_file, 'r') as file:
        data = yaml.safe_load(file)

    # Check if the specified list exists in the data
    if list_name in data and isinstance(data[list_name], list):
        if verbose:
            print(f'found {len(data[list_name])} items from {input_file}')
            print(f'sort the list according to the order of keys: {order}')

        sorted_list = []
        
        # Iterate over each element in the list
        for item in data[list_name]:
            if isinstance(item, dict):
                # Create an OrderedDict with keys in the desired order
                sorted_item = OrderedDict()
                for key in order:
                    if key in item:
                        sorted_item[key] = item[key]
                sorted_list.append(sorted_item)

        # Replace the original list with the sorted list
        data[list_name] = sorted_list

    # Write the updated data to a new YAML file
    with open(output_file, 'w') as file:
        yaml.add_representer(OrderedDict, represent_ordereddict)
        yaml.dump(data, file, default_flow_style=False)

    if verbose:
        print(f'save {len(data[list_name])} items to {output_file}')

    return

def sort_list_in_dict(input_dict, list_name, order, verbose=False):
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
                for key in order:
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

def combine_yaml(args) -> None:
    """
    Combine YAML files in `src` to a YAML file specified by `dst`.
    
    Parameters:
    src (str): The source path to the YAML files.
    dst (str): The destination path to the YAML file.
    """
    paths = sorted(glob.glob(args.src))

    count = 0
    content_datasets = {}
    content_datasets["datasets"] = []
    for _, path in enumerate(paths):
        content = load_yaml(path)

        for _, dataset in enumerate(content["datasets"]):
            if args.debug:
                print(f'{count}: {dataset["name"]}')
            content_datasets["datasets"] += [dataset]
            count += 1

    # Sort the content based on the 'name' key
    if args.sort:
        content_datasets["datasets"] = sorted(content_datasets["datasets"], key = lambda x: x["name"])

    # Sort the content based on the order defined
    key_order = ['enable', 'name', 'vol0_path', 'vol1_path', 'mask_path', 'flow_path']

    content_datasets_sorted = sort_list_in_dict(input_dict = content_datasets, 
                                                list_name = "datasets", 
                                                order = key_order, 
                                                verbose = args.debug)

    # save_yaml(content_datasets, args.dst)
    save_yaml_from_ordered(content_datasets_sorted, args.dst)

    if args.debug:
        print(f'write {count} items to {args.dst}')

    return

if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser(description='Combine YAML')
    parser.add_argument('--src', type=str, default='./dataset_list/datasets_*[0-9].yaml', help='File path to the source YAML files')
    parser.add_argument('--dst', type=str, default='./datasets_all.yaml', help='File path to the destination YAML file')
    parser.add_argument('--sort', action='store_true', help="Enable sorting the datasets according to its name")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    args = parser.parse_args()

    # Task
    combine_yaml(args)
