import yaml

class YAMLHandler:
    """
    A class for handling YAML files with static methods, eliminating the need for instance properties.
    """

    @staticmethod
    def read_yaml(filepath):
        """
        Safely reads and returns the content of a YAML file.

        Parameters:
            filepath (str): The path to the YAML file to be read.

        Returns:
            dict: The content of the YAML file as a dictionary.
        """
        try:
            with open(filepath, 'r') as file:
                return yaml.safe_load(file) or {}
        except FileNotFoundError:
            print(f"Warning: File {filepath} not found. Returning empty dictionary.")
            return {}
        except yaml.YAMLError as e:
            print(f"Error reading YAML file {filepath}: {e}")
            return {}

    @staticmethod
    def write_yaml(filepath, data, default_flow_style=False, sort_keys=False):
        """
        Writes the given data to a YAML file, handling YAML and IO errors.

        Parameters:
            filepath (str): The path to the YAML file to be written.
            data (dict): The data to be written to the YAML file.
        """
        try:
            with open(filepath, 'w') as file:
                yaml.safe_dump(data, file, default_flow_style=default_flow_style, sort_keys=sort_keys)
        except yaml.YAMLError as e:
            print(f"Error writing YAML data to {filepath}: {e}")
        except IOError as e:
            print(f"IOError when attempting to write to {filepath}: {e}")

    @staticmethod
    def update_yaml(filepath, updates):
        """
        Deeply updates a YAML file with the given data, supporting nested dictionaries.

        Parameters:
            filepath (str): The path to the YAML file to be updated.
            updates (dict): The dictionary with updates.
        """
        data = YAMLHandler.read_yaml(filepath)
        YAMLHandler.deep_update(data, updates)
        YAMLHandler.write_yaml(filepath, data)

    @staticmethod
    def add_entry(filepath, key, value):
        """
        Adds a new entry or updates an existing entry in a YAML file.

        Parameters:
            filepath (str): The path to the YAML file to be updated.
            key (str): The key for the entry.
            value: The value for the entry.
        """
        YAMLHandler.update_yaml(filepath, {key: value})

    @staticmethod
    def deep_update(source, updates):
        """
        Recursively updates a dictionary with another dictionary, supporting nested keys.

        Parameters:
            source (dict): The original dictionary to be updated.
            updates (dict): The dictionary with updates.

        Returns:
            dict: The updated dictionary.
        """
        for key, value in updates.items():
            if isinstance(value, dict) and value:
                returned = YAMLHandler.deep_update(source.get(key, {}), value)
                source[key] = returned
            else:
                source[key] = updates[key]
        return source

# Example usage:
# filepath = 'config.yaml'
# config_data = YAMLHandler.read_yaml(filepath)
# print(config_data)
# YAMLHandler.add_entry(filepath, 'new_nested_key', {'nested': 'value'})
# YAMLHandler.update_yaml(filepath, {'another_key': 'another_value'})
