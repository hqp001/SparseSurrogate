import os
import json
import hashlib
from datetime import datetime
import torch  # PyTorch library for model handling

def generate_id_from_args(input_args):
    """Generate a unique ID based on the input arguments using SHA256 hashing."""
    args_str = json.dumps(input_args, sort_keys=True)  # Convert dictionary to a sorted JSON string
    full_hash = hashlib.sha256(args_str.encode()).hexdigest()  # Generate SHA256 hash
    unique_id = full_hash[:16]  # Truncate to 8 characters
    return unique_id

def insert_input_arguments(base_path='models', input_args=None):
    """Insert input arguments, generate an ID, and create a corresponding folder."""
    if input_args is None:
        raise ValueError("Input arguments must be provided.")

    # Generate a unique ID from the input arguments
    input_id = generate_id_from_args(input_args)
    folder_path = os.path.join(base_path, input_id)

    # Create the folder if it does not exist
    os.makedirs(folder_path, exist_ok=True)

    # Save input arguments in input.json
    input_file_path = os.path.join(folder_path, 'input.json')
    with open(input_file_path, 'w') as json_file:
        json.dump(input_args, json_file, indent=4)
    print(f"Input arguments saved at {input_file_path}")

    # Update summary metadata after creating a new input ID folder
    update_summary_metadata(base_path, input_id, input_args)

    return input_id

def insert_model(base_path='models', input_id=None, model_name=None, output_args=None, model=None):
    """Insert model information and save it under the corresponding input ID folder."""
    if input_id is None or model_name is None or output_args is None or model is None:
        raise ValueError("All parameters (input_id, model_name, output_args, model) must be provided.")

    # Create subfolder for the specific model type under the given input ID
    model_type_folder = os.path.join(base_path, input_id, model_name)
    os.makedirs(model_type_folder, exist_ok=True)

    # Save output arguments in output.json
    output_file_path = os.path.join(model_type_folder, 'output.json')
    with open(output_file_path, 'w') as json_file:
        json.dump(output_args, json_file, indent=4)
    print(f"Output arguments saved at {output_file_path}")

    # Save the PyTorch model file
    model_file_path = os.path.join(model_type_folder, f"{model_name}.pth")
    torch.save(model.state_dict(), model_file_path)
    print(f"Model '{model_name}' saved at {model_file_path}")

def update_summary_metadata(base_path, input_id, input_args):
    """Update the summary metadata with a new entry for the input ID."""
    summary_file_path = os.path.join(base_path, 'summary.json')

    # Load existing summary data or initialize it
    if os.path.exists(summary_file_path):
        with open(summary_file_path, 'r') as json_file:
            summary_data = json.load(json_file)
    else:
        summary_data = {'models': []}

    # Check if the input_id already exists in the summary
    existing_entry = next((item for item in summary_data['models'] if item['input_id'] == input_id), None)

    if existing_entry:
        # Update the existing entry
        existing_entry['input_arguments'] = input_args
    else:
        # Add a new entry
        summary_data['models'].append({'input_id': input_id, 'input_arguments': input_args})

    # Save the updated summary data
    with open(summary_file_path, 'w') as json_file:
        json.dump(summary_data, json_file, indent=4)
    print(f"Summary metadata updated at {summary_file_path}")

def get_all_input_arguments(base_path='models'):
    """
    Loop through input IDs in summary.json file and return a dictionary of input arguments.

    Parameters:
    - base_path: Base directory where models and summary.json are stored.

    Returns:
    - inputs_dict: A dictionary where each key is an input ID and the value is its corresponding input arguments.
    """
    summary_file_path = os.path.join(base_path, 'summary.json')

    # Check if the summary file exists
    if not os.path.exists(summary_file_path):
        raise FileNotFoundError(f"Summary file not found at: {summary_file_path}")

    # Load the summary data
    with open(summary_file_path, 'r') as json_file:
        summary_data = json.load(json_file)

    # Extract the input arguments dictionary
    inputs_dict = {}
    for entry in summary_data.get('models', []):
        input_id = entry.get('input_id')
        input_args = entry.get('input_arguments')

        if input_id and input_args:
            inputs_dict[input_id] = input_args

    return inputs_dict

def get_all_model_names(base_path='models', input_id=None):
    """
    Get all model names in a given input ID.

    Parameters:
    - base_path: Base directory where models are stored.
    - input_id: The input ID to search for models.

    Returns:
    - model_names: A list of model names found under the input ID.
    """
    if input_id is None:
        raise ValueError("Input ID must be provided.")

    # Construct the path to the input ID folder
    input_folder_path = os.path.join(base_path, input_id)

    # Check if the input folder exists
    if not os.path.exists(input_folder_path):
        raise FileNotFoundError(f"No folder found for input ID: {input_id}")

    # Get all model names by listing subdirectories
    model_names = [model_name for model_name in os.listdir(input_folder_path) if os.path.isdir(os.path.join(input_folder_path, model_name))]

    return model_names

def update_model_info(base_path='models', input_id=None, model_name=None, new_data=None):
    """
    Update info in the JSON file based on the pair (input_id, model_name).

    Parameters:
    - base_path: Base directory where models are stored.
    - input_id: The input ID to search for models.
    - model_name: The name of the model to update.
    - new_data: A dictionary containing the new information to update.
    """
    if input_id is None or model_name is None or new_data is None:
        raise ValueError("Input ID, model name, and new data must be provided.")

    # Construct the path to the model folder and the JSON file
    model_folder_path = os.path.join(base_path, input_id, model_name)
    output_file_path = os.path.join(model_folder_path, 'output.json')

    # Check if the JSON file exists
    if not os.path.exists(output_file_path):
        raise FileNotFoundError(f"The JSON file for model '{model_name}' does not exist at: {output_file_path}")

    # Read the existing data from the JSON file
    with open(output_file_path, 'r') as json_file:
        existing_data = json.load(json_file)

    # Update the existing data with the new data
    existing_data.update(new_data)

    # Write the updated data back to the JSON file
    with open(output_file_path, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)

    print(f"Updated JSON file for model '{model_name}' at: {output_file_path}")

def get_model_info(base_path='models', input_id=None, model_name=None):
    """
    Retrieve the .pth file path and arguments from the JSON file based on the pair (input_id, model_name).

    Parameters:
    - base_path: Base directory where models are stored.
    - input_id: The input ID to search for models.
    - model_name: The name of the model to retrieve.

    Returns:
    - model_file_path: The path to the .pth file.
    - output_args: The arguments stored in the JSON file.
    """
    if input_id is None or model_name is None:
        raise ValueError("Input ID and model name must be provided.")

    # Construct the path to the model folder and the JSON file
    model_folder_path = os.path.join(base_path, input_id, model_name)
    model_file_path = os.path.join(model_folder_path, f"{model_name}.pth")
    output_file_path = os.path.join(model_folder_path, 'output.json')

    # Check if the model file exists
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"The .pth file for model '{model_name}' does not exist at: {model_file_path}")

    # Check if the JSON file exists
    if not os.path.exists(output_file_path):
        raise FileNotFoundError(f"The JSON file for model '{model_name}' does not exist at: {output_file_path}")

    # Read the data from the JSON file
    with open(output_file_path, 'r') as json_file:
        output_args = json.load(json_file)

    return model_file_path, output_args

def filter_arguments(input_dict, args_to_keep):
    """
    Filter a dictionary to keep only the arguments in the provided list.

    Parameters:
    - input_dict: The dictionary to filter.
    - args_to_keep: List of keys to keep in the dictionary.

    Returns:
    - filtered_dict: A new dictionary containing only the specified keys.
    """
    filtered_dict = {key: value for key, value in input_dict.items() if key in args_to_keep}
    return filtered_dict
