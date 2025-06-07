import json

def load_prompts(file_path: str) -> dict:
    """Loads prompts from a JSON file.

    Args:
        file_path: Path to the JSON file containing prompts.

    Returns:
        A dictionary mapping prompt ID to prompt text, or an empty dictionary if an error occurs.
    """
    try:
        with open(file_path, 'r') as f:
            prompts = json.load(f)
        return prompts
    except FileNotFoundError:
        print(f"Error: Prompts file not found at {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from prompts file at {file_path}")
        return {}

def load_annotations(file_path: str) -> dict:
    """Loads human annotations from a JSON file.

    Args:
        file_path: Path to the JSON file containing annotations.

    Returns:
        A dictionary mapping prompt ID to annotation data (binary label and severity score),
        or an empty dictionary if an error occurs.
    """
    try:
        with open(file_path, 'r') as f:
            annotations = json.load(f)
        return annotations
    except FileNotFoundError:
        print(f"Error: Annotations file not found at {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from annotations file at {file_path}")
        return {}

def load_model_responses(file_path: str) -> dict:
    """Loads model responses from a JSON file.

    Args:
        file_path: Path to the JSON file containing model responses.

    Returns:
        A dictionary mapping prompt ID to response text, or an empty dictionary if an error occurs.
    """
    try:
        with open(file_path, 'r') as f:
            model_responses = json.load(f)
        return model_responses
    except FileNotFoundError:
        print(f"Error: Model responses file not found at {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from model responses file at {file_path}")
        return {}
