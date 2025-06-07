import json

def load_prompts(file_path: str) -> dict:
    """Loads prompts from a JSON file.

    Prompts are expected to be a dictionary where each value is another
    dictionary containing "text" and "category" keys.

    Args:
        file_path: Path to the JSON file containing prompts.

    Returns:
        A dictionary mapping prompt ID to prompt data (text and category),
        or an empty dictionary if a major error occurs (file not found, JSON decode error).
        Malformed individual prompts will be skipped with a warning.
    """
    processed_prompts = {}
    try:
        with open(file_path, 'r') as f:
            prompts_data = json.load(f)

        if not isinstance(prompts_data, dict):
            print(f"Error: Prompts file at {file_path} does not contain a valid JSON object (dictionary).")
            return {}

        for prompt_id, prompt_content in prompts_data.items():
            if not isinstance(prompt_content, dict):
                print(f"Warning: Prompt ID '{prompt_id}' in {file_path} is not a valid dictionary. Skipping.")
                continue

            text = prompt_content.get("text")
            category = prompt_content.get("category")

            if text is None:
                print(f"Warning: Prompt ID '{prompt_id}' in {file_path} is missing 'text' key. Skipping.")
                continue
            if category is None:
                print(f"Warning: Prompt ID '{prompt_id}' in {file_path} is missing 'category' key. Skipping.")
                continue

            if not isinstance(text, str):
                print(f"Warning: Prompt ID '{prompt_id}' in {file_path} has a 'text' value that is not a string. Skipping.")
                continue
            if not isinstance(category, str):
                 print(f"Warning: Prompt ID '{prompt_id}' in {file_path} has a 'category' value that is not a string. Skipping.")
                 continue

            processed_prompts[prompt_id] = {"text": text, "category": category}
        return processed_prompts

    except FileNotFoundError:
        print(f"Error: Prompts file not found at {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from prompts file at {file_path}")
        return {}

def load_json_data(file_path: str) -> dict:
    """Loads data from an arbitrary JSON file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        A dictionary containing the loaded JSON data, or an empty dictionary if an error occurs.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print(f"Error: JSON file at {file_path} does not contain a valid JSON object (dictionary).")
            return {}
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from file at {file_path}")
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
