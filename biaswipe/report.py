import json

def generate_report(results: dict, output_file_path: str) -> None:
    """
    Generates a JSON report from the scoring results.

    Args:
        results: A dictionary containing the scoring results.
                 Example: {"model_name_1": {"SR": 0.5, "SSS": 0.6},
                           "model_name_2": {"SR": 0.2, "SSS": 0.3}}
        output_file_path: The path to save the JSON report file.
    """
    try:
        with open(output_file_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Report successfully generated at {output_file_path}")
    except IOError:
        print(f"Error: Could not write report to {output_file_path}")
    except Exception as e:
        print(f"An unexpected error occurred while generating the report: {e}")
