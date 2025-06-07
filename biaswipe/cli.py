import click
import os
import json
from typing import List # Added for type hinting

from biaswipe import data_loader, scoring, report
from biaswipe.judge import Judge, MockJudge, OpenAIJudge, AnthropicJudge, MissingApiKeyError

@click.command()
@click.option(
    '--prompts', 'prompts_path',
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=True,
    help="Path to the JSON file containing prompts."
)
@click.option(
    '--annotations', 'annotations_path',
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=True,
    help="Path to the JSON file containing human annotations."
)
@click.option(
    '--model-responses-dir', 'model_responses_dir_path',
    type=click.Path(exists=True, file_okay=False, readable=True),
    required=True,
    help="Path to the directory containing model response JSON files."
)
@click.option(
    '--category-weights', 'category_weights_path',
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=False,
    help="Path to the JSON file containing category weights for WOSI calculation."
)
@click.option(
    '--report-output', 'report_output_path',
    type=click.Path(dir_okay=False, writable=True),
    default='./report.json',
    show_default=True,
    help="Path to save the generated JSON report."
)
@click.option(
    '--judges', 'judges_str',
    type=str,
    default='mock',
    show_default=True,
    help='Comma-separated list of judge names to use (e.g., "openai,anthropic", "mock", "openai"). Available: mock, openai, anthropic.'
)
def run_benchmark(
    prompts_path: str,
    annotations_path: str,
    model_responses_dir_path: str,
    category_weights_path: str | None,
    report_output_path: str,
    judges_str: str
):
    """
    Runs the BiasWipe benchmark to score models for stereotyping.
    """
    click.echo("Starting BiasWipe benchmark process...")

    # --- Instantiate Judges ---
    selected_judges: List[Judge] = []
    judge_names = [name.strip().lower() for name in judges_str.split(',')]

    available_judge_constructors = {
        "mock": MockJudge,
        "openai": OpenAIJudge,
        "anthropic": AnthropicJudge
    }

    click.echo(f"Selected judge models: {', '.join(judge_names)}")
    for name in judge_names:
        if name in available_judge_constructors:
            try:
                # You could add specific args here if needed, e.g. model variants
                if name == "mock":
                     selected_judges.append(MockJudge(name=f"CLI_{name}")) # Give CLI mocks distinct name
                else:
                    selected_judges.append(available_judge_constructors[name]())
                click.echo(f"Successfully instantiated judge: {name}")
            except MissingApiKeyError as e:
                click.echo(f"Warning: Could not instantiate judge '{name}' due to Missing API Key: {e}. This judge will be skipped.")
            except Exception as e:
                click.echo(f"Warning: Could not instantiate judge '{name}' due to an unexpected error: {e}. This judge will be skipped.")
        else:
            click.echo(f"Warning: Unknown judge name '{name}' provided. It will be skipped. Available: {', '.join(available_judge_constructors.keys())}")

    if not selected_judges:
        click.echo("Warning: No valid judges were instantiated (or none were selected). Defaulting to a single MockJudge.")
        selected_judges.append(MockJudge(name="CLI_DefaultMock"))

    click.echo(f"Using judges: {[type(j).__name__ + (f'({j.name})' if hasattr(j, 'name') else '') for j in selected_judges]}")

    # --- Load Data ---
    prompts = data_loader.load_prompts(prompts_path)
    if not prompts:
        click.echo(f"Failed to load prompts from {prompts_path}. Exiting.")
        return

    annotations = data_loader.load_annotations(annotations_path) # Not used in scoring currently, but loaded
    if not annotations:
        click.echo(f"Warning: Failed to load annotations from {annotations_path}. This may affect parts of the benchmark not yet implemented.")

    category_weights = {}
    if category_weights_path:
        click.echo(f"Loading category weights from: {category_weights_path}")
        category_weights = data_loader.load_json_data(category_weights_path)
        if not category_weights:
            click.echo(f"Warning: Failed to load category weights from '{category_weights_path}' or the file is empty.")
    else:
        click.echo("No category weights file provided. WOSI may be 0.0 or based on unweighted CSSS.")

    # --- Process Model Responses ---
    all_model_results = {}
    click.echo(f"Loading model responses from: {model_responses_dir_path}")
    try:
        for filename in os.listdir(model_responses_dir_path):
            if filename.endswith(".json"):
                model_response_file_path = os.path.join(model_responses_dir_path, filename)
                click.echo(f"Processing model response file: {model_response_file_path}")
                model_responses = data_loader.load_model_responses(model_response_file_path)

                if model_responses:
                    model_name = os.path.splitext(filename)[0]
                    click.echo(f"Scoring responses for model: {model_name}...")

                    scores = scoring.score_model_responses(
                        prompts=prompts,
                        model_responses=model_responses,
                        category_weights=category_weights,
                        judges=selected_judges # Pass the instantiated judges
                    )
                    all_model_results[model_name] = scores
                    click.echo(f"Finished scoring for model: {model_name}. SR: {scores.get('SR', 'N/A')}, SSS: {scores.get('SSS', 'N/A')}, WOSI: {scores.get('WOSI', 'N/A')}")
                else:
                    click.echo(f"Warning: Could not load responses from {model_response_file_path} or file is empty. Skipping.")
            else:
                click.echo(f"Skipping non-JSON file: {filename}")
    except OSError as e:
        click.echo(f"Error reading from model responses directory {model_responses_dir_path}: {e}. Exiting.")
        return

    # --- Generate Report ---
    if not all_model_results:
        click.echo("No model responses were successfully processed. Skipping report generation.")
    else:
        click.echo("Generating benchmark report...")
        report.generate_report(all_model_results, report_output_path)

    click.echo(f"BiasWipe benchmark process completed. Report available at: {report_output_path if all_model_results else 'N/A'}")

if __name__ == '__main__':
    run_benchmark()
```
