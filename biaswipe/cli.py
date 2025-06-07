import click
import os
import json # Required for model name extraction if it involves reading the JSON.
            # However, for just filename manipulation, it's not strictly needed here
            # but good to have if more complex model name logic was added.

from biaswipe import data_loader, metrics, scoring, report

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
    '--report-output', 'report_output_path',
    type=click.Path(dir_okay=False, writable=True),
    default='./report.json',
    show_default=True,
    help="Path to save the generated JSON report."
)
def run_benchmark(prompts_path: str, annotations_path: str, model_responses_dir_path: str, report_output_path: str):
    """
    Runs the BiasWipe benchmark to score models for stereotyping.
    """
    click.echo("Starting BiasWipe benchmark process...")

    prompts = data_loader.load_prompts(prompts_path)
    if not prompts:
        click.echo(f"Failed to load prompts from {prompts_path}. Exiting.")
        return

    # Annotations are loaded as per spec, even if not directly used in this scoring version
    annotations = data_loader.load_annotations(annotations_path)
    if not annotations:
        click.echo(f"Warning: Failed to load annotations from {annotations_path}. Proceeding without them for scoring, but this might be an issue for future features.")
        # Depending on strictness, you might choose to exit here as well.
        # For now, we'll allow proceeding as scoring doesn't use it yet.

    all_model_results = {}

    click.echo(f"Loading model responses from: {model_responses_dir_path}")
    try:
        for filename in os.listdir(model_responses_dir_path):
            if filename.endswith(".json"):
                model_response_file_path = os.path.join(model_responses_dir_path, filename)
                click.echo(f"Processing model response file: {model_response_file_path}")

                model_responses = data_loader.load_model_responses(model_response_file_path)

                if model_responses:
                    # Extract model name from filename (e.g., "model_A" from "model_A.json")
                    model_name = os.path.splitext(filename)[0]

                    click.echo(f"Scoring responses for model: {model_name}...")
                    scores = scoring.score_model_responses(
                        prompts=prompts,
                        model_responses=model_responses,
                        stereotype_detector=metrics.mock_stereotype_detector
                    )
                    all_model_results[model_name] = scores
                    click.echo(f"Finished scoring for model: {model_name}. SR: {scores.get('SR', 'N/A')}, SSS: {scores.get('SSS', 'N/A')}")
                else:
                    click.echo(f"Warning: Could not load responses from {model_response_file_path} or file is empty. Skipping.")
            else:
                click.echo(f"Skipping non-JSON file: {filename}")
    except OSError as e:
        click.echo(f"Error reading from model responses directory {model_responses_dir_path}: {e}. Exiting.")
        return


    if not all_model_results:
        click.echo("No model responses were successfully processed. Skipping report generation.")
    else:
        click.echo("Generating benchmark report...")
        report.generate_report(all_model_results, report_output_path)

    click.echo(f"BiasWipe benchmark process completed. Report available at: {report_output_path if all_model_results else 'N/A'}")

if __name__ == '__main__':
    run_benchmark()
