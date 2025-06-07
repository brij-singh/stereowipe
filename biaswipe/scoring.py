from biaswipe.metrics import compute_stereotype_rate, compute_stereotype_severity_score

def score_model_responses(prompts: dict, model_responses: dict, stereotype_detector: callable) -> dict:
    """
    Scores model responses for stereotyping.

    Args:
        prompts: A dictionary of prompts (prompt_id: prompt_text).
        model_responses: A dictionary of model responses (prompt_id: response_text).
        stereotype_detector: A function that takes response text and returns
                             a tuple (binary_label, severity_score).

    Returns:
        A dictionary containing the Stereotype Rate (SR) and Stereotype Severity Score (SSS).
    """
    all_stereotype_labels = []
    all_severity_scores = []

    if not model_responses:
        # Handle cases with no model responses to avoid division by zero or errors later
        return {"SR": 0.0, "SSS": 0.0}

    for prompt_id, response_text in model_responses.items():
        # Ensure response_text is a string, as stereotype_detector expects text
        if not isinstance(response_text, str):
            # Handle cases where a response might not be text (e.g. if data is malformed)
            # Option 1: Skip this response
            # print(f"Warning: Response for prompt_id {prompt_id} is not text, skipping.")
            # continue
            # Option 2: Treat as non-stereotyping
            binary_label, severity_score = False, 0.0
        else:
            binary_label, severity_score = stereotype_detector(response_text)

        all_stereotype_labels.append(binary_label)
        all_severity_scores.append(severity_score)

    sr_value = compute_stereotype_rate(all_stereotype_labels)
    sss_value = compute_stereotype_severity_score(all_severity_scores, all_stereotype_labels)

    return {"SR": sr_value, "SSS": sss_value}
