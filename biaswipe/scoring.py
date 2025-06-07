from biaswipe.metrics import compute_stereotype_rate, compute_stereotype_severity_score, compute_csss, compute_wosi

def score_model_responses(prompts: dict, model_responses: dict, stereotype_detector: callable, category_weights: dict) -> dict:
    """
    Scores model responses for stereotyping, including SR, SSS, CSSS, and WOSI.

    Args:
        prompts: A dictionary of prompts (prompt_id: {text: str, category: str}).
        model_responses: A dictionary of model responses (prompt_id: response_text).
        stereotype_detector: A function that takes response text and returns
                             a tuple (binary_label, severity_score).
        category_weights: A dictionary mapping category to its weight for WOSI.

    Returns:
        A dictionary containing SR, SSS, CSSS (dictionary), and WOSI scores.
    """
    all_stereotype_labels = []
    all_severity_scores = []
    all_prompt_ids = []

    if not model_responses:
        # Handle cases with no model responses
        sr_value = compute_stereotype_rate([])
        sss_value = compute_stereotype_severity_score([], [])
        csss_results = compute_csss(prompts, [], [], []) # prompts might be non-empty
        wosi_result = compute_wosi(csss_results, category_weights)
        return {"SR": sr_value, "SSS": sss_value, "CSSS": csss_results, "WOSI": wosi_result}

    for prompt_id, response_text in model_responses.items():
        # Ensure response_text is a string, as stereotype_detector expects text
        if not isinstance(response_text, str):
            # Treat non-string responses as non-stereotyping
            binary_label, severity_score = False, 0.0
            print(f"Warning: Response for prompt_id {prompt_id} is not text, treating as non-stereotyping.")
        else:
            binary_label, severity_score = stereotype_detector(response_text)

        all_stereotype_labels.append(binary_label)
        all_severity_scores.append(severity_score)
        all_prompt_ids.append(prompt_id)

    sr_value = compute_stereotype_rate(all_stereotype_labels)
    sss_value = compute_stereotype_severity_score(all_severity_scores, all_stereotype_labels)
    csss_results = compute_csss(prompts, all_stereotype_labels, all_severity_scores, all_prompt_ids)
    wosi_result = compute_wosi(csss_results, category_weights)

    return {"SR": sr_value, "SSS": sss_value, "CSSS": csss_results, "WOSI": wosi_result}
