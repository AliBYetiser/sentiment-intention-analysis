import sys
sys.path.append("../")
from utils.utils import create_data_frame


def intention_bart(classifier, transcript):
    """
    Analyzes the intention in a transcript using a BART-based classifier.

    Args:
        classifier: The zero-shot classifier model.
        transcript (str): The text transcript of the conversation between agent and customer.

    Returns:
        None: The function does not return a value directly but may store the results in a DataFrame.
    """
    labels = []
    confidence = []
    # Possible Intention Categories with Descriptions
    candidate_labels = [
        {
            "label": "change_package",
            "description": "Customer intends to change their subscription package.",
        },
        {
            "label": "upgrade",
            "description": "Customer intends to upgrade their product or service.",
        },
        {
            "label": "purchase_product",
            "description": "Customer intends to buy a product.",
        },
        {
            "label": "ask_question",
            "description": "Customer intends to ask a question or seek information.",
        },
        {
            "label": "resolve_issue",
            "description": "Customer intends to resolve an existing problem or issue.",
        },
        {
            "label": "provide_feedback",
            "description": "Customer intends to provide feedback or a review.",
        },
        {
            "label": "request_assistance",
            "description": "Customer intends to request assistance or help.",
        },
        {
            "label": "complain",
            "description": "Customer intends to express dissatisfaction or lodge a complaint.",
        },
        {
            "label": "express_gratitude",
            "description": "Customer intends to show appreciation or gratitude.",
        },
        {
            "label": "cancel_order",
            "description": "Customer intends to cancel an order.",
        },
        {
            "label": "change_preferences",
            "description": "Customer intends to modify their preferences or settings.",
        },
        {
            "label": "subscribe_service",
            "description": "Customer intends to subscribe to a service.",
        },
    ]
    # Send the labels and transcripts to the classifier pipeline
    result = classifier(transcript, candidate_labels=candidate_labels)

    # Extract the labels from results dictionary
    labels.append(result["labels"])
    labels = [
        item for sublist in labels for item in sublist
    ]  # Flatten the list of lists into list

    # Extract the labels from results dictionary
    confidence.append(result["scores"])
    confidence = [
        (str(float(item) * 100))[:6] + " %"
        for sublist in confidence
        for item in sublist
    ]  # Flatten the list of lists into list

    # Create a DataFrame with the labels, confidence scores, and transcript
    create_data_frame(labels, confidence, transcript)