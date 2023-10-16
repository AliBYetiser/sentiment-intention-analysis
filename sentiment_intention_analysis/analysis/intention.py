import sys

sys.path.append("../")
from utils.utils import create_dict


def intention_bart(classifier, transcript, candidate_labels_with_descriptions):
    """
    Analyzes the intention in a transcript using a BART-based classifier.

    Args:
        classifier (pipeline): The zero-shot classifier model.
        transcript (str): The text transcript of the conversation between agent and customer.
        candidate_labels_with_descriptions (list): List of candidate intention labels and their descriptions.

    Returns:
        dict: A dictionary containing the transcript and intention classification results.

    """
    labels = []
    confidence = []

    # Send the labels and transcripts to the classifier pipeline
    result = classifier(transcript, candidate_labels=candidate_labels_with_descriptions)

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
    return create_dict(labels, confidence, transcript)
