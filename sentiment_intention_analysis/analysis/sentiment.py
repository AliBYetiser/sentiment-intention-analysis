import sys

sys.path.append("../")
from utils.utils import create_result


def sentiment_bart(classifier, transcript, candidate_labels, hypothesis_template=None):
    """
    Analyzes the sentiment in a transcript using a BART-based classifier.

    Args:
        classifier (pipeline): The zero-shot classifier model.
        transcript (str): The text transcript of the conversation between agent and customer.
        candidate_labels (list): List of candidate sentiment labels to predict.
        hypothesis_template (str, optional): A template for hypothesis generation. Default is None.

    Returns:
        string: A string containing the transcript and sentiment classification results.


    Example:
        sentiment_bart(classifier, "Hello! How can I assist you today?", ["positive", "negative", "neutral"])
    """
    labels = []
    confidence = []

    # Send the labels and transcripts to the classifier pipeline
    result = classifier(
        transcript,
        candidate_labels=candidate_labels,
        hypothesis_template=hypothesis_template,
    )

    # Extract the labels from results dictionary
    labels.append(result["labels"])
    labels = [
        item for sublist in labels for item in sublist
    ]  # Flatten the list of lists into list

    # Extract the confidence scores and format as percentages
    confidence.append(result["scores"])
    confidence = [
        (str(float(item) * 100))[:6] + " %"
        for sublist in confidence
        for item in sublist
    ]  # Flatten the list of lists into list

    # Create a DataFrame with the labels, confidence scores, and transcript
    return create_result(labels, confidence, transcript)
