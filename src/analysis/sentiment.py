import sys
sys.path.append("../")
from utils.utils import create_data_frame


def sentiment_bart(classifier, transcript):
    """
    Analyzes the sentiment in a transcript using a BART-based classifier.

    Args:
        classifier: The zero-shot classifier model.
        transcript (str): The text transcript of the conversation between agent and customer.

    Returns:
        None: The function does not return a value directly but may store the results in a DataFrame.
    """
    labels = []
    confidence = []
    # Possible Sentiment Categories
    # Define the candidate sentiment labels that can be predicted
    candidate_labels = [
        "positive",  # Indicates a positive sentiment in the transcript.
        "negative",  # Indicates a negative sentiment in the transcript.
        "neutral",
    ]  # Indicates a neutral sentiment in the transcript.

    # Set the hypothesis template
    hypothesis_template = "The sentiment of this text is {}."

    # Send the labels and transcripts to the classifier pipeline
    result = classifier(
        transcript, candidate_labels, hypothesis_template=hypothesis_template
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
    create_data_frame(labels, confidence, transcript)