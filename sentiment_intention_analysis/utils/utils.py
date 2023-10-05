import json
import pandas as pd
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer


def initialize_classifier(model_path):
    """
    Initialize a zero-shot classification pipeline with a pre-trained model and tokenizer.

    Args:
        model_path (str): The directory path or model identifier specifying the pre-trained model to use.

    Returns:
        pipeline: A zero-shot classification pipeline configured with the specified model and tokenizer.

    Example:
        classifier = initialize_classifier("facebook/bart-large-mnli")
        result = classifier("Text to classify", candidate_labels=["label_1", "label_2"])
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)
    return classifier


def load_transcript_data(transcript_path):
    """
    Load transcript data from a JSON file located in the specified directory.

    Args:
        transcript_path (str): The directory path specifying the location of transcript.json.

    Returns:
        list: A list of transcript entries, each entry represented as a dictionary.
              Each dictionary should have a 'transcript' key containing the text transcript.

    Example usage:
        transcript_data = load_transcript_data("./data/transcript.json")
    """
    # Load JSON data from the file
    with open(transcript_path, "r") as json_file:
        data = json.load(json_file)

    return data


def create_dict(labels, confidence, transcript):
    """
    Create a DataFrame to display sentiment/intention classification results.

    Args:
        labels (list): List of sentiment labels (e.g., 'positive', 'negative', 'neutral').
        confidence (list): List of confidence scores for each sentiment label.
        transcript (str): The text transcript being analyzed.

    Returns:
            dict: A dictionary containing the transcript and probability distribution results.

    """
    # Create DataFrames for labels and confidence scores
    labels_df = pd.DataFrame({"Labels": labels})
    confidence_df = pd.DataFrame({"Confidence Scores": confidence})

    # Concatenate the DataFrames horizontally without resetting the index
    scores = pd.concat([labels_df, confidence_df], ignore_index=False, axis=1)
    result = {
        "Transcript": transcript,
        "Probability Distribution": scores.to_string(
            index=False
        ),
    }
    return result
