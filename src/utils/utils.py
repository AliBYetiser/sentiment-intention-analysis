import os
import json
import pandas as pd
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Get the current, parent, and data directories
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
transcript_file = "transcript.json"  # Name of the JSON file
transcript_path = os.path.join(parent_dir, "data/transcript.json")
model_path = os.path.join(parent_dir, "resources/bart-large-mnli")


# Initialize the analysis model and tokenizer
def initialize_classifier():
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)
    return classifier


def load_transcript_data():
    """
    Load transcript data from a JSON file located in the specified directory.

    Args:
        None: The function loads the transcript.json in data folder.

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


def create_data_frame(labels, confidence, transcript):
    """
    Create a DataFrame to display sentiment/intention classification results.

    Args:
        labels (list): List of sentiment labels (e.g., 'positive', 'negative', 'neutral').
        confidence (list): List of confidence scores for each sentiment label.
        transcript (str): The text transcript being analyzed.

    Returns:
        None: The function prints the DataFrame to the console.
    """
    # Create DataFrames for labels and confidence scores
    labels_df = pd.DataFrame({"Labels": labels})
    confidence_df = pd.DataFrame({"Confidence Scores": confidence})

    # Concatenate the DataFrames horizontally without resetting the index
    scores = pd.concat([labels_df, confidence_df], ignore_index=False, axis=1)

    # Print the results
    print(
        "\n--------------------------------------------------------------------------------------"
    )
    print(f"\n Entered input sentence: {transcript}")
    print("\n Sentiment of the transcript (Probability Distribution): ")
    print(scores.to_string(index=False))
    print(
        "\n--------------------------------------------------------------------------------------"
    )