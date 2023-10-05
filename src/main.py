import os
from analysis.sentiment import sentiment_bart
from analysis.intention import intention_bart
from utils.utils import initialize_classifier, load_transcript_data
import configparser
import ast

# Get relevant directories
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)

# Read configuration values from config.ini
config_path = os.path.join(parent_dir, "config/config.ini")
config = configparser.ConfigParser()
config.read(config_path)

# Model configuration
model_path = os.path.join(parent_dir, config["Model"]["model_location"])
# Detailed model configuration is set in resources/config.json

# Sentiment configuration
sentiment_candidate_labels = ast.literal_eval(
    config["Sentiment"]["sentiment_candidate_labels"]
)
sentiment_hypothesis_template = config["Sentiment"]["sentiment_hypothesis_template"]

# Intention configuration
intention_labels_with_descriptions = ast.literal_eval(
    config["Intention"]["intention_labels_with_descriptions"]
)

transcript_path = os.path.join(parent_dir, config["Data"]["transcript_location"])

classifier = initialize_classifier(model_path)

# Driver program
while True:
    print(
        "Neural Sentiment and Intention Analysis of Agent-Customer Interaction Transcripts with BART"
    )
    print("\n------Available Options------")
    print("1. Inference on sample transcript")
    print("2. Enter custom transcript")
    print("3. Exit")
    print("\nPlease select an option from the above:")
    choice = int(input())
    if choice == 1:
        # Load transcript data from the JSON file
        transcript_data = load_transcript_data(transcript_path)
        print("\n------Available Options------")
        print("1. Infer on the whole transcript")
        print(
            "     In customer experience analysis or summarization tasks, "
            "analyzing the entire conversation as a whole might be preferred."
        )
        print("2. Infer on each customer message")
        print(
            "     In customer support or chatbot application context this could make more sense."
        )
        print("\nPlease select an option from the above:")
        inference_selection = int(input())

        if inference_selection == 1:
            # Initialize an empty string to store concatenated customer responses
            concatenated_responses = ""
            # Iterate through the transcript data and concatenate customer responses
            for entry in transcript_data:
                if "role" in entry and "customer" == entry["role"]:
                    concatenated_responses += (
                        entry["message"] + " "
                    )  # Concatenate with a space between responses

            sentiment_bart(
                classifier,
                concatenated_responses,
                sentiment_candidate_labels,
                sentiment_hypothesis_template,
            )
            intention_bart(
                classifier, concatenated_responses, intention_labels_with_descriptions
            )

        elif inference_selection == 2:
            # Perform sentiment and intention analysis on each customer message
            for entry in transcript_data:
                if "role" in entry and "customer" == entry["role"]:
                    message = entry["message"]
                    sentiment_bart(
                        classifier,
                        message,
                        sentiment_candidate_labels,
                        sentiment_hypothesis_template,
                    )
                    intention_bart(
                        classifier, message, intention_labels_with_descriptions
                    )

    elif choice == 2:
        print("\nPlease enter a transcript:")
        user_input = input()
        sentiment_bart(
            classifier,
            user_input,
            sentiment_candidate_labels,
            sentiment_hypothesis_template,
        )
        intention_bart(classifier, user_input, intention_labels_with_descriptions)

    elif choice == 3:
        print("\nExiting...")
        break
