# Neural Sentiment and Intention Analysis of Agent-Customer Interaction Transcripts with BART
import os
from sentiment_intention_analysis.analysis.sentiment import sentiment_bart
from sentiment_intention_analysis.analysis.intention import intention_bart
from sentiment_intention_analysis.utils.utils import initialize_classifier, load_transcript_data
import configparser
import ast
import flask
from flask import Flask
from flask import request

app = Flask(__name__)


@app.route("/sample_transcript")
def sample_transcript():
    # Initialize an empty string to store concatenated customer responses
    concatenated_responses = ""
    # Iterate through the transcript data and concatenate customer responses
    for entry in transcript_data:
        if "role" in entry and "customer" == entry["role"]:
            concatenated_responses += (
                entry["message"] + " "
            )  # Concatenate with a space between responses

    sentiment = sentiment_bart(
        classifier,
        concatenated_responses,
        sentiment_candidate_labels,
        sentiment_hypothesis_template,
    )
    intention = intention_bart(
        classifier, concatenated_responses, intention_labels_with_descriptions
    )
    response = {"sentiment": sentiment, "intention": intention}
    return flask.jsonify(response)

@app.route("/predict")
def predict():
    user_input = request.args.get("input")
    sentiment = sentiment_bart(
        classifier,
        user_input,
        sentiment_candidate_labels,
        sentiment_hypothesis_template,
    )
    intention = intention_bart(
        classifier, user_input, intention_labels_with_descriptions
    ),
    response = {"sentiment": sentiment, "intention": intention}
    return flask.jsonify(response)


if __name__ == "__main__":
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
    # Load transcript data from path
    transcript_data = load_transcript_data(transcript_path)
    classifier = initialize_classifier(model_path)
    app.run(host="0.0.0.0", port="8080")
