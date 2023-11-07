# Neural Sentiment and Intention Analysis of Agent-Customer Interaction Transcripts with BART
import os
import sys
sys.path.append(os.getcwd())
from analysis.sentiment import sentiment_bart
from analysis.intention import intention_bart
from utils.utils import initialize_classifier, load_transcript_data
import configparser
import ast
import flask
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    user_input = None
    sample_transcript = None
    _, customer_responses = sample()
    sample_prediction = None

    if request.method == 'POST':
        user_input = predict(request.form['user_input'])

    if request.method == 'GET':
        if 'sample_button' in request.args:
            sample_transcript,_ = sample()
    
    if request.method == 'GET':
        if 'predict_sample_button' in request.args:
            sample_prediction = predict(customer_responses)

    return render_template('index.html', user_input=user_input, sample=sample_transcript, sample_prediction=sample_prediction)


def sample():
    transcript = ""
    # Initialize an empty string to store concatenated customer responses
    customer_responses = ""
    # Iterate through the transcript data and concatenate customer responses
    for entry in transcript_data:
        transcript += entry["message"] + " \n"

        if "role" in entry and "customer" == entry["role"]:
            customer_responses += (
                entry["message"] + " "
            )  # Concatenate with a space between responses
    
    transcript = transcript.split('\n')
    return transcript, customer_responses


def predict(input="!"):
    sentiment = sentiment_bart(
        classifier,
        input,
        sentiment_candidate_labels,
        sentiment_hypothesis_template,
    )
    intention = intention_bart(
        classifier, input, intention_labels_with_descriptions
    )

    response = "--------------------- Sentiment ---------------------\n" + sentiment +"\n" + "--------------------- Intention ---------------------\n" + intention + "\n"
    return response.split('\n')


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
    app.run()
