# Import all the required libraries
import pandas as pd
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def create_data_frame(labels, confidence, transcript):
    """
    Create a DataFrame to display sentiment classification results.

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
    sentiment_scores = pd.concat([labels_df, confidence_df], ignore_index=False, axis=1)

    # Print the results
    print("\n--------------------------------------------------------------------------------------")
    print(f"\n Entered input sentence: {transcript}")
    print("\n Sentiment of the transcript (Probability Distribution): ")
    print(sentiment_scores.to_string(index=False))
    print("\n--------------------------------------------------------------------------------------")


def sentiment_bart(transcript):
    """
    Analyzes the sentiment in a transcript using a BART-based classifier.

    Args:
        transcript (str): The text transcript of the conversation between agent and customer.

    Returns:
        None: The function does not return a value directly but may store the results in a DataFrame.
    """

    labels = []
    confidence = []

    # Possible Sentiment Categories
    # Define the candidate sentiment labels that can be predicted
    candidate_labels = ["positive",   # Indicates a positive sentiment in the transcript.
                        "negative",   # Indicates a negative sentiment in the transcript.
                        "neutral"]    # Indicates a neutral sentiment in the transcript.

    # Send the labels and transcripts to the classifier pipeline
    result = classifier(transcript, candidate_labels)

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


def intention_bart(transcript):
    """
    Analyzes the intention in a transcript using a BART-based classifier.
    
    Args:
        transcript (str): The text transcript of the conversation between agent and customer.
    
    Returns:
        None: The function does not return a value directly but may store the results in a DataFrame.
    """
    labels = []
    confidence = []

    # Possible Intention Categories
    candidate_labels = [
        "change_package",      # Customer intends to change their subscription package.
        "upgrade",             # Customer intends to upgrade their product or service.
        "purchase_product",    # Customer intends to buy a product.
        "ask_question",        # Customer intends to ask a question or seek information.
        "resolve_issue",       # Customer intends to resolve an existing problem or issue.
        "provide_feedback",    # Customer intends to provide feedback or a review.
        "request_assistance",  # Customer intends to request assistance or help.
        "complain",            # Customer intends to express dissatisfaction or lodge a complaint.
        "express_gratitude",   # Customer intends to show appreciation or gratitude.
        "cancel_order",        # Customer intends to cancel an order.
        "change_preferences",  # Customer intends to modify their preferences or settings.
        "subscribe_service"    # Customer intends to subscribe to a service.
    ]
    # Send the labels and transcripts to the classifier pipeline
    result = classifier(transcript, candidate_labels)

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


# Driver program
while True:
    print("Neural Sentiment and Intention Analysis of Agent-Customer Interaction Transcripts with BART")
    print("\n------Available Options------")
    print("1. Inference on Sample Transcripts")
    print("2. Enter Custom Transcript")
    print("3. Exit")
    print("\nPlease select an option from the above:")
    choice = int(input())

    if choice == 1:
        sample_1 = "Hi there! I'm interested in buying the new iPhone 14. Can you provide me with some information about it?"
        sentiment_bart(sample_1)
        intention_bart(sample_1)

        sample_2 = "Can you help me set it up? I can't figure it out, it is not user friendly at all!"
        sentiment_bart(sample_2)
        intention_bart(sample_2)

    elif choice == 2:
        print("\nPlease enter a transcript:")
        user_input = input()
        sentiment_bart(user_input)
        intention_bart(user_input)

    elif choice == 3:
        print("\nExiting...")
        break
