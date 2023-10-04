from analysis.sentiment import sentiment_bart
from analysis.intention import intention_bart
from utils.utils import initialize_classifier, load_transcript_data


classifier = initialize_classifier()

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
        transcript_data = load_transcript_data()
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

            sentiment_bart(classifier, concatenated_responses)
            intention_bart(classifier, concatenated_responses)

        elif inference_selection == 2:
            # Perform sentiment and intention analysis on each customer message
            for entry in transcript_data:
                if "role" in entry and "customer" == entry["role"]:
                    message = entry["message"]
                    sentiment_bart(classifier, message)
                    intention_bart(classifier, message)

    elif choice == 2:
        print("\nPlease enter a transcript:")
        user_input = input()
        sentiment_bart(user_input)
        intention_bart(user_input)

    elif choice == 3:
        print("\nExiting...")
        break
