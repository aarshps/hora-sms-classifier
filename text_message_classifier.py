import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

class CreditDebitClassifier:
    def __init__(self, train_data, train_labels):
        # Initialize the vectorizer
        self.vectorizer = TfidfVectorizer()

        # Initialize the classifier
        self.classifier = LogisticRegression()

        # Fit the model to the training data
        self.vectorizer.fit(train_data)
        self.classifier.fit(self.vectorizer.transform(train_data), train_labels)

    def predict(self, text_message):
        # Transform the text message into a TF-IDF vector
        text_message_vector = self.vectorizer.transform([text_message])

        # Predict the class of the text message
        prediction = self.classifier.predict_proba(text_message_vector)[0][1]

        # Return the prediction
        return prediction

def train_model(train_data, train_labels):
    # Create a new classifier object
    classifier = CreditDebitClassifier(train_data, train_labels)

    # Save the trained model to a file
    with open("hora_classifier.pickle", "wb") as f:
        pickle.dump(classifier, f)

def predict_class(text_message):
    # Load the trained model from the file
    with open("hora_classifier.pickle", "rb") as f:
        classifier = pickle.load(f)

    # Predict the class of the text message
    prediction = classifier.predict(text_message)

    # Return the prediction
    return prediction

# Train the model on the training data
# Credited transactions
train_data_credit = [
    "Credited $100 to your account.",
    "Credited $200 to your account.",
    "Credited $300 to your account.",
    "Credited $400 to your account.",
    "Credited $500 to your account.",
    "Credited $600 to your account.",
    "Credited $700 to your account.",
    "Credited $800 to your account.",
    "Credited $900 to your account.",
    "Credited $1000 to your account.",
    "Credited $50 to your account.",
    "Credited $2500 to your account.",
    "Credited $10000 to your account.",
    "Credited your account with a bonus of $100.",
    "Credited your account with a refund of $50.",
    "Credited your account with a dividend of $25.",
    "Credited your account with interest of $10.",
    "Credited your account with a transfer of $500 from another account.",
    "Credited your account with a check deposit of $1000.",
    "Credited your account with a direct deposit of $2000."
]

# Debited transactions
train_data_debit = [
    "Debited $100 from your account.",
    "Debited $200 from your account.",
    "Debited $300 from your account.",
    "Debited $400 from your account.",
    "Debited $500 from your account.",
    "Debited $600 from your account.",
    "Debited $700 from your account.",
    "Debited $800 from your account.",
    "Debited $900 from your account.",
    "Debited $1000 from your account.",
    "Debited $50 from your account.",
    "Debited $2500 from your account.",
    "Debited $10000 from your account.",
    "Debited your account for a fee of $10.",
    "Debited your account for a purchase of $50.",
    "Debited your account for a withdrawal of $250.",
    "Debited your account for a transfer of $500 to another account.",
    "Debited your account for a bill payment of $1000.",
    "Debited your account for a recurring payment of $25."
]

train_data = train_data_credit + train_data_debit

train_labels = [1] * len(train_data_credit) + [0] * len(train_data_debit)

train_model(train_data, train_labels)

# Predict the class of a new text message
text_message = "Credited $10000 to your account."
prediction = predict_class(text_message)

# Print the prediction
print(prediction)
