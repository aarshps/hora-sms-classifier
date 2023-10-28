import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

class HDFCFederalBankClassifier:
    def __init__(self, train_data, train_labels):
        self.vectorizer = TfidfVectorizer()
        self.classifier = LogisticRegression()
        self.vectorizer.fit(train_data)
        self.classifier.fit(self.vectorizer.transform(train_data), train_labels)

    def predict(self, text_message):
        return self.classifier.predict_proba(self.vectorizer.transform([text_message]))[0]

def train_model(train_data, train_labels):
    return HDFCFederalBankClassifier(train_data, train_labels)


def predict(text_message, classifier):
    return classifier.predict(text_message)

train_data = [
    "Credited $100 to your HDFC account."
    , "Debited $50 from your Federal Bank account."
]
train_labels = [1, 0]
classifier = train_model(train_data, train_labels)
text_message = "Credited $100 to your HDFC account."
prediction = predict(text_message, classifier)
print(prediction)
