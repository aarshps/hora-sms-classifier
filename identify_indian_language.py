import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class IndianLanguageIdentifier:
    def __init__(self):
        # Define the regular expressions to extract language-specific features
        self.regex_hindi_features = re.compile(r'(?i)(?:hindi|हिंदी|हिन्दी)')
        self.regex_bengali_features = re.compile(r'(?i)(?:bengali|বাংলা|বাঙ্গ्ला)')
        self.regex_telugu_features = re.compile(r'(?i)(?:telugu|తెలుగు)')
        self.regex_marathi_features = re.compile(r'(?i)(?:marathi|मराठी)')
        self.regex_tamil_features = re.compile(r'(?i)(?:tamil|தமிழ்)')
        self.regex_kannada_features = re.compile(r'(?i)(?:kannada|ಕನ್ನಡ)')

        # Create a TfidfVectorizer object to transform the text into TF-IDF vectors
        self.vectorizer = TfidfVectorizer()

        # Create a LogisticRegression object to train and evaluate the model
        self.classifier = LogisticRegression()

    def train(self, X_train, y_train):
        # Transform the training data into TF-IDF vectors
        X_train_vectors = self.vectorizer.fit_transform(X_train)

        # Train the classifier
        self.classifier.fit(X_train_vectors, y_train)

    def predict(self, X_test):
        # Transform the test data into TF-IDF vectors
        X_test_vectors = self.vectorizer.transform(X_test)

        # Make predictions on the test data
        y_pred = self.classifier.predict_proba(X_test_vectors)[:, 1]

        return y_pred

def load_data(train_file):
    # Load the training data
    X_train = []
    y_train = []
    with open(train_file, 'r') as f:
        for line in f:
            text, language = line.strip().split(',')
            X_train.append(text)
            y_train.append(language)

    return X_train, y_train

# Load the training data
train_file = 'indian_language_identification_training_data.csv'

X_train, y_train = load_data(train_file)

# Create an IndianLanguageIdentifier object
identifier = IndianLanguageIdentifier()

# Train the model on the training data
identifier.train(X_train, y_train)

# Evaluate the model's performance on the training data
y_pred = identifier.predict(X_train)
accuracy = np.mean(y_pred == y_train)

print('Accuracy:', accuracy)
