import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class HDFCvsFederalBankClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
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

def load_data(train_file, test_file):
    # Load the training data
    X_train = []
    y_train = []
    with open(train_file, 'r') as f:
        for line in f:
            text_message, label = line.strip().split(',')
            X_train.append(text_message)
            y_train.append(int(label))

    # Load the test data
    X_test = []
    y_test = []
    with open(test_file, 'r') as f:
        for line in f:
            text_message, label = line.strip().split(',')
            X_test.append(text_message)
            y_test.append(int(label))

    return X_train, y_train, X_test, y_test

# Load the training and test data
train_file = 'hdfc_vs_federal_bank_training_data.csv'
test_file = 'hdfc_vs_federal_bank_test_data.csv'

X_train, y_train, X_test, y_test = load_data(train_file, test_file)

# Create a HDFCvsFederalBankClassifier object
classifier = HDFCvsFederalBankClassifier()

# Train the classifier on the training data
classifier.train(X_train, y_train)

# Evaluate the model's performance on the test data
y_pred = classifier.predict(X_test)
accuracy = np.mean(y_pred == y_test)

print('Accuracy:', accuracy)
