import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np

# Load the text message dataset
data = pd.read_csv('text_messages.csv')

# Remove the NaN values from the data
data = data.dropna()

# Transform the data into TF-IDF vectors
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(data['text'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, data['label'], test_size=0.25)

# Create a logistic regression classifier
classifier = LogisticRegression()

# Train the classifier on the training data
classifier.fit(X_train, y_train)

# Save the trained classifier
with open('model.pkl', 'wb') as f:
    pickle.dump(classifier, f)

# Load the trained classifier
with open('model.pkl', 'rb') as f:
    classifier = pickle.load(f)

# Evaluate the classifier on the testing data
y_pred = classifier.predict(X_test)
accuracy = np.mean(y_pred == y_test)

print('Accuracy:', accuracy)
