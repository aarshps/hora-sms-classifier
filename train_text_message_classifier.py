import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the text message dataset
data = pd.read_csv('text_messages.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.25)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform the training and testing data into TF-IDF vectors
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Create a logistic regression classifier
classifier = LogisticRegression()

# Train the classifier on the training data
classifier.fit(X_train_tfidf, y_train)

# Evaluate the classifier on the testing data
y_pred = classifier.predict(X_test_tfidf)
accuracy = np.mean(y_pred == y_test)

print('Accuracy:', accuracy)

# Save the trained classifier
classifier.save_model('text_message_classifier.pkl')
