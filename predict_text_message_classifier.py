import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained classifier
classifier = pickle.load(open('model.pkl', 'rb'))

# Predict the label of a new text message
text_message = 'Dear HDFC Bank customer, your account has been credited with Rs.10,000.'

vectorizer = TfidfVectorizer()

text_message_tfidf = vectorizer.transform([text_message])
label = classifier.predict(text_message_tfidf)

print('Predicted label:', label)
