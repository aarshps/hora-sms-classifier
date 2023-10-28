import re
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class TransactionExtractor:
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
        prediction = self.classifier.predict_proba(text_message_vector)[0]

        # Extract the bank name, amount, type of transaction, account number, and card number from the text message
        bank_name = self.extract_bank_name(text_message)
        amount = self.extract_amount(text_message)
        transaction_type = self.extract_transaction_type(prediction)
        account_number = self.extract_account_number(text_message)
        card_number = self.extract_card_number(text_message)

        # Return the extracted features
        return bank_name, amount, transaction_type, account_number, card_number

def _extract_bank_name(text_message):
    return re.findall(r'(?i)(?:bank|sbi|icici|hdfc|axis|pnb|kotak|idbi|yes|rbl|paytm|amazon|flipkart)', text_message)[0]

def _extract_amount(text_message):
    amount = re.findall(r'\d+', text_message)
    return max(amount) if len(amount) > 0 else None

def _extract_transaction_type(prediction):
    return 'credit' if prediction[1] > prediction[0] else 'debit'

def _extract_account_number(text_message):
    account_number = re.findall(r'\d{10,}', text_message)
    return account_number[0] if len(account_number) > 0 else None

def _extract_card_number(text_message):
    card_number = re.findall(r'\d{4}-\d{4}-\d{4}-\d{4}', text_message)
    return card_number[0] if len(card_number) > 0 else None

def _train_model(train_data, train_labels):
    vectorizer = TfidfVectorizer()
    classifier = LogisticRegression()

    vectorizer.fit(train_data)
    classifier.fit(vectorizer.transform(train_data), train_labels)

    with open("transaction_extractor.pickle", "wb") as f:
        pickle.dump(TransactionExtractor(vectorizer, classifier), f)

def _predict(text_message):
    with open("transaction_extractor.pickle", "rb") as f:
        transaction_extractor = pickle.load(f)

    bank_name, amount, transaction_type, account_number, card_number = transaction_extractor.predict(text_message)

    print("Bank name:", bank_name)
    print("Amount:", amount)
    print("Transaction type:", transaction_type)
    print("Account number:", account_number)
    print("Card number:", card_number)

if __name__ == "__main__":
    train_data = pd.read_csv("train_data.csv")
    X_train = train_data["text_message"]
    y_train = train_data["label"]

    _train_model(X_train, y_train)

    text_message = "Credited $100 to your SBI account."
    _predict(text_message)
