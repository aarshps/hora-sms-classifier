import re
import pickle
import datetime

class IndianBankSMSParser:
    def __init__(self):
        # Define the regular expressions to parse the Indian banking SMS
        self.regex_bank_name = re.compile(r'(?i)(?:bank|sbi|icici|hdfc|axis|pnb|kotak|idbi|yes|rbl|paytm|amazon|flipkart)')
        self.regex_transaction_type = re.compile(r'(?i)(?:credit|debit|transfer)')
        self.regex_amount = re.compile(r'\d+(?:\.\d+)?')
        self.regex_date = re.compile(r'\d{2}-\d{2}-\d{4}')
        self.regex_account_number = re.compile(r'\d{10}')
        self.regex_card_number = re.compile(r'\d{16}')

    def parse(self, sms):
        # Extract the bank name (if present)
        bank_name = self.regex_bank_name.findall(sms)
        if bank_name:
            bank_name = bank_name[0]
        else:
            bank_name = None

        # Extract the transaction type (if present)
        transaction_type = self.regex_transaction_type.findall(sms)
        if transaction_type:
            transaction_type = transaction_type[0]
        else:
            transaction_type = None

        # Extract the amount (if present)
        amount = self.regex_amount.findall(sms)
        if amount:
            amount = amount[0]
        else:
            amount = None

        # Extract the date (if present)
        date = self.regex_date.findall(sms)
        if date:
            date = datetime.datetime.strptime(date[0], '%d-%m-%Y')
        else:
            date = None

        # Extract the account number (if present)
        account_number = self.regex_account_number.findall(sms)
        if account_number:
            account_number = account_number[0]
        else:
            account_number = None

        # Extract the card number (if present)
        card_number = self.regex_card_number.findall(sms)
        if card_number:
            card_number = card_number[0]
        else:
            card_number = None

        # Return the parsed data
        return {
            "bank_name": bank_name,
            "transaction_type": transaction_type,
            "amount": amount,
            "date": date,
            "account_number": account_number,
            "card_number": card_number
        }

def train_model(train_data):
    # Create an IndianBankSMSParser object
    parser = IndianBankSMSParser()

    # Parse the training data
    parsed_training_data = []
    for message in train_data:
        parsed_training_data.append(parser.parse(message))

    # Save the parsed training data to a file
    with open("parsed_training_data.pickle", "wb") as f:
        pickle.dump(parsed_training_data, f)

def parse_sms(sms):
    # Load the parsed training data from the file
    with open("parsed_training_data.pickle", "rb") as f:
        parsed_training_data = pickle.load(f)

    # Create an IndianBankSMSParser object
    parser = IndianBankSMSParser()

    # Parse the SMS
    parsed_sms = parser.parse(sms)

    return parsed_sms

# Example usage:

# Train the model on the training data
train_data = ["Credited ₹100 to your HDFC account 1234567890 on 2023-10-29.", "Debited ₹50 from your SBI account 9876543210 on 2023-10-29.", "Transferred ₹200 from your ICICI account 9876543210 to your Axis Bank account 1234567890123456 on 2023-10-29."]

train_model(train_data)

# Parse a new SMS
sms = "Debited ₹20 from your HDFC account 1234567890 for your monthly subscription to Netflix on 2023-10-29."

parsed_sms = parse_sms(sms)
print(parsed_sms)