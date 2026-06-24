import nltk
from nltk.tokenize import word_tokenize

# Download the necessary tokenizer model (only needs to be run once)
nltk.download('punkt_tab')

def tokenize_messages(text_data):
    """
    Splits a message into individual words/tokens.
    """
    # word_tokenize handles punctuation and contractions intelligently
    return word_tokenize(text_data)

# Example Usage
messages = [
    "The shipment arrived damaged today.",
    "Could you please provide an update on order #12345?"
]

for i, msg in enumerate(messages, 1):
    tokens = tokenize_messages(msg)
    print(f"Message {i}: {tokens}")
