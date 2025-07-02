import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def process_text(text):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')

    text = re.sub(r'https?://[^\s\n\r]+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text.lower())

    # Remove stopwords and stem
    cleaned_tokens = [
        stemmer.stem(word)
        for word in tokens
        if word not in stopwords_english and word.isalpha()
    ]

    return cleaned_tokens

def build_freqs(texts, labels):
    freqs = {}
    for label, text in zip(labels, texts):
        for word in process_text(text):
            pair = (word, label)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
    return freqs