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

    cleaned_tokens = []
    for word in tokens:
        if word.isalpha() and word not in stopwords_english:
            stemmed_word = stemmer.stem(word)
            cleaned_tokens.append(stemmed_word)

    return cleaned_tokens

def build_freqs(texts, labels):
    freqs = {}
    for label, text in zip(labels, texts):
        words = process_text(text)
        for word in words:
            pair = (word, float(label))
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
    return freqs
