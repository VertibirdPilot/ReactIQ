# preprocessing.py
import re
from typing import List
import nltk

# If you have not downloaded these, run once:
# nltk.download('stopwords')
# nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove urls
    text = re.sub(r"@\w+", "", text)     # remove mentions
    text = re.sub(r"[^a-z\s]", " ", text) # remove non-letters
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_and_lemmatize(text: str, remove_stopwords: bool = True) -> str:
    text = basic_clean(text)
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)

if _name_ == "_main_":
    print(tokenize_and_lemmatize("I loved the product! It's amazing :) http://a.com @user"))