
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple
import os

VECT_PATH = "vectorizer.pkl"

def fit_vectorizer(corpus, max_features: int = 5000) -> TfidfVectorizer:
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    vec.fit(corpus)
    return vec

def save_vectorizer(vec, path: str = VECT_PATH):
    with open(path, "wb") as f:
        pickle.dump(vec, f)

def load_vectorizer(path: str = VECT_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Train vectorizer first.")
    with open(path, "rb") as f:
        return pickle.load(f)