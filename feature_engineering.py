
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple
import os

VECT_PATH = "vectorizer.pkl"

def fit_vectorizer(corpus, max_features: int = 5000) -> TfidfVectorizer:
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    vec.fit(corpus)
    return vec
