# feature_engineering.py
import os
import pickle
from typing import Iterable, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_VECTORIZER_PATH = "vectorizer.pkl"


def fit_vectorizer(
    texts: Iterable[str],
    *,
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> TfidfVectorizer:
    """Fit a TF-IDF vectorizer on a given corpus of texts."""
    texts = list(texts)
    if len(texts) == 0:
        raise ValueError("Cannot fit vectorizer: empty corpus provided.")
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    vectorizer.fit(texts)
    return vectorizer


def save_vectorizer(vectorizer: TfidfVectorizer, path: str = DEFAULT_VECTORIZER_PATH) -> None:
    """Persist a fitted vectorizer to disk."""
    with open(path, "wb") as file:
        pickle.dump(vectorizer, file)


def load_vectorizer(path: str = DEFAULT_VECTORIZER_PATH) -> TfidfVectorizer:
    """Load a vectorizer from disk."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No vectorizer found at {path}. Please fit and save one first.")
    with open(path, "rb") as file:
        return pickle.load(file)
