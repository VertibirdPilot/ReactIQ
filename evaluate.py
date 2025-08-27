# evaluate.py
from __future__ import annotations

import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from feature_engineering import load_vectorizer
from model_utils import load_model, print_metrics, plot_confusion
from preprocessing import tokenize_and_lemmatize
from dataset_loader import load_dataset


def evaluate_on(path: str = "dataset.csv",
                test_size_fraction: float = 0.2,
                random_state: int = 42) -> None:
    """Evaluate the saved model on a held-out test split."""
    df = load_dataset(path)

    # Basic schema check
    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input dataset must contain columns: {required}")

    # Deterministic split
    _, df_test = train_test_split(
        df,
        test_size=test_size_fraction,
        random_state=random_state,
        stratify=df["label"],
    )
    df_test = df_test.copy()

    # Load artifacts
    vec = load_vectorizer()
    model = load_model()

    # Preprocess and vectorize
    df_test.loc[:, "clean"] = df_test["text"].astype(str).apply(tokenize_and_lemmatize)
    X_test_vec = vec.transform(df_test["clean"])
    y_test = df_test["label"].astype(int)

    # Predict & report
    preds = model.predict(X_test_vec)
    print_metrics(y_test, preds)
    plot_confusion(y_test, preds, out_path="confusion_test.png")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "dataset.csv"
    evaluate_on(path)
