# dataset_loader.py
import pandas as pd
from typing import Tuple

POSSIBLE_TEXT_COLS = ["text", "content", "tweet", "message", "review"]
POSSIBLE_LABEL_COLS = ["label", "sentiment", "target", "score"]

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # find text col
    text_col = next((c for c in df.columns if c.lower() in POSSIBLE_TEXT_COLS), None)
    label_col = next((c for c in df.columns if c.lower() in POSSIBLE_LABEL_COLS), None)

    if text_col is None or label_col is None:
        # fallback: try heuristics
        for c in df.columns:
            if df[c].dtype == object and text_col is None:
                text_col = c
            if df[c].nunique() <= 10 and label_col is None and df[c].dtype in ['int64','float64','object']:
                label_col = c

