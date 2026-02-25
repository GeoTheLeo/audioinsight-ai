"""
Sentiment evaluation module using pretrained DistilBERT.
"""

import os
import pandas as pd
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix
from src.config import SENTIMENT_MODEL

PROCESSED_DIR = "data/processed"


def load_sentiment_model():
    """
    Load pretrained sentiment model with truncation enabled.
    """
    return pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        truncation=True,       # ✅ Critical fix
        max_length=512         # ✅ Ensures safe input size
    )


def evaluate_sentiment_model(df: pd.DataFrame) -> None:
    """
    Evaluate model against star-based sentiment mapping.
    """

    print("Loading sentiment model...")

    classifier = load_sentiment_model()

    sample_df = df.sample(min(2000, len(df)), random_state=42)

    predictions = classifier(
        sample_df["text"].tolist(),
        batch_size=16
    )

    sample_df["predicted_label"] = [
        "positive" if p["label"] == "POSITIVE" else "negative"
        for p in predictions
    ]

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    sample_df.to_csv(
        f"{PROCESSED_DIR}/sentiment_evaluation_sample.csv",
        index=False
    )

    print("\nClassification Report:\n")
    print(
        classification_report(
            sample_df["sentiment_label"],
            sample_df["predicted_label"]
        )
    )

    print("\nConfusion Matrix:\n")
    print(
        confusion_matrix(
            sample_df["sentiment_label"],
            sample_df["predicted_label"]
        )
    )