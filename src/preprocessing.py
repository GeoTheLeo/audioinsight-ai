"""
Data preprocessing module.

Loads Amazon Electronics reviews from Hugging Face
(McAuley Amazon Reviews 2023),
filters for consumer audio products,
maps binary sentiment labels,
and saves cleaned dataset to disk.
"""

import os
import pandas as pd
from datasets import load_dataset
from src.config import DATASET_SAMPLE_SIZE, AUDIO_KEYWORDS

PROCESSED_DIR = "data/processed"


def load_and_filter_data() -> pd.DataFrame:
    """
    Load Amazon Electronics dataset and filter audio-related reviews.

    Returns:
        pd.DataFrame: Filtered review-level dataset.
    """

    print("Loading Amazon Electronics dataset...")

    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_review_Electronics",
        split=f"full[:{DATASET_SAMPLE_SIZE}]",
        trust_remote_code=True
    )

    df = pd.DataFrame(dataset)

    # ✅ PRESERVE PRODUCT TITLE
    df = df[["asin", "title", "rating", "text"]].dropna()

    print(f"Initial dataset size: {len(df)}")

    # Filter for audio-related keywords
    keyword_pattern = "|".join(AUDIO_KEYWORDS)
    df = df[df["text"].str.lower().str.contains(keyword_pattern)]

    print(f"Filtered audio-related reviews: {len(df)}")

    return df.reset_index(drop=True)


def map_sentiment_label(rating: int) -> str:
    """
    Map star rating to binary sentiment.

    1–2 stars → negative
    3–5 stars → positive
    """

    if rating <= 2:
        return "negative"
    return "positive"


def preprocess() -> pd.DataFrame:
    """
    Full preprocessing pipeline.

    Returns:
        pd.DataFrame: Cleaned and labeled dataset.
    """

    df = load_and_filter_data()

    df["sentiment_label"] = df["rating"].apply(map_sentiment_label)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(f"{PROCESSED_DIR}/clean_reviews.csv", index=False)

    print("Saved clean_reviews.csv")

    return df