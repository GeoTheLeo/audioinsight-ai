"""
Product aggregation module.

Transforms review-level data into product-level intelligence.
"""

import os
import pandas as pd

PROCESSED_DIR = "data/processed"


def aggregate_products(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate review-level data into product-level metrics.

    Returns:
        pd.DataFrame: Product-level dataset.
    """

    print("Aggregating reviews at product level...")

    # -----------------------------
    # Validate title column exists
    # -----------------------------
    if "title" not in df.columns:
        raise ValueError(
            "Column 'title' not found in dataset. "
            "Ensure preprocessing step preserves product title."
        )

    # -----------------------------
    # Group by ASIN
    # -----------------------------
    grouped = df.groupby("asin")

    product_df = grouped.agg(
        title=("title", "first"),  # ← NEW: Preserve product name
        review_count=("asin", "count"),
        avg_rating=("rating", "mean"),
        negative_count=("sentiment_label", lambda x: (x == "negative").sum()),
        positive_count=("sentiment_label", lambda x: (x == "positive").sum()),
        combined_text=("text", lambda x: " ".join(x))
    ).reset_index()

    # -----------------------------
    # Compute negative ratio
    # -----------------------------
    product_df["negative_ratio"] = (
        product_df["negative_count"] / product_df["review_count"]
    )

    print(f"Number of unique products: {len(product_df)}")

    # -----------------------------
    # Save to disk
    # -----------------------------
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    product_df.to_csv(f"{PROCESSED_DIR}/products.csv", index=False)

    print("Saved products.csv")

    return product_df