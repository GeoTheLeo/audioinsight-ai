"""
Ranking module.

Computes Bayesian-adjusted product scores
and ranks products within each cluster.
"""

import os
import pandas as pd

PROCESSED_DIR = "data/processed"


def compute_bayesian_score(product_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Bayesian-adjusted weighted rating.
    """

    print("Computing Bayesian scores...")

    C = product_df["avg_rating"].mean()
    m = product_df["review_count"].median()

    v = product_df["review_count"]
    R = product_df["avg_rating"]

    product_df["bayesian_rating"] = (
        (v / (v + m)) * R +
        (m / (v + m)) * C
    )

    return product_df


def apply_sentiment_penalty(product_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply penalty based on negative review ratio.
    """

    product_df["sentiment_penalty"] = 1 - product_df["negative_ratio"]

    product_df["final_score"] = (
        product_df["bayesian_rating"] *
        product_df["sentiment_penalty"]
    )

    return product_df


def rank_within_clusters(product_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank products inside each cluster.
    """

    product_df["cluster_rank"] = (
        product_df
        .groupby("cluster")["final_score"]
        .rank(ascending=False, method="first")
    )

    product_df.to_csv(
        f"{PROCESSED_DIR}/ranked_products.csv",
        index=False
    )

    print("Saved ranked_products.csv")

    return product_df