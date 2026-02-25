"""
Clustering module.

Generates product embeddings and performs KMeans clustering.
"""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from src.config import EMBEDDING_MODEL, N_CLUSTERS, RANDOM_STATE

PROCESSED_DIR = "data/processed"


def filter_products(product_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only products with at least 3 reviews.
    """

    # 🔥 UPDATED THRESHOLD HERE
    filtered = product_df[product_df["review_count"] >= 3].copy()

    print(f"Products with ≥3 reviews: {len(filtered)}")

    return filtered.reset_index(drop=True)


def generate_embeddings(product_df: pd.DataFrame) -> np.ndarray:
    """
    Generate sentence embeddings for combined product text.
    """

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Generating embeddings...")
    embeddings = model.encode(
        product_df["combined_text"].tolist(),
        show_progress_bar=True,
        batch_size=32
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(f"{PROCESSED_DIR}/product_embeddings.npy", embeddings)

    print("Saved product_embeddings.npy")

    return embeddings


def perform_clustering(
    product_df: pd.DataFrame,
    embeddings: np.ndarray
) -> pd.DataFrame:
    """
    Run KMeans clustering and compute silhouette score.
    """

    print("Running KMeans clustering...")

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(embeddings)

    product_df["cluster"] = cluster_labels

    score = silhouette_score(embeddings, cluster_labels)

    print(f"Silhouette Score: {score:.4f}")

    product_df.to_csv(
        f"{PROCESSED_DIR}/clusters.csv",
        index=False
    )

    print("Saved clusters.csv")

    return product_df