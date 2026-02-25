"""
Cluster interpretation module.

Extracts TF-IDF keywords per cluster
and identifies representative products.
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

PROCESSED_DIR = "data/processed"


def interpret_clusters(clustered_df: pd.DataFrame) -> None:
    """
    Generate cluster summaries using TF-IDF and save results.
    """

    print("\nInterpreting clusters...")

    summaries = []

    for cluster_id in sorted(clustered_df["cluster"].unique()):

        cluster_subset = clustered_df[clustered_df["cluster"] == cluster_id]

        # TF-IDF on combined text of cluster
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=20
        )

        tfidf_matrix = vectorizer.fit_transform(
            cluster_subset["combined_text"]
        )

        terms = vectorizer.get_feature_names_out()
        mean_scores = tfidf_matrix.mean(axis=0).A1

        top_indices = mean_scores.argsort()[-10:][::-1]
        top_terms = [terms[i] for i in top_indices]

        # Representative products (highest review_count)
        representative_products = cluster_subset.sort_values(
            "review_count",
            ascending=False
        ).head(3)

        summary_text = f"""
===============================
Cluster {cluster_id}
===============================
Number of Products: {len(cluster_subset)}

Top Keywords:
{", ".join(top_terms)}

Representative Products (ASIN + review_count):
"""

        for _, row in representative_products.iterrows():
            summary_text += f"- {row['asin']} ({row['review_count']} reviews)\n"

        print(summary_text)
        summaries.append(summary_text)

    # Save to file
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    with open(f"{PROCESSED_DIR}/cluster_summary.txt", "w", encoding="utf-8") as f:
        f.writelines(summaries)

    print("Saved cluster_summary.txt")