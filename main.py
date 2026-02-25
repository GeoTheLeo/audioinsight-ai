"""
Main execution pipeline for
AI-Powered Review Intelligence for Consumer Audio Devices.
"""

from src.preprocessing import preprocess
from src.sentiment import evaluate_sentiment_model
from src.aggregation import aggregate_products
from src.clustering import (
    filter_products,
    generate_embeddings,
    perform_clustering
)
from src.cluster_interpretation import interpret_clusters
from src.ranking import (
    compute_bayesian_score,
    apply_sentiment_penalty,
    rank_within_clusters
)
from src.generation_openai import generate_reports


def main():

    print("=== PHASE 2: PREPROCESSING ===")
    df = preprocess()

    print("\n=== PHASE 2B: SENTIMENT EVALUATION ===")
    evaluate_sentiment_model(df)

    print("\n=== PHASE 3: PRODUCT AGGREGATION ===")
    product_df = aggregate_products(df)

    print("\n=== PHASE 4: CLUSTERING ===")
    product_df = filter_products(product_df)
    embeddings = generate_embeddings(product_df)
    clustered_df = perform_clustering(product_df, embeddings)

    print("\n=== PHASE 4B: CLUSTER INTERPRETATION ===")
    interpret_clusters(clustered_df)

    print("\n=== PHASE 5: RANKING ENGINE ===")
    ranked_df = compute_bayesian_score(clustered_df)
    ranked_df = apply_sentiment_penalty(ranked_df)
    ranked_df = rank_within_clusters(ranked_df)

    print("\n=== PHASE 6: GENERATIVE REPORTING ===")
    generate_reports(ranked_df)

    print("\nPipeline complete.")
    print("Artifacts saved in data/processed/")


if __name__ == "__main__":
    main()