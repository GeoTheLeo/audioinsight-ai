import streamlit as st
import pandas as pd
import os
from src.generation_openai import generate_report

st.set_page_config(
    page_title="AudioInsight AI",
    layout="wide"
)

st.title("AI-Powered Review Intelligence")
st.subheader("Consumer Audio Device Intelligence Platform")

st.markdown("""
This platform integrates:

- Transformer-based Sentiment Analysis  
- Embedding-Based Clustering  
- Bayesian Product Ranking  
- LLM-Generated Executive Reports  

Select a product category below to explore insights.
""")

RANKED_PATH = "data/processed/ranked_products.csv"

if not os.path.exists(RANKED_PATH):
    st.error("Ranked products file not found. Run main.py locally first.")
    st.stop()

ranked_df = pd.read_csv(RANKED_PATH)

clusters = sorted(ranked_df["cluster"].unique())

selected_cluster = st.selectbox(
    "Select Product Category",
    clusters
)

cluster_df = ranked_df[ranked_df["cluster"] == selected_cluster]

st.markdown("### Top Products")

top_products = cluster_df.sort_values("cluster_rank").head(5)

st.dataframe(
    top_products[
        [
            "asin",
            "final_score",
            "avg_rating",
            "review_count",
            "negative_ratio"
        ]
    ],
    use_container_width=True
)

if st.button("Generate Executive Report"):

    with st.spinner("Generating AI analysis..."):
        report = generate_report(selected_cluster, cluster_df)

    st.markdown("## AI-Generated Executive Report")
    st.markdown(report)