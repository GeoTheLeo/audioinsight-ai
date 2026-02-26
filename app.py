import streamlit as st
import pandas as pd
import io
from src.generation_openai import generate_report

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI-Powered Review Intelligence",
    layout="wide"
)

st.title("AI-Powered Review Intelligence for Consumer Audio Devices")

st.markdown(
"""
### How to Use This App

1. Select a product category.
2. Review the top-ranked products based on sentiment-adjusted scoring.
3. Click **Generate Executive Report** for AI-powered analysis.
4. Download the report as a professional PDF.

This system combines sentiment modeling, product clustering, Bayesian ranking,
and generative AI to deliver structured product intelligence.
"""
)

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/ranked_products.csv")

df = load_data()

# -----------------------------
# Human-Friendly Category Names
# -----------------------------
CATEGORY_MAP = {
    0: "Portable Bluetooth Speakers",
    1: "Car Audio & Radio Devices",
    2: "Home / TV Speaker Systems",
    3: "Headphones & Earbuds",
    4: "Smart Speakers (Alexa / Echo)"
}

df["Category Name"] = df["cluster"].map(CATEGORY_MAP)

# -----------------------------
# Category Selection
# -----------------------------
selected_category = st.selectbox(
    "Select Category:",
    sorted(df["Category Name"].unique())
)

cluster_id = [
    key for key, value in CATEGORY_MAP.items()
    if value == selected_category
][0]

cluster_df = df[df["cluster"] == cluster_id]

# -----------------------------
# Show Top Ranked Products
# -----------------------------
st.subheader("Top Ranked Products")

top_products = (
    cluster_df
    .sort_values("cluster_rank")
    .head(5)
)

# Create display column
top_products["Product"] = (
    top_products["title"] + " (" + top_products["asin"] + ")"
)

display_df = top_products[
    [
        "Product",
        "final_score",
        "avg_rating",
        "review_count",
        "negative_ratio",
    ]
].copy()

display_df = display_df.rename(
    columns={
        "final_score": "Score",
        "avg_rating": "Avg Rating",
        "review_count": "Reviews",
        "negative_ratio": "Negative Ratio",
    }
)

st.dataframe(display_df, use_container_width=True)

# -----------------------------
# Generate Executive Report
# -----------------------------
st.subheader("AI Executive Report")

if st.button("Generate Executive Report"):
    with st.spinner("Generating AI-powered report..."):
        report_text = generate_report(cluster_id, cluster_df)

    st.markdown(report_text)

    # -----------------------------
    # PDF Download
    # -----------------------------
    pdf_buffer = io.BytesIO()
    pdf_buffer.write(report_text.encode("utf-8"))
    pdf_buffer.seek(0)

    st.download_button(
        label="Download Report as PDF",
        data=pdf_buffer,
        file_name=f"{selected_category}_Report.pdf",
        mime="application/pdf"
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "Built with Sentiment Analysis, Clustering, Bayesian Ranking, and Generative AI."
)