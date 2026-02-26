import os
import json
import pandas as pd
import streamlit as st
from openai import OpenAI


# ================================
# OPENAI CLIENT LOADER
# ================================

def get_openai_client():
    """
    Loads OpenAI API key from:
    1. Local environment variable
    2. Streamlit Cloud secrets
    """

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            raise ValueError("OPENAI_API_KEY not found in environment or Streamlit secrets.")

    return OpenAI(api_key=api_key)


# ================================
# JSON SERIALIZATION FIX
# ================================

def convert_numpy(obj):
    """
    Converts numpy types to native Python types
    so they can be JSON serialized.
    """
    if hasattr(obj, "item"):
        return obj.item()
    return obj


# ================================
# REPORT GENERATION
# ================================

def generate_report(cluster_id, cluster_df):
    """
    Generates executive + blog-style report
    for a specific product cluster.
    """

    client = get_openai_client()

    # Sort cluster
    cluster_df = cluster_df.sort_values("cluster_rank")

    # Top 3 products
    top_products = cluster_df.head(3)

    # Worst product (lowest final score)
    worst_product = cluster_df.sort_values("final_score").head(1)

    context = {
        "cluster_id": int(cluster_id),
        "top_products": top_products[
            ["asin", "final_score", "review_count", "avg_rating", "negative_ratio"]
        ].applymap(convert_numpy).to_dict(orient="records"),
        "worst_product": worst_product[
            ["asin", "final_score", "review_count", "avg_rating", "negative_ratio"]
        ].applymap(convert_numpy).to_dict(orient="records")[0],
    }

    prompt = f"""
You are an AI business analyst.

Using the structured product data below, generate:

SECTION 1 — Executive Brief (bullet points, concise)
SECTION 2 — Blog Article (engaging, structured, buying advice)

Context:
{json.dumps(context, indent=2)}

Requirements:
- Explain key strengths
- Explain weaknesses
- Compare top 3 products clearly
- Provide buying guidance
- Explain why worst product should be avoided
- Professional corporate tone
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professional business intelligence analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=900
    )

    return response.choices[0].message.content


# ================================
# BULK GENERATION (Optional CLI use)
# ================================

def generate_reports(ranked_df):
    """
    Generates reports for all clusters.
    Used in main.py pipeline.
    """

    print("\nGenerating AI reports via OpenAI...")

    clusters = sorted(ranked_df["cluster"].unique())

    full_output = ""

    for cluster_id in clusters:
        cluster_df = ranked_df[ranked_df["cluster"] == cluster_id]

        print(f"Generating report for cluster {cluster_id}...")

        report = generate_report(cluster_id, cluster_df)

        full_output += (
            "\n====================================\n"
            f"CATEGORY {cluster_id} REPORT\n"
            "====================================\n\n"
            f"{report}\n\n"
        )

    output_path = "data/processed/generated_reports.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_output)

    print("Saved generated_reports.txt")

    return full_output