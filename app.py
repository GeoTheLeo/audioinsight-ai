import streamlit as st
import pandas as pd
import os
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import ListFlowable, ListItem
from reportlab.platypus import Preformatted
from reportlab.lib.pagesizes import LETTER

from src.generation_openai import generate_report

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AudioInsight AI",
    layout="wide"
)

st.title("AI-Powered Review Intelligence")
st.subheader("Consumer Audio Device Intelligence Platform")

# ==========================================
# PUBLIC USAGE INSTRUCTIONS
# ==========================================

st.markdown("""
### How to Use This Platform

1. **Select a product category** from the dropdown menu.  
2. Review the statistically ranked top products.  
3. Click **Generate Executive Report** to produce an AI-written analysis.  
4. Download the report as a PDF if needed.

This platform integrates:
- Transformer-based sentiment analysis  
- Embedding-driven clustering  
- Bayesian product ranking  
- LLM-generated executive insights  
""")

# ==========================================
# LOAD DATA
# ==========================================

RANKED_PATH = "data/processed/ranked_products.csv"

if not os.path.exists(RANKED_PATH):
    st.error("Ranked products file not found. Run main.py locally first.")
    st.stop()

ranked_df = pd.read_csv(RANKED_PATH)

# ==========================================
# CATEGORY NAME MAPPING
# ==========================================

CATEGORY_NAMES = {
    0: "Portable Bluetooth Speakers",
    1: "Car Audio & Connectivity",
    2: "Home Audio & TV Speakers",
    3: "Headphones & Earbuds",
    4: "Smart Speakers & Voice Assistants"
}

ranked_df["category_name"] = ranked_df["cluster"].map(CATEGORY_NAMES)

# ==========================================
# CATEGORY SELECTION
# ==========================================

selected_category = st.selectbox(
    "Select Category",
    sorted(ranked_df["category_name"].unique())
)

cluster_df = ranked_df[ranked_df["category_name"] == selected_category]

# ==========================================
# DISPLAY TOP PRODUCTS
# ==========================================

st.markdown("### Top Ranked Products")

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

# ==========================================
# GENERATE REPORT
# ==========================================

if st.button("Generate Executive Report"):

    with st.spinner("Generating AI analysis..."):
        report = generate_report(
            cluster_df["cluster"].iloc[0],
            cluster_df
        )

    st.markdown("## Executive Report")
    st.markdown(report)

    # ======================================
    # PDF GENERATION
    # ======================================

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=LETTER)

    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("AI-Powered Review Intelligence Report", styles["Heading1"]))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(f"Category: {selected_category}", styles["Heading2"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Preformatted(report, styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)

    st.download_button(
        label="Download Report as PDF",
        data=buffer,
        file_name=f"{selected_category}_report.pdf",
        mime="application/pdf"
    )