"""
Generative reporting module.
Stable implementation using AutoModelForSeq2SeqLM.
"""

import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.config import GENERATION_MODEL

PROCESSED_DIR = "data/processed"


def load_generation_model():
    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL)
    return tokenizer, model


def generate_text(tokenizer, model, prompt: str) -> str:

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False
        )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


def build_prompt(cluster_id: int, cluster_df: pd.DataFrame) -> str:

    top3 = cluster_df.sort_values("cluster_rank").head(3)
    worst = cluster_df.sort_values(
        "cluster_rank",
        ascending=False
    ).head(1)

    product_block = ""
    for _, row in top3.iterrows():
        product_block += (
            f"ASIN: {row['asin']} | "
            f"Score: {round(row['final_score'], 3)} | "
            f"Reviews: {row['review_count']} | "
            f"Avg Rating: {round(row['avg_rating'], 2)}\n"
        )

    worst_block = ""
    for _, row in worst.iterrows():
        worst_block += (
            f"ASIN: {row['asin']} | "
            f"Score: {round(row['final_score'], 3)} | "
            f"Negative Ratio: {round(row['negative_ratio'], 2)}\n"
        )

    prompt = f"""
You are a professional consumer technology analyst.

Create a complete product recommendation report.

Category ID: {cluster_id}

Top 3 Products:
{product_block}

Worst Product:
{worst_block}

Write two clearly labeled sections:

SECTION 1 — Executive Brief (bullet points, concise)

SECTION 2 — Blog Article (engaging, structured, buying advice)

Generate the full report now.
"""

    return prompt.strip()


def generate_reports(ranked_df: pd.DataFrame):

    print("\nGenerating AI reports...")

    tokenizer, model = load_generation_model()
    cluster_ids = sorted(ranked_df["cluster"].unique())

    all_reports = []

    for cluster_id in cluster_ids:

        cluster_df = ranked_df[
            ranked_df["cluster"] == cluster_id
        ]

        prompt = build_prompt(cluster_id, cluster_df)
        output = generate_text(tokenizer, model, prompt)

        report_text = f"""
====================================
CATEGORY {cluster_id} REPORT
====================================

{output}

"""

        print(report_text)
        all_reports.append(report_text)

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    with open(
        f"{PROCESSED_DIR}/generated_reports.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.writelines(all_reports)

    print("Saved generated_reports.txt")