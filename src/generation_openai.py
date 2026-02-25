import os
import json
from dotenv import load_dotenv
from openai import OpenAI

PROCESSED_DIR = "data/processed"

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def build_structured_context(cluster_id, cluster_df):

    top3 = cluster_df.sort_values("cluster_rank").head(3)
    worst = cluster_df.sort_values(
        "cluster_rank", ascending=False
    ).head(1)

    context = {
        "category_id": int(cluster_id),
        "top_products": [],
        "worst_product": {}
    }

    for _, row in top3.iterrows():
        context["top_products"].append({
            "asin": str(row["asin"]),
            "score": float(row["final_score"]),
            "review_count": int(row["review_count"]),
            "avg_rating": float(row["avg_rating"]),
            "negative_ratio": float(row["negative_ratio"])
        })

    for _, row in worst.iterrows():
        context["worst_product"] = {
            "asin": str(row["asin"]),
            "score": float(row["final_score"]),
            "negative_ratio": float(row["negative_ratio"])
        }

    return context


def generate_report(cluster_id, cluster_df):

    context = build_structured_context(cluster_id, cluster_df)

    system_prompt = """
You are a senior consumer technology analyst writing for a professional audience.
Write high-quality, polished, publication-ready content.
Be analytical, balanced, and clear.
"""

    user_prompt = f"""
Using the structured data below, create:

1) Executive Brief (concise bullet format)
2) Blog Article (professional, engaging, structured)

Data:
{json.dumps(context, indent=2)}

Include:
- Key strengths
- Key weaknesses
- Differences between top 3
- Buying guidance
- Why the worst product underperforms
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.4
    )

    return response.choices[0].message.content


def generate_reports(ranked_df):

    print("\nGenerating AI reports via OpenAI...")

    cluster_ids = sorted(ranked_df["cluster"].unique())
    all_reports = []

    for cluster_id in cluster_ids:

        cluster_df = ranked_df[
            ranked_df["cluster"] == cluster_id
        ]

        output = generate_report(cluster_id, cluster_df)

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