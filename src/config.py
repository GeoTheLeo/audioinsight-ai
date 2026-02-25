"""
Central configuration file for
AI-Powered Review Intelligence for Consumer Audio Devices.
"""

# -----------------------------
# Reproducibility
# -----------------------------
RANDOM_STATE = 42

# -----------------------------
# Dataset Configuration
# -----------------------------
DATASET_SAMPLE_SIZE = 50000  # Stable, efficient, sufficient for clustering

# -----------------------------
# Model Names
# -----------------------------
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GENERATION_MODEL = "google/flan-t5-base"

# -----------------------------
# Clustering Configuration
# -----------------------------
N_CLUSTERS = 5  # Corporate-appropriate number of meta-categories

# -----------------------------
# Audio Filtering Keywords
# -----------------------------
AUDIO_KEYWORDS = [
    "headphone",
    "earbud",
    "speaker",
    "bluetooth",
    "soundbar",
    "microphone",
    "audio",
    "turntable",
    "subwoofer",
    "amp",
    "receiver"
]