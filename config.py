import os

# ==============================
# Project Directories
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

# ==============================
# Files
# ==============================

DATA_PATH = os.path.join(DATA_DIR, "creditcard.csv")

ISOLATION_MODEL_PATH = os.path.join(MODEL_DIR, "isolation_forest.pkl")
KMEANS_MODEL_PATH = os.path.join(MODEL_DIR, "kmeans.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_STATS_PATH = os.path.join(MODEL_DIR, "feature_stats.pkl")

# ==============================
# Create folders automatically
# ==============================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)