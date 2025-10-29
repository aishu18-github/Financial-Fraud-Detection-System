# train_time_amount_model.py
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Paths
DATA_CSV = "creditcard.csv"   # ensure this is in your project root
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Load data
print("Loading dataset...")
df = pd.read_csv(DATA_CSV)

# Use only Time and Amount (real features available)
X = df[["Time", "Amount"]].copy()
y = df["Class"].copy()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale Time and Amount
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
X_test_scaled[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

# Train a RandomForest (robust for imbalanced data when tuned; simplified here)
print("Training RandomForest on Time + Amount...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)

# Evaluate
y_pred = rf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")
print("Classification report:")
print(classification_report(y_test, y_pred, digits=4))

# Save model + scaler
joblib.dump(rf, os.path.join(MODELS_DIR, "time_amount_model.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
with open(os.path.join(MODELS_DIR, "time_amount_model_accuracy.txt"), "w") as f:
    f.write(str(acc))

print("Saved time_amount_model.pkl and scaler.pkl to 'models/'")
print("Done.")
