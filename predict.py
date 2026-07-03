import joblib
import numpy as np

from config import (
    ISOLATION_MODEL_PATH,
    KMEANS_MODEL_PATH,
    SCALER_PATH,
    FEATURE_STATS_PATH
)

from feature_engineering.feature_engineering import FeatureEngineer


class FraudPredictor:

    def __init__(self):

        print("Loading models...")

        self.model = joblib.load(ISOLATION_MODEL_PATH)
        self.kmeans = joblib.load(KMEANS_MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.feature_stats = joblib.load(FEATURE_STATS_PATH)
        self.feature_columns = joblib.load("models/feature_columns.pkl")

        self.engineer = FeatureEngineer(self.feature_stats)

        print("Models loaded successfully.\n")

    # ----------------------------------------------------

    def preprocess(self, df):

        processed = self.engineer.transform(df.copy())

        processed = processed.reindex(
            columns=self.feature_columns,
            fill_value=0
        )

        scaled = self.scaler.transform(processed)

        return scaled, processed

    # ----------------------------------------------------

    @staticmethod
    def get_risk_level(score):

        if score >= 80:
            return "Critical"
        elif score >= 60:
            return "High"
        elif score >= 30:
            return "Medium"
        else:
            return "Low"

    # ----------------------------------------------------

    @staticmethod
    def get_recommendation(level):

        mapping = {
            "Critical": "BLOCK TRANSACTION",
            "High": "MANUAL REVIEW",
            "Medium": "VERIFY USER",
            "Low": "APPROVE"
        }

        return mapping[level]

    # ----------------------------------------------------

    def explain_risk(self, row):

        reasons = []

        amount = float(row["Amount"])
        hour = int(row["Time"] // 3600)

        if amount > 100000:
            reasons.append("Extremely high transaction amount")
        elif amount > 50000:
            reasons.append("High transaction amount")
        elif amount > 20000:
            reasons.append("Large transaction amount")

        if hour <= 5:
            reasons.append("Transaction during unusual hours")

        if row.get("is_international", 0) == 1:
            reasons.append("International transaction")

        if row.get("is_new_device", 0) == 1:
            reasons.append("Transaction from a new device")

        if row.get("is_online", 0) == 1:
            reasons.append("Online transaction")

        if len(reasons) == 0:
            reasons.append("Transaction pattern appears normal")

        return reasons

    # ----------------------------------------------------

    def predict(self, df):

        original_df = df.copy()

        X_scaled, processed_df = self.preprocess(df)

        # Isolation Forest prediction
        pred = self.model.predict(X_scaled)
        pred = np.where(pred == -1, 1, 0)

        scores = self.model.decision_function(X_scaled)

        cluster = self.kmeans.predict(X_scaled)

        # Since only ONE transaction is predicted
        row = original_df.iloc[0]

        amount = float(row["Amount"])
        hour = int(row["Time"] // 3600)

        rule_score = 0

        # ---------------- Amount ----------------

        if amount > 100000:
            rule_score += 40
        elif amount > 50000:
            rule_score += 30
        elif amount > 20000:
            rule_score += 20
        elif amount > 10000:
            rule_score += 10

        # ---------------- Time ----------------

        if hour <= 5:
            rule_score += 15

        # ---------------- International ----------------

        if row.get("is_international", 0) == 1:
            rule_score += 20

        # ---------------- New Device ----------------

        if row.get("is_new_device", 0) == 1:
            rule_score += 15

        # ---------------- Online ----------------

        if row.get("is_online", 0) == 1:
            rule_score += 10

        # ---------------- ML Contribution ----------------

        if pred[0] == 1:
            ml_score = min(100, 70 + abs(scores[0]) * 120)
        else:
            ml_score = max(10, 30 - scores[0] * 50)

        # ---------------- Final Hybrid Score ----------------

        final_score = 0.6 * rule_score + 0.4 * ml_score
        final_score = float(np.clip(final_score, 0, 100))

        risk_level = self.get_risk_level(final_score)

        prediction = "Fraud" if final_score >= 60 else "Normal"

        recommendation = self.get_recommendation(risk_level)

        reasons = self.explain_risk(row)

        # ---------------- Output ----------------

        processed_df["Prediction"] = prediction
        processed_df["RiskScore"] = round(final_score, 2)
        processed_df["RiskLevel"] = risk_level
        processed_df["Recommendation"] = recommendation
        processed_df["Cluster"] = int(cluster[0])
        processed_df["IsolationScore"] = round(float(scores[0]), 4)
        processed_df["Reasons"] = [reasons]

        return processed_df