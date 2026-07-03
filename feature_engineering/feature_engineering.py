import numpy as np
import pandas as pd


class FeatureEngineer:

    def __init__(self, stats=None):
        self.stats = stats

    def transform(self, df):

        data = df.copy()

        # -------------------------
        # FORCE numeric safety
        # -------------------------
        data["Amount"] = pd.to_numeric(data["Amount"], errors="coerce").fillna(0)
        data["Time"] = pd.to_numeric(data["Time"], errors="coerce").fillna(0)

        # -------------------------
        # basic transformations
        # -------------------------
        data["LogAmount"] = np.log1p(data["Amount"].values)   # ✅ FIX HERE

        data["Hour"] = ((data["Time"] // 3600) % 24).astype(int)

        # -------------------------
        # payment feature safety
        # -------------------------
        if "payment_mode" in data.columns:

            payment_map = {
                "UPI": 0.2,
                "Wallet": 0.3,
                "Card": 0.6,
                "Netbanking": 0.5
            }

            data["payment_mode_score"] = (
                data["payment_mode"]
                .astype(str)
                .map(payment_map)
                .fillna(0.3)
            )

        else:
            data["payment_mode_score"] = 0.3

        # -------------------------
        # binary flags safety
        # -------------------------
        for col in ["is_online", "is_international", "is_new_device"]:
            if col not in data.columns:
                data[col] = 0
            else:
                data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

        # -------------------------
        # stats (training only)
        # -------------------------
        if self.stats is None:
            self.stats = {
                "amount_mean": data["Amount"].mean(),
                "amount_std": data["Amount"].std(),
                "high_amount_threshold": data["Amount"].quantile(0.95)
            }

        return data

    def get_stats(self):
        return self.stats