import numpy as np
import pandas as pd


class TransactionGenerator:

    def __init__(self, dataset_path):

        try:
            self.df = pd.read_csv(dataset_path)

            if "Class" not in self.df.columns:
                raise ValueError("Dataset must contain a 'Class' column.")

            self.normal_df = self.df[self.df["Class"] == 0]

            if self.normal_df.empty:
                raise ValueError("No normal transactions found in dataset.")

            print("Transaction Generator Loaded")

        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")

    # ----------------------------------------------------------

    def generate(
        self,
        amount,
        hour,
        payment_mode,
        txn_type,
        international,
        new_device
    ):

        base = self.normal_df.sample(n=1).iloc[0].copy()

        # ----------------------------
        # User Inputs
        # ----------------------------

        base["Amount"] = float(amount)
        base["Time"] = int(hour) * 3600

        base["payment_mode"] = payment_mode
        base["is_online"] = 1 if txn_type == "Online" else 0
        base["is_international"] = 1 if international == "Yes" else 0
        base["is_new_device"] = 1 if new_device == "Yes" else 0

        # ----------------------------
        # Risk Estimation
        # ----------------------------

        risk = 0

        if amount >= 100000:
            risk += 6
        elif amount >= 50000:
            risk += 5
        elif amount >= 20000:
            risk += 4
        elif amount >= 10000:
            risk += 3
        elif amount >= 5000:
            risk += 2

        if hour <= 5:
            risk += 3

        if international == "Yes":
            risk += 4

        if new_device == "Yes":
            risk += 3

        if txn_type == "Online":
            risk += 2

        if payment_mode in ["Card", "Wallet"]:
            risk += 1

        # ----------------------------
        # PCA Feature Perturbation
        # ----------------------------

        sigma = 0.05 + (risk * 0.08)

        for i in range(1, 29):

            col = f"V{i}"

            if col in base.index:
                base[col] += np.random.normal(0, sigma)

        # ----------------------------
        # Strong Fraud Simulation
        # ----------------------------

        if risk >= 12:

            features = [
                "V2", "V3", "V4", "V5",
                "V7", "V9", "V10", "V11",
                "V12", "V14", "V16",
                "V17", "V18"
            ]

            factor = np.random.uniform(3.0, 5.0)

        elif risk >= 8:

            features = [
                "V3", "V4", "V7",
                "V10", "V12",
                "V14", "V17"
            ]

            factor = np.random.uniform(2.0, 3.2)

        elif risk >= 5:

            features = [
                "V3",
                "V4",
                "V10",
                "V12"
            ]

            factor = np.random.uniform(1.4, 2.2)

        else:

            features = []
            factor = 1

        for feature in features:

            if feature in base.index:
                base[feature] *= factor

        # Small randomness

        base["Amount"] *= np.random.uniform(0.98, 1.02)

        return base.to_frame().T