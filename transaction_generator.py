import pandas as pd
import numpy as np


class TransactionGenerator:

    def __init__(self, dataset_path):

        self.df = pd.read_csv(dataset_path)

        # Use only genuine transactions as the base
        self.normal_df = self.df[self.df["Class"] == 0]

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

        # Pick a random normal transaction
        base = self.normal_df.sample(1).iloc[0].copy()

        # ------------------------------------------------------
        # User inputs
        # ------------------------------------------------------

        base["Amount"] = float(amount)
        base["Time"] = int(hour) * 3600

        base["payment_mode"] = payment_mode
        base["is_online"] = 1 if txn_type == "Online" else 0
        base["is_international"] = 1 if international == "Yes" else 0
        base["is_new_device"] = 1 if new_device == "Yes" else 0

        # ------------------------------------------------------
        # Risk score used only for synthetic feature generation
        # ------------------------------------------------------

        risk = 0

        # Amount
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

        # Night transactions
        if hour <= 5:
            risk += 3

        # International
        if international == "Yes":
            risk += 4

        # New device
        if new_device == "Yes":
            risk += 3

        # Online
        if txn_type == "Online":
            risk += 2

        # Payment mode
        if payment_mode == "Card":
            risk += 1
        elif payment_mode == "Wallet":
            risk += 1

        # ------------------------------------------------------
        # Add random noise to PCA features
        # ------------------------------------------------------

        sigma = 0.05 + (risk * 0.08)

        for i in range(1, 29):

            feature = f"V{i}"

            if feature in base.index:

                base[feature] += np.random.normal(0, sigma)

        # ------------------------------------------------------
        # Strong anomaly injection
        # ------------------------------------------------------

        if risk >= 12:

            important = [
                "V2", "V3", "V4", "V5", "V7",
                "V9", "V10", "V11", "V12",
                "V14", "V16", "V17", "V18"
            ]

            for feature in important:

                if feature in base.index:

                    base[feature] *= np.random.uniform(3.0, 5.0)

        elif risk >= 8:

            important = [
                "V3", "V4", "V7", "V10",
                "V12", "V14", "V17"
            ]

            for feature in important:

                if feature in base.index:

                    base[feature] *= np.random.uniform(2.0, 3.2)

        elif risk >= 5:

            important = [
                "V3", "V4", "V10", "V12"
            ]

            for feature in important:

                if feature in base.index:

                    base[feature] *= np.random.uniform(1.4, 2.2)

        # ------------------------------------------------------
        # Small amount variation
        # ------------------------------------------------------

        base["Amount"] *= np.random.uniform(0.98, 1.02)

        return base.to_frame().T