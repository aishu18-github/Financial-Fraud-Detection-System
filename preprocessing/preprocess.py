import pandas as pd

from config import DATA_PATH

from feature_engineering.feature_engineering import FeatureEngineer


class DataPreprocessor:

    def __init__(self):

        self.engineer = FeatureEngineer()

    def load_data(self):

        return pd.read_csv(DATA_PATH)

    def preprocess(self):

        print("=" * 60)
        print("Loading Dataset...")
        print("=" * 60)

        df = self.load_data()

        print(f"Dataset Shape : {df.shape}")

        before = len(df)

        df = df.drop_duplicates()

        after = len(df)

        print(f"Duplicates Removed : {before-after}")

        if df.isnull().sum().sum() > 0:

            df = df.fillna(df.median(numeric_only=True))

            print("Missing values handled.")

        else:

            print("No Missing Values.")

        print("Performing Feature Engineering...")

        df = self.engineer.transform(df)

        return df