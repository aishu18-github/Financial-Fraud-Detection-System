import joblib
import numpy as np
from datetime import datetime

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    ISOLATION_MODEL_PATH,
    KMEANS_MODEL_PATH,
    SCALER_PATH,
    FEATURE_STATS_PATH,
    MODEL_DIR
)

from preprocessing.preprocess import DataPreprocessor
from evaluation.metrics import ModelEvaluator
from evaluation.visualization import ReportGenerator


FEATURE_COLUMNS_PATH = MODEL_DIR + "/feature_columns.pkl"


def main():

    print("=" * 70)
    print("FINANCIAL FRAUD DETECTION TRAINING PIPELINE")
    print("=" * 70)

    ###################################################
    # Load Dataset
    ###################################################

    processor = DataPreprocessor()

    df = processor.preprocess()

    feature_stats = processor.engineer.get_stats()

    X = df.drop(columns=["Class"])
    y = df["Class"]

    print(f"\nDataset Shape : {df.shape}")
    print(f"Features      : {len(X.columns)}")

    ###################################################
    # Save Feature Order
    ###################################################

    feature_columns = list(X.columns)

    joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)

    print("\nFeature order saved.")

    ###################################################
    # Split
    ###################################################

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=0.2,
        random_state=42
    )

    ###################################################
    # Train only on normal transactions
    ###################################################

    X_train_normal = X_train[y_train == 0]

    ###################################################
    # Scaling
    ###################################################

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_normal)

    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, SCALER_PATH)

    joblib.dump(feature_stats, FEATURE_STATS_PATH)

    ###################################################
    # Isolation Forest
    ###################################################

    contamination = max(
        0.001,
        min(0.02, y.mean())
    )

    print(f"\nUsing contamination = {contamination}")

    isolation = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    isolation.fit(X_train_scaled)

    ###################################################
    # KMeans
    ###################################################

    kmeans = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_train_scaled)

    ###################################################
    # Predictions
    ###################################################

    prediction = isolation.predict(X_test_scaled)

    prediction = np.where(
        prediction == -1,
        1,
        0
    )

    scores = isolation.decision_function(X_test_scaled)

    cluster_labels = kmeans.predict(X_test_scaled)

    ###################################################
    # Evaluation
    ###################################################

    evaluator = ModelEvaluator()

    metrics, cm = evaluator.evaluate(
        y_test,
        prediction
    )

    ###################################################
    # Reports
    ###################################################

    report = ReportGenerator()

    report.save_metrics(metrics)

    report.plot_confusion_matrix(cm)

    report.plot_fraud_distribution(y)

    report.plot_cluster_distribution(cluster_labels)

    report.plot_anomaly_scores(scores)

    ###################################################
    # Save Model Info
    ###################################################

    model_info = {

        "Algorithm": "Isolation Forest",

        "Secondary Model": "KMeans",

        "Training Date": str(datetime.now()),

        "Training Samples": len(X_train),

        "Testing Samples": len(X_test),

        "Features": feature_columns,

        "Total Features": len(feature_columns),

        "Contamination": contamination

    }

    report.save_model_info(model_info)

    ###################################################
    # Save High Risk
    ###################################################

    results = X_test.copy()

    results["Actual"] = y_test.values

    results["Prediction"] = prediction

    results["IsolationScore"] = scores

    results["Cluster"] = cluster_labels

    results[results["Prediction"] == 1].to_csv(
        "reports/high_risk_transactions.csv",
        index=False
    )

    ###################################################
    # Save Models
    ###################################################

    joblib.dump(
        isolation,
        ISOLATION_MODEL_PATH
    )

    joblib.dump(
        kmeans,
        KMEANS_MODEL_PATH
    )

    print("\nModels Saved Successfully.")

    print("\nTraining Completed Successfully.")

    print("=" * 70)


if __name__ == "__main__":
    main()