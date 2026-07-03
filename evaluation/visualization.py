import os
import json
import matplotlib.pyplot as plt
import numpy as np

from config import REPORT_DIR


class ReportGenerator:

    def __init__(self):

        os.makedirs(REPORT_DIR, exist_ok=True)

    ##################################################

    def save_metrics(self, metrics):

        with open(
            os.path.join(REPORT_DIR, "metrics.json"),
            "w"
        ) as f:

            json.dump(metrics, f, indent=4)

    ##################################################

    def save_model_info(self, info):

        with open(
            os.path.join(REPORT_DIR, "model_info.json"),
            "w"
        ) as f:

            json.dump(info, f, indent=4)

    ##################################################

    def plot_confusion_matrix(self, cm):

        plt.figure(figsize=(5,5))

        plt.imshow(cm)

        plt.title("Confusion Matrix")

        plt.colorbar()

        plt.xticks([0,1],["Normal","Fraud"])

        plt.yticks([0,1],["Normal","Fraud"])

        plt.xlabel("Predicted")

        plt.ylabel("Actual")

        for i in range(2):

            for j in range(2):

                plt.text(j,i,str(cm[i,j]),
                         ha="center",
                         va="center")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                REPORT_DIR,
                "confusion_matrix.png"
            )
        )

        plt.close()

    ##################################################

    def plot_fraud_distribution(self, y):

        values = y.value_counts()

        plt.figure(figsize=(5,5))

        plt.pie(
            values,
            labels=["Normal","Fraud"],
            autopct="%1.2f%%"
        )

        plt.title("Fraud Distribution")

        plt.savefig(
            os.path.join(
                REPORT_DIR,
                "fraud_distribution.png"
            )
        )

        plt.close()

    ##################################################

    def plot_cluster_distribution(self, labels):

        plt.figure(figsize=(6,4))

        plt.hist(labels,bins=2)

        plt.title("Cluster Distribution")

        plt.xlabel("Cluster")

        plt.ylabel("Transactions")

        plt.savefig(
            os.path.join(
                REPORT_DIR,
                "cluster_distribution.png"
            )
        )

        plt.close()

    ##################################################

    def plot_anomaly_scores(self,scores):

        plt.figure(figsize=(6,4))

        plt.hist(scores,bins=50)

        plt.title("Isolation Forest Scores")

        plt.xlabel("Score")

        plt.ylabel("Frequency")

        plt.savefig(

            os.path.join(

                REPORT_DIR,

                "anomaly_score_distribution.png"

            )

        )

        plt.close()