from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


class ModelEvaluator:

    def evaluate(self, y_true, y_pred):

        metrics = {

            "accuracy": accuracy_score(y_true, y_pred),

            "precision": precision_score(y_true, y_pred),

            "recall": recall_score(y_true, y_pred),

            "f1_score": f1_score(y_true, y_pred)

        }

        print("\n" + "="*60)

        print("MODEL EVALUATION")

        print("="*60)

        for key, value in metrics.items():

            print(f"{key:12}: {value:.4f}")

        print()

        print(classification_report(y_true, y_pred))

        cm = confusion_matrix(y_true, y_pred)

        return metrics, cm