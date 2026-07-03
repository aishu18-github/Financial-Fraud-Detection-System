from flask import Flask, render_template, request

from config import DATA_PATH
from predict import FraudPredictor
from transaction_generator import TransactionGenerator

app = Flask(__name__)

print("Loading Fraud Detection System...")

predictor = FraudPredictor()
generator = TransactionGenerator(DATA_PATH)

print("System Ready")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        amount = float(request.form["amount"])
        hour = int(request.form["hour"])

        payment_mode = request.form["payment_mode"]
        txn_type = request.form["txn_type"]
        international = request.form["international"]
        new_device = request.form["new_device"]

        transaction = generator.generate(
            amount,
            hour,
            payment_mode,
            txn_type,
            international,
            new_device
        )

        if "Class" in transaction.columns:
            transaction = transaction.drop(columns=["Class"])

        result = predictor.predict(transaction)

        return render_template(
            "results.html",
            result=result.iloc[0].to_dict()
        )

    except Exception as e:

        print(e)

        return render_template(
            "results.html",
            result={
                "Prediction": "Error",
                "RiskScore": 0,
                "RiskLevel": "Low",
                "Recommendation": "Unable to analyze transaction",
                "Reasons": [str(e)]
            }
        )


if __name__ == "__main__":
    app.run(debug=True)