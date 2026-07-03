from flask import Flask, render_template, request
import os

from transaction_generator import TransactionGenerator
from config import DATA_PATH
from predict import FraudPredictor

app = Flask(__name__)

print("Loading model...")
predictor = FraudPredictor()
generator = TransactionGenerator(DATA_PATH)
print("System Ready")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    amount = float(request.form["amount"])
    hour = int(request.form["hour"])

    payment_mode = request.form["payment_mode"]
    txn_type = request.form["txn_type"]
    international = request.form["international"]
    new_device = request.form["new_device"]

    df = generator.generate(
        amount,
        hour,
        payment_mode,
        txn_type,
        international,
        new_device
    )

    if "Class" in df.columns:
        df = df.drop(columns=["Class"])

    result = predictor.predict(df).iloc[0].to_dict()

    return render_template("results.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)