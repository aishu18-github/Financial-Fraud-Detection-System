from flask import Flask, render_template, request
from datetime import datetime

app = Flask(__name__)

# Transaction type risk multiplier
TYPE_RISK = {
    "UPI": 0.8,
    "Card": 1.0,
    "International": 1.6,
    "Wallet": 1.8
}

def parse_datetime_local(dt_str):
    if not dt_str:
        return {"hour": None, "seconds": 0, "raw": ""}

    dt_str = dt_str.replace(" ", "T")
    try:
        dt = datetime.fromisoformat(dt_str)
    except Exception:
        try:
            dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
        except Exception:
            return {"hour": None, "seconds": 0, "raw": dt_str}

    seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
    return {"hour": dt.hour, "seconds": seconds, "raw": dt.strftime("%Y-%m-%d %H:%M:%S")}


def compute_risk(amount: float, hour: int, tx_type: str):
    breakdown = []
    amt = float(amount or 0)
    tx_type = (tx_type or "UPI").strip()

    # ✅ Amount scoring tuned to trigger high fraud for ₹45k+ etc.
    if amt <= 1000:
        amt_score = 5;  expl_amt = "Very small amount—low risk"
    elif amt <= 5000:
        amt_score = 12; expl_amt = "Small amount—low risk"
    elif amt <= 20000:
        amt_score = 25; expl_amt = "Moderate amount—moderate risk"
    elif amt <= 50000:
        amt_score = 48; expl_amt = "High amount—likely fraud"
    else:
        amt_score = 70; expl_amt = "Extremely high amount—major fraud alert"

    # ✅ Time scoring - heavier penalty at night
    if hour is None:
        time_score = 0; expl_time = "Time not available"
    elif 0 <= hour <= 4:
        time_score = 40; expl_time = "Late-night transaction—very suspicious"
    elif 5 <= hour <= 7:
        time_score = 10; expl_time = "Very early morning—unusual"
    elif hour >= 23:
        time_score = 18; expl_time = "Late-night activity—suspicious"
    else:
        time_score = 5;  expl_time = "Normal working hours"

    # ✅ Transaction type scoring
    if tx_type == "UPI":
        type_score = 10; expl_type = "UPI—trusted channel"
    elif tx_type == "Card":
        type_score = 15; expl_type = "Card—moderate risk"
    elif tx_type == "International":
        type_score = 50; expl_type = "International—high fraud risk"
    else:
        type_score = 35; expl_type = "Wallet—elevated risk"

    # ✅ Amplifying Conditions
    interaction_bonus = 0
    interaction_expl = ""
    if amt > 20000 and tx_type == "International":
        interaction_bonus += 20
        interaction_expl = "Large International Transaction—critical alert"
    elif amt > 20000 and (hour is not None and hour <= 5):
        interaction_bonus += 18
        interaction_expl = "High amount at late-night—critical alert"

    # ✅ Final score
    risk_pct = amt_score * 0.45 + time_score * 0.25 + type_score * 0.25 + interaction_bonus * 0.05
    risk_pct = max(0.0, min(100.0, risk_pct))

    breakdown.extend([
        {"name": "Amount", "value": amt_score, "contribution": round(amt_score * 0.45, 2), "explanation": expl_amt},
        {"name": "Time", "value": time_score, "contribution": round(time_score * 0.25, 2), "explanation": expl_time},
        {"name": "Transaction Type", "value": type_score, "contribution": round(type_score * 0.25, 2), "explanation": expl_type},
    ])
    if interaction_bonus:
        breakdown.append({
            "name": "Interaction",
            "value": interaction_bonus,
            "contribution": round(interaction_bonus * 0.05, 2),
            "explanation": interaction_expl
        })

    # ✅ Thresholds adjusted
    if risk_pct >= 65:
        verdict = "🚨 Fraud Very Likely"
    elif risk_pct >= 45:
        verdict = "⚠ Suspicious — Verify"
    else:
        verdict = "✅ Safe"

    return {
        "risk_pct": round(risk_pct, 2),
        "verdict": verdict,
        "breakdown": breakdown
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        amount = float(request.form.get("transaction_amount", "0"))
        tx_type = request.form.get("transaction_type", "UPI")
        parsed = parse_datetime_local(request.form.get("transaction_time", ""))
        
        result = compute_risk(amount, parsed["hour"], tx_type)

        return render_template(
            "result.html",
            prediction_text=f"{result['verdict']} — Risk {result['risk_pct']}%",
            risk_pct=result["risk_pct"],
            amount=amount,
            time_str=parsed["raw"],
            transaction_type=tx_type,
            breakdown=result["breakdown"]
        )
    except Exception as e:
        return render_template(
            "result.html",
            prediction_text=f"Error: {str(e)}",
            risk_pct=0,
            amount=0,
            time_str="",
            transaction_type="",
            breakdown=[]
        )


if __name__ == "__main__":
    app.run(debug=True)
