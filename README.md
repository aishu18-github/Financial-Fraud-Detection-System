# 🛡️ Financial Fraud Detection System

A Machine Learning-powered Financial Fraud Detection System built using **Python, Flask, Isolation Forest, and K-Means Clustering**. The application analyzes transaction details entered through a web interface and predicts whether a transaction is likely to be fraudulent, along with a risk score, risk level, recommendation, and explanation.

---

# 📌 Project Overview

This project simulates a real-world fraud detection system used in banking and digital payment platforms.

The user enters transaction details such as:

- Transaction Amount
- Transaction Time
- Payment Mode
- Transaction Type
- International Transaction
- New Device

The system generates a realistic transaction, performs feature engineering, applies trained Machine Learning models, and returns a fraud analysis report.

---

# 🚀 Features

- ✅ Fraud Detection using Isolation Forest
- ✅ Customer Segmentation using K-Means Clustering
- ✅ Hybrid Risk Scoring (Machine Learning + Rule-Based)
- ✅ Automatic Feature Engineering
- ✅ Transaction Risk Explanation
- ✅ Risk Levels (Low / Medium / High / Critical)
- ✅ Intelligent Recommendations
- ✅ Clean Flask Web Interface
- ✅ Training Pipeline Included
- ✅ Report Generation

---

# 🧠 Machine Learning Models

### Isolation Forest

Used for anomaly detection by identifying unusual transaction patterns.

### K-Means Clustering

Used for transaction clustering to identify behavioral similarities.

### Feature Engineering

Additional features generated include:

- LogAmount
- Hour
- Payment Mode Score
- Online Transaction Flag
- International Transaction Flag
- New Device Flag

---

# 🖥️ Application Workflow

```text
User Input
      │
      ▼
Transaction Generator
      │
      ▼
Feature Engineering
      │
      ▼
Feature Scaling
      │
      ▼
Isolation Forest
      │
      ▼
K-Means Clustering
      │
      ▼
Hybrid Risk Score
      │
      ▼
Prediction + Recommendation + Explanation
```

---

# 🛠 Tech Stack

## Backend

- Python
- Flask

## Machine Learning

- Scikit-Learn
- Isolation Forest
- K-Means
- NumPy
- Pandas

## Frontend

- HTML
- CSS
- Jinja2 Templates

---

# 📂 Project Structure

```
Financial-Fraud-Detection-System/
│
├── app.py
├── train.py
├── predict.py
├── transaction_generator.py
├── config.py
├── requirements.txt
│
├── feature_engineering/
│   └── feature_engineering.py
│
├── preprocessing/
│   └── preprocess.py
│
├── evaluation/
│   ├── metrics.py
│   └── visualization.py
│
├── templates/
│   ├── index.html
│   └── results.html
│
├── static/
│
├── models/
│
├── reports/
│
└── README.md
```

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/aishu18-github/Financial-Fraud-Detection-System.git

cd Financial-Fraud-Detection-System
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv fraud-env

fraud-env\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv fraud-env

source fraud-env/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Add Dataset

Download the **Credit Card Fraud Detection Dataset** from Kaggle:

Place the dataset as:

```
data/
    creditcard.csv
```

> The dataset is not included in this repository because of its large size.

---

## 5. Train the Models

```bash
python train.py
```

This generates:

- Isolation Forest Model
- K-Means Model
- Standard Scaler
- Feature Statistics
- Feature Order
- Evaluation Reports

---

## 6. Run the Application

```bash
python app.py
```

---

## 7. Open in Browser

```
http://127.0.0.1:5000
```

---

# 📷 Application Output

The application displays:

- Fraud Prediction
- Risk Score
- Risk Level
- Recommendation
- Reasons Behind Prediction

---

# 📊 Reports Generated

During training, the following reports are automatically generated:

- Confusion Matrix
- Fraud Distribution
- Cluster Distribution
- Isolation Forest Score Distribution
- High Risk Transactions
- Evaluation Metrics

---

# 📈 Future Improvements

- Deep Learning based Fraud Detection
- SHAP Explainability
- Real-time Transaction Monitoring
- User Authentication
- REST API
- Docker Deployment
- Cloud Deployment (Render / Railway / Azure)

---


# 👨‍💻 Author

**Aishwarya G**

GitHub:
https://github.com/aishu18-github

---

# ⭐ If you found this project useful

Please consider giving the repository a ⭐ on GitHub.
