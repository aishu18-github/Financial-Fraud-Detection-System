# 🛡️ Financial Fraud Detection System

A financial fraud detection system that analyzes transaction data using Machine Learning and flags suspicious activity instantly. Includes a responsive UI for monitoring alerts and performance analytics.

---

## 🚀 Features
- ✅ Fraud prediction using trained ML model  
- 📊 Interactive dashboard to visualize fraud activity  
- 🔍 Secure API with request validation  
- 💾 Model + pre-processing pipeline included  
- 🖥️ Clean and responsive frontend interface  

---

## 🧰 Tech Stack

### **Backend**
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-API-success)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Model-orange)

### **Frontend**
![HTML5](https://img.shields.io/badge/HTML5-UI-yellow)
![CSS3](https://img.shields.io/badge/CSS3-Design-blue)
![JavaScript](https://img.shields.io/badge/JavaScript-ES6-yellow)
![Chart.js](https://img.shields.io/badge/Chart.js-Visualizations-pink)

---
## 🛠️ Setup & Installation
### 1️⃣ Clone Repository
```bash
git clone https://github.com/aishu18-github/Financial-Fraud-Detection-System.git
```
### 2️⃣ Create & Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate # Mac/Linux
venv\Scripts\activate # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application
```bash
python app.py
```


### 5️⃣ Open in Browser
```bash
http://127.0.0.1:5000
```

## 🧪 API Testing
```bash
curl -X POST http://127.0.0.1:5000/predict

-H "Content-Type: application/json"
-d "{"Time":10,"Amount":120.50,"V1":-1.2,"V2":0.45}"
```
## 🙌 Acknowledgements
Dataset derived from public Credit Card Fraud Detection datasets for academic & research usage only.

## ⭐ Support
If you like this project, please ⭐ the repo!
