# Customer Churn Prediction – Streamlit App

Project ini bertujuan untuk memprediksi kemungkinan pelanggan
melakukan **churn** (berhenti berlangganan) menggunakan pendekatan
*machine learning classification*.

Aplikasi dibangun menggunakan **Streamlit** dan menampilkan:
- Halaman deskripsi project
- Exploratory Data Analysis (EDA)
- Halaman prediksi churn interaktif

---

## 📌 Fitur Utama
- Multi-page Streamlit application
- Exploratory Data Analysis dengan visualisasi
- Model Logistic Regression
- Threshold decision slider
- Risk level interpretasi (Low / Medium / High)
- Model inference menggunakan `joblib`

---

## 📁 Struktur Folder
```
STREAMLIT_PROJECT
├── app.py
├── train_model.py
├── README.md
├── requirements.txt
├── data/
│ └── churn.csv
├── model/
│ ├── churn_model.joblib
│ ├── encoder.joblib
│ └── scaler.joblib
├── pages/
│ ├── tentang.py
│ ├── eda.py
│ └── prediksi.py
```
---

## 🚀 Cara Menjalankan Aplikasi (Local)
1. Clone repository ini
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Jalankan aplikasi:
    ```bash
    streamlit run app.py
    ```
---

## 🌐 Demo Aplikasi
Aplikasi dapat diakses melalui link berikut:

👉 Streamlit App – Customer Churn Prediction


## 👩‍💻 Author
*Deasy R Alwani*

## 📫 Kontak
LinkedIn: https://linkedin.com/in/deasyalwani  
Email: drahmawahida@gmail.com

