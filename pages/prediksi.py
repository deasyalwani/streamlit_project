import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ======================================================
# Load model & preprocessing objects
# ======================================================
model = joblib.load("model/churn_model.joblib")
scaler = joblib.load("model/scaler.joblib")
encoder = joblib.load("model/encoder.joblib")


def show():
    st.markdown("## Prediksi Churn Pelanggan")

    st.write(
        """
        Halaman ini digunakan untuk memprediksi kemungkinan seorang pelanggan
        akan melakukan churn (berhenti berlangganan) berdasarkan karakteristik
        layanan dan perilaku pelanggan.
        """
    )

    st.markdown("---")

    # ==================================================
    # Threshold control
    # ==================================================
    st.markdown("### Pengaturan Threshold")

    threshold = st.slider(
        "Threshold probabilitas churn",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05
    )

    st.caption(
        "Threshold digunakan sebagai batas keputusan. "
        "Semakin rendah threshold, semakin sensitif model dalam mendeteksi churn."
    )

    st.markdown("---")

    # ==================================================
    # Input Form (Balanced Layout)
    # ==================================================
    st.markdown("### Input Data Pelanggan")

    col_left, col_right = st.columns(2)

    # -------- LEFT COLUMN : Billing & Contract --------
    with col_left:
        st.markdown("#### Informasi Billing & Kontrak")

        Tenure = st.number_input(
            "Tenure (bulan)",
            min_value=0,
            max_value=100,
            value=12
        )

        MonthlyCharges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0
        )

        Contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        PaymentMethod = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        PaperlessBilling = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

    # -------- RIGHT COLUMN : Demografi + Layanan --------
    with col_right:
        st.markdown("#### Informasi Pelanggan")

        Gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        Partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        Dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        SeniorCitizen = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"]
        )

        st.markdown("#### Informasi Layanan")

        InternetService = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        OnlineSecurity = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        TechSupport = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

    st.markdown("")
    predict = st.button("Prediksi Churn", use_container_width=True)

    # ==================================================
    # Prediction
    # ==================================================
    if predict:
        input_data = pd.DataFrame(
            {
                "Tenure": [Tenure],
                "MonthlyCharges": [MonthlyCharges],
                "SeniorCitizen": [1 if SeniorCitizen == "Yes" else 0],
                "Gender": [Gender],
                "Partner": [Partner],
                "Dependents": [Dependents],
                "PaperlessBilling": [PaperlessBilling],
                "Contract": [Contract],
                "PaymentMethod": [PaymentMethod],
                "InternetService": [InternetService],
                "OnlineSecurity": [OnlineSecurity],
                "TechSupport": [TechSupport],
            }
        )

        num_cols = list(scaler.feature_names_in_)
        cat_cols = list(encoder.feature_names_in_)

        X_num = scaler.transform(input_data[num_cols])
        X_cat = encoder.transform(input_data[cat_cols])

        X_final = np.hstack([X_num, X_cat])

        churn_prob = model.predict_proba(X_final)[0][1]
        churn_label = "Churn" if churn_prob >= threshold else "Tidak Churn"

        if churn_prob < 0.3:
            risk_level = "Low Risk"
        elif churn_prob < 0.6:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"

        st.markdown("---")
        st.markdown("### Hasil Prediksi")

        st.write(f"**Status Prediksi:** {churn_label}")
        st.write(f"**Probabilitas Churn:** {churn_prob:.2%}")
        st.write(f"**Risk Level:** {risk_level}")

        if risk_level == "High Risk":
            st.error(
                "Pelanggan memiliki risiko churn yang tinggi. "
                "Disarankan untuk dilakukan tindakan retensi segera."
            )
        elif risk_level == "Medium Risk":
            st.warning(
                "Pelanggan berada pada risiko churn menengah. "
                "Perlu dipertimbangkan strategi pencegahan churn."
            )
        else:
            st.success(
                "Risiko churn pelanggan rendah. "
                "Tidak diperlukan tindakan retensi khusus."
            )
