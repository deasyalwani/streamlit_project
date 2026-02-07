import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def show():
    st.markdown("## Exploratory Data Analysis")

    # ==============================
    # Load data
    # ==============================
    df = pd.read_csv("data/churn.csv")

    # ==============================
    # Ringkasan awal dataset
    # ==============================
    st.markdown("### Ringkasan awal dataset pelanggan")

    st.write(
        """
        Dataset berisi informasi karakteristik pelanggan, layanan yang digunakan,
        serta status churn pelanggan. Tahap ini bertujuan untuk memahami struktur
        data dan distribusi fitur utama.
        """
    )

    st.dataframe(df.head())

    st.markdown("---")

    # ==============================
    # Distribusi target churn
    # ==============================
    st.markdown("### Distribusi Target Churn")

    churn_pct = (
        df["Churn"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .reset_index()
    )
    churn_pct.columns = ["Churn", "Persentase (%)"]

    st.bar_chart(churn_pct.set_index("Churn"))

    st.markdown("---")

    # ==============================
    # Analisis fitur numerik
    # ==============================
    st.markdown("### Analisis Fitur Numerik")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    selected_num = st.selectbox(
        "Pilih fitur numerik:",
        numeric_cols
    )

    use_log = st.checkbox("Gunakan skala log (untuk distribusi skewed)")

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.histplot(
        df[selected_num],
        bins=30,
        kde=True,
        ax=ax
    )

    ax.set_title(f"Distribusi {selected_num}")
    ax.set_xlabel(selected_num)
    ax.set_ylabel("Frekuensi")

    if use_log:
        ax.set_xscale("log")

    st.pyplot(fig)

    st.markdown("---")

    # ==============================
    # Analisis fitur kategorikal
    # ==============================
    st.markdown("### Analisis Fitur Kategorikal")

    categorical_cols = df.select_dtypes(include="object").columns

    selected_cat = st.selectbox(
        "Pilih fitur kategorikal:",
        categorical_cols
    )

    cat_dist = (
        df[selected_cat]
        .value_counts()
        .reset_index()
    )
    cat_dist.columns = [selected_cat, "Jumlah"]

    st.bar_chart(cat_dist.set_index(selected_cat))
