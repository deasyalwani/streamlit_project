import streamlit as st

st.set_page_config(layout="wide")

# =========================
# HERO HEADER
# =========================
st.markdown(
    """
    <div style="margin-bottom:24px;">
        <div style="font-size:42px; font-weight:700; line-height:1.2;">
            Customer Churn Prediction
        </div>
        <div style="font-size:16px; color:#6c757d; margin-top:4px;">
            Portfolio Project · <b>Deasy R Alwani</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    Aplikasi ini menampilkan hasil analisis dan model prediksi customer churn
    yang dikembangkan sebagai bagian dari portfolio Data Science.

    Dalam kasus ini, dilakukan analisis customer churn prediction menggunakan pendekatan supervised machine learning classification pada dataset pelanggan telekomunikasi. Customer churn merupakan kondisi di mana pelanggan memutuskan untuk berhenti menggunakan layanan perusahaan. Masalah ini menjadi krusial karena pada prakteknya, mempertahankan pelanggan lama umumnya lebih efisien dibandingkan mengakuisisi pelanggan baru, sehingga perusahaan perlu mengambil langkah yang tepat untuk mengatasinya. Tujuan utama dari analisis ini adalah untuk membangun model prediksi yang mampu mengidentifikasi pelanggan yang berpotensi melakukan churn, sehingga perusahaan dapat menyusun strategi retensi pelanggan yang lebih proaktif dan tepat sasaran.
    """
)

st.markdown("---")

# =========================
# NAVIGASI
# =========================
st.sidebar.title("Navigasi")

menu = st.sidebar.radio(
    "Pilih Halaman",
    ["Tentang Project", "Exploratory Data Analysis", "Prediksi Churn"]
)

if menu == "Tentang Project":
    import pages.tentang as tentang
    tentang.show()

elif menu == "Exploratory Data Analysis":
    import pages.eda as eda
    eda.show()

elif menu == "Prediksi Churn":
    import pages.prediksi as prediksi
    prediksi.show()
