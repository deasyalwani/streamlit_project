import streamlit as st

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)


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
    yang dikembangkan untuk membangun model prediksi yang mampu mengidentifikasi pelanggan yang berpotensi melakukan churn, sehingga perusahaan dapat menyusun strategi retensi pelanggan yang lebih proaktif dan tepat sasaran.
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
