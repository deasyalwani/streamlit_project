import streamlit as st

def show():
    st.markdown("## Tentang Project")

    st.markdown(
        """
        Project ini bertujuan untuk memprediksi kemungkinan pelanggan
        melakukan <b>churn</b> (berhenti berlangganan) menggunakan pendekatan
        <i>machine learning classification</i>.
        """,
        unsafe_allow_html=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div style="padding:16px; border-radius:10px; background-color:#f8f9fa;">
                <div style="font-size:14px; color:#6c757d;">Jenis Masalah</div>
                <div style="font-size:22px; font-weight:600;">Classification</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div style="padding:16px; border-radius:10px; background-color:#f8f9fa;">
                <div style="font-size:14px; color:#6c757d;">Target</div>
                <div style="font-size:22px; font-weight:600;">Customer Churn</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div style="padding:16px; border-radius:10px; background-color:#f8f9fa;">
                <div style="font-size:14px; color:#6c757d;">Model</div>
                <div style="font-size:22px; font-weight:600;">Logistic Regression</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        Model difokuskan pada metrik <b>recall churn</b>, dengan tujuan
        meminimalkan pelanggan churn yang tidak terdeteksi
        (<i>false negative</i>).
        """,
        unsafe_allow_html=True
    )
