
import streamlit as st

st.set_page_config(
    page_title="Diamond Dynamics",
    page_icon="💎",
    layout="wide"
)

st.title("Diamond Dynamics")

st.subheader(
    "Diamond Price Prediction & Market Segmentation"
)

st.write(
    """
    This application predicts diamond prices in INR and
    identifies diamond market segments using machine learning.
    """
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Regression",
        "XGBoost"
    )

with col2:
    st.metric(
        "Clustering",
        "K-Means"
    )

st.info(
    "Use the sidebar to access Price Prediction and Market Segmentation."
)
