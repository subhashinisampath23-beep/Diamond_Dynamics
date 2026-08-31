
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.title("Diamond Price Prediction")

project_dir = Path(__file__).resolve().parent.parent

model_path = (
    project_dir /
    "best_diamond_price_model.pkl"
)

model = joblib.load(
    model_path
)

