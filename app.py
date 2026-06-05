import streamlit as st
import pandas as pd
import numpy as np

st.title("Predictive Maintenance Analytics")

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())
