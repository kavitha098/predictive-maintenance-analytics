import streamlit as st
import pandas as pd

st.sidebar.title("Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Maintenance Dataset",
    type=["csv"]
)

st.sidebar.slider(
    "Z-Score Threshold",
    min_value=2.0,
    max_value=4.0,
    value=3.0
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    equipment = st.sidebar.selectbox(
        "Select Equipment",
        sorted(df["Equipment_ID"].unique())
    )

    st.write("Selected Equipment:", equipment)
    st.dataframe(df.head())

else:
    st.info("Please upload a CSV file.")
