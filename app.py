st.sidebar.file_uploader()

st.sidebar.slider(
    "Z-Score Threshold",
    min_value=2.0,
    max_value=4.0,
    value=3.0
)

st.sidebar.selectbox(
    "Select Equipment",
    df["Equipment_ID"].unique()
)
