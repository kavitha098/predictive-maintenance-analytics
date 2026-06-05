# utils/data_loader.py

import pandas as pd
import streamlit as st


@st.cache_data
def load_data(uploaded_file=None):

    try:

        if uploaded_file is not None:

            df = pd.read_csv(
                uploaded_file
            )

        else:

            df = pd.read_csv(
                "data/sensor_maintenance_data.csv"
            )

        return df

    except Exception as e:

        st.error(
            f"Error loading file: {e}"
        )

        return pd.DataFrame()


def get_equipment_ids(df):

    if "Equipment_ID" not in df.columns:
        return ["All"]

    equipment_ids = (
        df["Equipment_ID"]
        .dropna()
        .unique()
        .tolist()
    )

    equipment_ids.sort()

    return ["All"] + equipment_ids


def filter_equipment(
    df,
    equipment_id
):

    if equipment_id == "All":
        return df

    return df[
        df["Equipment_ID"]
        ==
        equipment_id
    ]


def get_failure_rows(df):

    possible_targets = [
        "Failure_Status",
        "Fault Detected",
        "Predictive Maintenance Trigger"
    ]

    for col in possible_targets:

        if col in df.columns:

            return df[
                df[col] == 1
            ]

    return pd.DataFrame()


def save_processed_data(
    df,
    filepath="data/processed_data.csv"
):

    df.to_csv(
        filepath,
        index=False
    )

    return filepath
