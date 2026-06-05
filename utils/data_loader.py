import pandas as pd
import streamlit as st


def load_data(uploaded_file):
    """
    Load CSV file and convert Timestamp column to datetime.
    """

    try:
        df = pd.read_csv(uploaded_file)

        # Convert timestamp if available
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(
                df["Timestamp"],
                errors="coerce"
            )

            df = df.sort_values("Timestamp")

        return df

    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


def validate_columns(df):
    """
    Validate required columns.
    """

    required_columns = [
        "Timestamp",
        "Equipment_ID",
        "Temperature (°C)",
        "Vibration (m/s²)",
        "Voltage (V)"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        st.error(
            f"Missing Columns: {missing_columns}"
        )
        return False

    return True


def clean_data(df):
    """
    Basic preprocessing.
    """

    df = df.copy()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Forward fill missing values
    df = df.fillna(method="ffill")

    # Backward fill remaining values
    df = df.fillna(method="bfill")

    return df


def get_equipment_list(df):
    """
    Return unique equipment IDs.
    """

    if "Equipment_ID" in df.columns:
        return sorted(
            df["Equipment_ID"]
            .dropna()
            .unique()
            .tolist()
        )

    return []


def filter_equipment(df, equipment_id):
    """
    Filter dataframe by Equipment_ID.
    """

    if equipment_id == "All":
        return df

    return df[
        df["Equipment_ID"] == equipment_id
    ]


def calculate_operating_hours(df):
    """
    Calculate total operating hours.
    """

    if "Timestamp" not in df.columns:
        return 0

    if len(df) < 2:
        return 0

    start_time = df["Timestamp"].min()
    end_time = df["Timestamp"].max()

    hours = (
        end_time - start_time
    ).total_seconds() / 3600

    return round(hours, 2)


def get_current_temperature(df):
    """
    Get latest temperature value.
    """

    if "Temperature (°C)" not in df.columns:
        return 0

    return round(
        df["Temperature (°C)"].iloc[-1],
        2
    )


def get_current_voltage(df):
    """
    Get latest voltage value.
    """

    if "Voltage (V)" not in df.columns:
        return 0

    return round(
        df["Voltage (V)"].iloc[-1],
        2
    )


def get_current_vibration(df):
    """
    Get latest vibration value.
    """

    if "Vibration (m/s²)" not in df.columns:
        return 0

    return round(
        df["Vibration (m/s²)"].iloc[-1],
        2
    )


def get_failure_count(df):
    """
    Count failure records.
    """

    if "Failure_Status" in df.columns:
        return int(
            df["Failure_Status"].sum()
        )

    if "Fault Detected" in df.columns:
        return int(
            df["Fault Detected"].sum()
        )

    return 0
