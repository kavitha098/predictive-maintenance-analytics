import pandas as pd


def failure_summary(df):

    failure_df = df[
        df["Fault Detected"] == 1
    ]

    summary = {
        "Total Failures": len(failure_df),
        "Average Temperature":
        round(
            failure_df["Temperature (°C)"].mean(),
            2
        ),
        "Average Vibration":
        round(
            failure_df["Vibration (m/s²)"].mean(),
            2
        ),
        "Average Voltage":
        round(
            failure_df["Voltage (V)"].mean(),
            2
        )
    }

    return summary


def failure_type_analysis(df):

    if "Failure Type" not in df.columns:
        return pd.DataFrame()

    return (
        df[df["Fault Detected"] == 1]
        ["Failure Type"]
        .value_counts()
        .reset_index()
    )


def criticality_analysis(df):

    if "Equipment Criticality" not in df.columns:
        return pd.DataFrame()

    return (
        df[df["Fault Detected"] == 1]
        ["Equipment Criticality"]
        .value_counts()
        .reset_index()
    )


def root_cause_records(df):

    columns = [
        "Equipment_ID",
        "Failure Type",
        "Equipment Criticality",
        "Temperature (°C)",
        "Vibration (m/s²)",
        "Voltage (V)"
    ]

    available_cols = [
        col for col in columns
        if col in df.columns
    ]

    return df[
        df["Fault Detected"] == 1
    ][available_cols]
