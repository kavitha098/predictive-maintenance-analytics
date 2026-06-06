
import pandas as pd


def equipment_failure_analysis(df):

    failure_df = df[df["Fault Detected"] == 1]

    result = (
        failure_df.groupby("Equipment_ID")
        .size()
        .reset_index(name="Failure_Count")
        .sort_values("Failure_Count", ascending=False)
    )

    return result


def failure_type_analysis(df):

    return (
        df.groupby("Failure Type")
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )


def critical_equipment_analysis(df):

    return (
        df.groupby("Equipment Criticality")
        .size()
        .reset_index(name="Count")
    )
