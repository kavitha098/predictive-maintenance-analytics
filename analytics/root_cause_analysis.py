import pandas as pd


def failure_summary(df):
    """
    Generate failure summary statistics.
    """

    failure_col = None

    for col in df.columns:
        if "failure" in col.lower():
            failure_col = col
            break

    if failure_col is None:
        return {
            "Total Failures": 0,
            "Failure Rate (%)": 0
        }

    total_failures = int(df[failure_col].sum())

    failure_rate = round(
        (total_failures / len(df)) * 100,
        2
    )

    return {
        "Total Failures": total_failures,
        "Failure Rate (%)": failure_rate
    }


def equipment_failure_analysis(df):
    """
    Failure count by equipment.
    """

    equipment_col = None
    failure_col = None

    for col in df.columns:

        if "equipment" in col.lower():
            equipment_col = col

        if "failure" in col.lower():
            failure_col = col

    if equipment_col is None or failure_col is None:
        return pd.DataFrame()

    result = (
        df.groupby(equipment_col)[failure_col]
        .sum()
        .reset_index()
    )

    result.columns = [
        "Equipment_ID",
        "Failure_Count"
    ]

    result = result.sort_values(
        "Failure_Count",
        ascending=False
    )

    return result
