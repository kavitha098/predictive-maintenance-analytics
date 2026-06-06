import pandas as pd

def root_cause_analysis(df):
    """
    Identify top causes of failures.
    """

    if df is None or len(df) == 0:
        return pd.DataFrame()

    failure_cols = [
        col for col in df.columns
        if "failure" in col.lower()
        or "fault" in col.lower()
        or "breakdown" in col.lower()
    ]

    if not failure_cols:
        return pd.DataFrame(
            {
                "Metric": ["No Failure Columns Found"],
                "Value": [0]
            }
        )

    results = []

    for col in failure_cols:
        results.append(
            {
                "Metric": col,
                "Value": df[col].sum()
                if pd.api.types.is_numeric_dtype(df[col])
                else df[col].count()
            }
        )

    return pd.DataFrame(results)


def failure_summary(df):

    if df is None:
        return {}

    return {
        "rows": len(df),
        "columns": len(df.columns)
    }
