import pandas as pd


def load_data(file):
    """
    Load CSV data and convert Timestamp column to datetime.
    """

    try:
        df = pd.read_csv(file)

        if "Timestamp" not in df.columns:
            raise ValueError(
                "Dataset must contain a 'Timestamp' column."
            )

        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"],
            errors="coerce"
        )

        df = df.dropna(subset=["Timestamp"])

        df.set_index("Timestamp", inplace=True)

        return df

    except Exception as e:
        raise Exception(f"Error loading data: {e}")


def get_equipment_ids(df):
    """
    Return unique equipment IDs.
    """

    if "Equipment_ID" not in df.columns:
        return []

    return sorted(df["Equipment_ID"].dropna().unique())


def filter_equipment(df, equipment_id):
    """
    Filter dataframe by equipment ID.
    """

    if equipment_id is None:
        return df

    return df[df["Equipment_ID"] == equipment_id]


def get_failure_records(df):
    """
    Return rows where failure detected.
    """

    if "Fault Detected" not in df.columns:
        return pd.DataFrame()

    return df[df["Fault Detected"] == 1]


def dataset_summary(df):
    """
    Basic dataset statistics.
    """

    return {
        "Rows": len(df),
        "Columns": len(df.columns),
        "Missing Values": int(df.isnull().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum())
    }
