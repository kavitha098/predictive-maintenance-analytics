"""
data_loader.py

Handles:
1. CSV loading
2. Datetime conversion
3. Missing value handling
4. Basic dataset information
"""
import pandas as pd


def load_data(file_path):
    """
    Load CSV dataset

    Parameters:
    ----------
    file_path : str or uploaded file

    Returns:
    -------
    DataFrame
    """

    try:
        df = pd.read_csv(file_path)

        return df

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def convert_timestamp(df):
    """
    Convert Timestamp column to datetime
    """

    if "Timestamp" in df.columns:

        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"],
            errors="coerce"
        )

        df = df.sort_values("Timestamp")

    return df


def set_datetime_index(df):
    """
    Set Timestamp as index
    """

    if "Timestamp" in df.columns:

        df.set_index(
            "Timestamp",
            inplace=True
        )

    return df


def handle_missing_values(df):
    """
    Fill missing values
    """

    numeric_cols = df.select_dtypes(
        include=["number"]
    ).columns

    for col in numeric_cols:

        df[col] = df[col].fillna(
            df[col].median()
        )

    categorical_cols = df.select_dtypes(
        include=["object"]
    ).columns

    for col in categorical_cols:

        df[col] = df[col].fillna(
            "Unknown"
        )

    return df


def get_dataset_info(df):
    """
    Return dataset summary
    """

    info = {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": int(df.isnull().sum().sum())
    }

    return info


def get_numeric_columns(df):
    """
    Return numeric columns
    """

    return list(
        df.select_dtypes(
            include=["number"]
        ).columns
    )


def get_categorical_columns(df):
    """
    Return categorical columns
    """

    return list(
        df.select_dtypes(
            include=["object"]
        ).columns
    )


def filter_equipment(df, equipment_id):
    """
    Filter selected equipment
    """

    if equipment_id == "All":
        return df

    return df[
        df["Equipment_ID"] == equipment_id
    ]


def load_and_preprocess(file_path):
    """
    Complete preprocessing pipeline
    """

    df = load_data(file_path)

    if df is None:
        return None

    df = convert_timestamp(df)

    df = handle_missing_values(df)

    return df


if __name__ == "__main__":

    df = load_and_preprocess(
        "data/sensor_maintenance_data.csv"
    )

    print(get_dataset_info(df))

    print(df.head())
