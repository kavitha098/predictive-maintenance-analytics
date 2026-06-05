# utils/data_loader.py

import pandas as pd


def load_data(file):
    """
    Load CSV file and return DataFrame.
    """
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {e}")


def convert_timestamp(df):
    """
    Convert Timestamp column to datetime format.
    """
    try:
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(
                df["Timestamp"],
                errors="coerce"
            )
        return df

    except Exception as e:
        raise Exception(f"Error converting timestamp: {e}")


def get_equipment_list(df):
    """
    Return list of equipment IDs.
    """
    try:
        if "Equipment ID" in df.columns:
            return sorted(df["Equipment ID"].dropna().unique().tolist())

        elif "Machine_ID" in df.columns:
            return sorted(df["Machine_ID"].dropna().unique().tolist())

        return []

    except Exception as e:
        raise Exception(f"Error getting equipment list: {e}")


def filter_equipment(df, equipment):
    """
    Filter dataframe by selected equipment.
    """
    try:

        if equipment == "All":
            return df

        if "Equipment ID" in df.columns:
            return df[df["Equipment ID"] == equipment]

        elif "Machine_ID" in df.columns:
            return df[df["Machine_ID"] == equipment]

        return df

    except Exception as e:
        raise Exception(f"Error filtering equipment: {e}")


def get_dataset_info(df):
    """
    Return dataset statistics.
    """
    try:
        return {
            "Rows": int(df.shape[0]),
            "Columns": int(df.shape[1]),
            "Missing Values": int(df.isnull().sum().sum())
        }

    except Exception as e:
        raise Exception(f"Error getting dataset info: {e}")
