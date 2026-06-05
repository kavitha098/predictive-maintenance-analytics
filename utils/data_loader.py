import pandas as pd


def load_data(file):
    return pd.read_csv(file)


def convert_timestamp(df):
    timestamp_cols = [
        col for col in df.columns
        if "time" in col.lower()
        or "date" in col.lower()
    ]

    if timestamp_cols:
        try:
            df[timestamp_cols[0]] = pd.to_datetime(
                df[timestamp_cols[0]]
            )
        except:
            pass

    return df


def get_equipment_list(df):
    if "Equipment ID" in df.columns:
        return sorted(
            df["Equipment ID"].dropna().unique()
        )

    return []


def filter_equipment(df, equipment):

    if (
        equipment == "All"
        or "Equipment ID" not in df.columns
    ):
        return df

    return df[
        df["Equipment ID"] == equipment
    ]


def get_dataset_info(df):

    return {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": df.isnull().sum().sum()
    }
