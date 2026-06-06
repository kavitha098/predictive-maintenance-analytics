import pandas as pd


def load_data(file):

    df = pd.read_csv(file)

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    df.set_index("Timestamp", inplace=True)

    return df


def get_equipment_ids(df):

    return sorted(df["Equipment_ID"].unique())


def filter_equipment(df, equipment_id):

    return df[df["Equipment_ID"] == equipment_id]
