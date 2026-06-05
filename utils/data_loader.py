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
