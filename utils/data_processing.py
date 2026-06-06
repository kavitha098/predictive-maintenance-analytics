"""
Data processing utilities for sensor maintenance data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_and_process_data(df):
    """
    Load and preprocess the sensor data
    
    Args:
        df: Input DataFrame
        
    Returns:
        Processed DataFrame
    """
    # Make a copy to avoid warnings
    df = df.copy()
    
    # Convert timestamp to datetime
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)
    
    # Rename columns for consistency
    column_mapping = {
        'Temperature (°C)': 'Temperature_C',
        'Vibration (m/s²)': 'Vibration_mps2',
        'Voltage (V)': 'Voltage_V',
        'Current (A)': 'Current_A',
        'Power (W)': 'Power_W',
        'Humidity (%)': 'Humidity_percent',
        'Ambient Temperature (°C)': 'Ambient_Temperature_C',
        'Ambient Humidity (%)': 'Ambient_Humidity_percent'
    }
    
    df.rename(columns=column_mapping, inplace=True)
    
    # Create binary fault status
    if 'Fault Detected' in df.columns:
        df['Fault_Status'] = (df['Fault Detected'] == 'Fault Detected').astype(int)
    elif 'Fault Status' in df.columns:
        df['Fault_Status'] = (df['Fault Status'] == 'Fault Detected').astype(int)
    else:
        # Use Predictive Maintenance Trigger as proxy if available
        if 'Predictive Maintenance Trigger' in df.columns:
            df['Fault_Status'] = df['Predictive Maintenance Trigger']
    
    # Ensure numeric columns are proper type
    numeric_cols = ['Temperature_C', 'Vibration_mps2', 'Voltage_V', 'Current_A', 
                    'Power_W', 'Humidity_percent', 'Ambient_Temperature_C', 
                    'Ambient_Humidity_percent']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with critical NaN values
    critical_cols = ['Temperature_C', 'Vibration_mps2', 'Voltage_V']
    critical_cols = [col for col in critical_cols if col in df.columns]
    
    if critical_cols:
        df = df.dropna(subset=critical_cols)
    
    return df


def calculate_rolling_stats(df, window=10, columns=None):
    """
    Calculate rolling mean and standard deviation for sensor parameters
    
    Args:
        df: Input DataFrame
        window: Rolling window size
        columns: List of columns to calculate rolling stats for
        
    Returns:
        DataFrame with rolling statistics
    """
    df = df.copy()
    
    if columns is None:
        columns = ['Temperature_C', 'Vibration_mps2', 'Voltage_V', 'Current_A', 'Power_W']
    
    columns = [col for col in columns if col in df.columns]
    
    for col in columns:
        df[f'{col}_RollingMean'] = df[col].rolling(window=window, min_periods=1).mean()
        df[f'{col}_RollingStd'] = df[col].rolling(window=window, min_periods=1).std()
    
    return df


def detect_anomalies_zscore(df, threshold=3.0, columns=None):
    """
    Detect anomalies using Z-score method
    
    Args:
        df: Input DataFrame
        threshold: Z-score threshold for anomaly detection
        columns: List of columns to check for anomalies
        
    Returns:
        DataFrame with anomaly flags
    """
    df = df.copy()
    
    if columns is None:
        columns = ['Temperature_C', 'Vibration_mps2', 'Voltage_V']
    
    columns = [col for col in columns if col in df.columns]
    
    # Calculate Z-scores
    for col in columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        
        if std_val > 0:
            df[f'Z_Score_{col.split("_")[0]}'] = (df[col] - mean_val) / std_val
        else:
            df[f'Z_Score_{col.split("_")[0]}'] = 0
    
    # Create combined anomaly flag
    zscore_cols = [f'Z_Score_{col.split("_")[0]}' for col in columns]
    zscore_cols = [col for col in zscore_cols if col in df.columns]
    
    if zscore_cols:
        # Anomaly if any Z-score exceeds threshold
        df['Anomaly_Alert'] = (
            np.abs(df[zscore_cols]) > threshold
        ).any(axis=1).astype(int)
        
        # Calculate combined anomaly score
        df['Anomaly_Score'] = np.abs(df[zscore_cols]).max(axis=1)
    
    return df


def prepare_features_for_modeling(df, feature_cols):
    """
    Prepare features and target for machine learning
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature column names
        
    Returns:
        X: Feature matrix
        y: Target vector
    """
    # Ensure no NaN values
    available_features = [col for col in feature_cols if col in df.columns]
    
    X = df[available_features].copy()
    y = df['Fault_Status'].copy()
    
    # Handle NaN values
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    return X, y


def create_time_based_features(df):
    """
    Create time-based features from timestamp index
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        DataFrame with additional time features
    """
    df = df.copy()
    
    if df.index.name == 'Timestamp' or isinstance(df.index, pd.DatetimeIndex):
        df['Hour'] = df.index.hour
        df['DayOfWeek'] = df.index.dayofweek
        df['DayOfMonth'] = df.index.day
        df['WeekOfYear'] = df.index.isocalendar().week
        df['IsWeekend'] = (df.index.dayofweek >= 5).astype(int)
    
    return df


def create_lagged_features(df, columns, lags=[1, 2, 3]):
    """
    Create lagged features for time series prediction
    
    Args:
        df: Input DataFrame
        columns: List of columns to create lags for
        lags: List of lag values
        
    Returns:
        DataFrame with lagged features
    """
    df = df.copy()
    
    for col in columns:
        if col in df.columns:
            for lag in lags:
                df[f'{col}_Lag{lag}'] = df[col].shift(lag)
    
    return df
