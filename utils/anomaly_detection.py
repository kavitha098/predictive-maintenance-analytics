"""
Anomaly detection utilities for sensor data
"""

import numpy as np
import pandas as pd
from scipy import stats


def compute_zscore_matrix(df, columns):
    """
    Compute real-time Z-scores across continuous feature matrix
    
    Args:
        df: Input DataFrame
        columns: List of columns to compute Z-scores for
        
    Returns:
        DataFrame with Z-scores
    """
    df_zscore = df.copy()
    
    for col in columns:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            
            if std > 0:
                df_zscore[f'{col}_Zscore'] = (df[col] - mean) / std
            else:
                df_zscore[f'{col}_Zscore'] = 0
    
    return df_zscore


def apply_anomaly_flag(df, threshold=3.0, zscore_columns=None):
    """
    Apply np.where() to assign binary anomaly flag when Z-score exceeds threshold
    
    Args:
        df: Input DataFrame with Z-score columns
        threshold: Z-score threshold
        zscore_columns: List of Z-score column names
        
    Returns:
        DataFrame with anomaly flag
    """
    df = df.copy()
    
    if zscore_columns is None:
        zscore_columns = [col for col in df.columns if 'Zscore' in col]
    
    if not zscore_columns:
        return df
    
    # Calculate absolute max Z-score across columns
    zscore_matrix = np.abs(df[zscore_columns].values)
    max_zscore = np.max(zscore_matrix, axis=1)
    
    # Apply threshold using np.where
    df['Anomaly_Flag'] = np.where(max_zscore > threshold, 1, 0)
    df['Max_Zscore'] = max_zscore
    
    return df


def detect_multivariate_anomalies(df, columns, contamination=0.05):
    """
    Detect multivariate anomalies using Mahalanobis distance
    
    Args:
        df: Input DataFrame
        columns: Feature columns
        contamination: Expected proportion of outliers
        
    Returns:
        DataFrame with multivariate anomaly flag
    """
    from scipy.spatial.distance import mahalanobis
    
    df = df.copy()
    data = df[columns].dropna()
    
    if len(data) < len(columns) + 1:
        df['Multivariate_Anomaly'] = 0
        return df
    
    mean = np.mean(data.values, axis=0)
    cov = np.cov(data.values.T)
    
    try:
        inv_cov = np.linalg.pinv(cov)
        
        distances = []
        for i in range(len(data)):
            dist = mahalanobis(data.iloc[i].values, mean, inv_cov)
            distances.append(dist)
        
        # Use chi-square threshold
        from scipy.stats import chi2
        threshold = np.sqrt(chi2.ppf(1 - contamination, len(columns)))
        
        # Map back to original indices
        anomaly_map = pd.Series(0, index=df.index)
        anomaly_map[data.index] = (np.array(distances) > threshold).astype(int)
        
        df['Multivariate_Anomaly'] = anomaly_map.values
        
    except:
        df['Multivariate_Anomaly'] = 0
    
    return df


def detect_iqr_outliers(df, columns, multiplier=1.5):
    """
    Detect outliers using IQR method
    
    Args:
        df: Input DataFrame
        columns: List of columns
        multiplier: IQR multiplier (default 1.5 for moderate outliers)
        
    Returns:
        DataFrame with IQR outlier flags
    """
    df = df.copy()
    
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            df[f'{col}_IQR_Outlier'] = ((df[col] < lower_bound) | (df[col] > upper_bound)).astype(int)
    
    # Combined outlier flag
    outlier_cols = [col for col in df.columns if 'IQR_Outlier' in col]
    if outlier_cols:
        df['Any_IQR_Outlier'] = df[outlier_cols].any(axis=1).astype(int)
    
    return df


def detect_ewma_anomalies(df, columns, span=10, threshold=3):
    """
    Detect anomalies using EWMA (Exponentially Weighted Moving Average)
    
    Args:
        df: Input DataFrame
        columns: List of columns
        span: EWMA span
        threshold: Control limit multiplier
        
    Returns:
        DataFrame with EWMA anomaly flags
    """
    df = df.copy()
    
    for col in columns:
        if col in df.columns:
            # Calculate EWMA
            ewma = df[col].ewm(span=span, adjust=False).mean()
            
            # Calculate moving standard deviation
            rolling_std = df[col].rolling(window=span, min_periods=1).std()
            
            # Calculate control limits
            upper_limit = ewma + threshold * rolling_std
            lower_limit = ewma - threshold * rolling_std
            
            # Detect anomalies
            df[f'{col}_EWMA_Anomaly'] = ((df[col] > upper_limit) | (df[col] < lower_limit)).astype(int)
    
    return df
