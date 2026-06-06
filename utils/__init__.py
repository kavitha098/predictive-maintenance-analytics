"""
Utilities package for data processing and visualization
"""

from .data_processing import (
    load_and_process_data,
    calculate_rolling_stats,
    detect_anomalies_zscore,
    prepare_features_for_modeling,
    create_time_based_features,
    create_lagged_features
)

from .anomaly_detection import (
    compute_zscore_matrix,
    apply_anomaly_flag,
    detect_multivariate_anomalies,
    detect_iqr_outliers,
    detect_ewma_anomalies
)

from .visualization import (
    plot_sensor_drift,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_model_comparison,
    plot_failure_distribution,
    plot_equipment_failure_rates,
    plot_time_series_decomposition
)

__all__ = [
    'load_and_process_data',
    'calculate_rolling_stats',
    'detect_anomalies_zscore',
    'prepare_features_for_modeling',
    'create_time_based_features',
    'create_lagged_features',
    'compute_zscore_matrix',
    'apply_anomaly_flag',
    'detect_multivariate_anomalies',
    'detect_iqr_outliers',
    'detect_ewma_anomalies',
    'plot_sensor_drift',
    'plot_correlation_heatmap',
    'plot_feature_importance',
    'plot_model_comparison',
    'plot_failure_distribution',
    'plot_equipment_failure_rates',
    'plot_time_series_decomposition'
]
