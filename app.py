"""
Predictive Maintenance Analytics Dashboard
Streamlit application for sensor data analysis and failure prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

from utils.data_processing import (
    load_and_process_data, 
    calculate_rolling_stats,
    detect_anomalies_zscore,
    prepare_features_for_modeling
)
from utils.visualization import (
    plot_sensor_drift,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_model_comparison
)
from models.train_models import (
    train_linear_regression,
    train_ridge_regression,
    train_knn_regression,
    train_decision_tree,
    train_random_forest,
    train_xgboost,
    evaluate_models
)

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .alert-critical {
        background-color: #ff6b6b;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<h1 class="main-header">🔧 Predictive Maintenance Analytics System</h1>', unsafe_allow_html=True)
    
    # ==================== SIDEBAR INPUTS ====================
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Sensor Logs (.csv)",
        type=['csv'],
        help="Upload CSV file with sensor maintenance data"
    )
    
    # Default file path
    default_file = "data/sensor_maintenance_data.csv"
    
    # Load data
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        try:
            df = pd.read_csv(default_file)
            st.sidebar.success(f"Loaded default dataset: {default_file}")
        except FileNotFoundError:
            st.error(f"Default file not found at {default_file}. Please upload a CSV file.")
            st.stop()
    
    # Process data
    df = load_and_process_data(df)
    
    # Sidebar controls
    st.sidebar.header("⚙️ Anomaly Detection Settings")
    zscore_threshold = st.sidebar.slider(
        "Z-score Threshold",
        min_value=2.0,
        max_value=4.0,
        value=3.0,
        step=0.1,
        help="Higher values are less sensitive to anomalies"
    )
    
    st.sidebar.header("🔍 Filter by Equipment")
    equipment_list = ['All'] + sorted(df['Equipment_ID'].unique().tolist())
    selected_equipment = st.sidebar.selectbox("Select Equipment ID", equipment_list)
    
    # Rolling window size
    window_size = st.sidebar.slider(
        "Rolling Window Size",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
        help="Window size for rolling statistics calculation"
    )
    
    # Filter data
    if selected_equipment != 'All':
        df_filtered = df[df['Equipment_ID'] == selected_equipment].copy()
    else:
        df_filtered = df.copy()
    
    # ==================== MAIN DASHBOARD ====================
    
    # Calculate metrics
    total_operating_hours = len(df_filtered)
    current_temp = df_filtered['Temperature_C'].iloc[-1] if not df_filtered.empty else 0
    
    # Detect anomalies
    df_filtered = calculate_rolling_stats(df_filtered, window=window_size)
    df_filtered = detect_anomalies_zscore(df_filtered, threshold=zscore_threshold)
    
    # Calculate active alerts
    active_alerts = df_filtered['Anomaly_Alert'].sum()
    
    # Metric row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total Operating Hours",
            value=f"{total_operating_hours:,}",
            delta="Records"
        )
    
    with col2:
        st.metric(
            label="🌡️ Current Temperature",
            value=f"{current_temp:.1f}°C",
            delta=f"{current_temp - df_filtered['Temperature_C'].mean():.1f}°C from avg"
        )
    
    with col3:
        st.metric(
            label="⚠️ Active Critical Alerts",
            value=f"{active_alerts}",
            delta=f"{active_alerts/total_operating_hours*100:.1f}%" if total_operating_hours > 0 else "0%",
            delta_color="inverse"
        )
    
    with col4:
        failure_rate = df_filtered['Fault_Status'].mean() * 100 if not df_filtered.empty else 0
        st.metric(
            label="🔧 Failure Rate",
            value=f"{failure_rate:.1f}%",
            delta="Critical" if failure_rate > 20 else "Normal"
        )
    
    st.divider()
    
    # ==================== SENSOR DRIFT CHART ====================
    st.subheader("📈 Sensor Drift Analysis with Failure Threshold")
    
    # Select parameter for drift chart
    param_options = ['Temperature_C', 'Vibration_mps2', 'Voltage_V', 'Power_W', 'Current_A']
    selected_param = st.selectbox("Select Parameter to Visualize", param_options, key="drift_param")
    
    # Create drift chart
    fig = plot_sensor_drift(df_filtered, selected_param, zscore_threshold)
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # ==================== ANOMALY DETECTION TABLE ====================
    st.subheader("⚠️ Anomaly Alert Records")
    
    anomaly_records = df_filtered[df_filtered['Anomaly_Alert'] == 1].copy()
    
    if not anomaly_records.empty:
        st.warning(f"⚠️ {len(anomaly_records)} anomaly records detected!")
        
        # Display table
        display_cols = ['Timestamp', 'Equipment_ID', 'Temperature_C', 'Vibration_mps2', 
                        'Voltage_V', 'Power_W', 'Fault_Status', 'Z_Score_Temp', 'Z_Score_Vibration']
        available_cols = [col for col in display_cols if col in anomaly_records.columns]
        
        st.dataframe(
            anomaly_records[available_cols].head(100),
            use_container_width=True,
            height=300
        )
    else:
        st.info("✅ No anomalies detected with current threshold settings.")
    
    st.divider()
    
    # ==================== MACHINE LEARNING MODELS ====================
    st.subheader("🤖 Predictive Models for Failure Prediction")
    
    # Prepare features for modeling
    feature_cols = ['Temperature_C', 'Vibration_mps2', 'Voltage_V', 'Power_W', 
                    'Current_A', 'Humidity_percent', 'Ambient_Temperature_C', 
                    'Ambient_Humidity_percent']
    
    df_ml = df_filtered.dropna(subset=feature_cols + ['Fault_Status'])
    
    if len(df_ml) >= 100:
        X, y = prepare_features_for_modeling(df_ml, feature_cols)
        
        # Model selection
        st.markdown("### Model Training & Evaluation")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_models = st.multiselect(
                "Select Models to Train",
                options=['Linear Regression', 'Ridge Regression', 'KNN Regression', 
                        'Decision Tree', 'Random Forest', 'XGBoost'],
                default=['Linear Regression', 'KNN Regression', 'Random Forest']
            )
            
            test_size = st.slider("Test Set Size", 0.2, 0.4, 0.25, 0.05)
            
            train_model_btn = st.button("🚀 Train Selected Models", type="primary")
        
        with col2:
            st.info("""
            **Available Models:**
            - **Linear/Ridge Regression**: Baseline models for linear relationships
            - **KNN Regression**: Captures local patterns in sensor data
            - **Decision Tree/Random Forest**: Handles non-linear relationships
            - **XGBoost**: Advanced gradient boosting for high accuracy
            """)
        
        if train_model_btn:
            with st.spinner("Training models... This may take a moment."):
                results = {}
                trained_models = {}
                
                # Train selected models
                if 'Linear Regression' in selected_models:
                    results['Linear Regression'], trained_models['Linear Regression'] = train_linear_regression(X, y, test_size)
                
                if 'Ridge Regression' in selected_models:
                    results['Ridge Regression'], trained_models['Ridge Regression'] = train_ridge_regression(X, y, test_size)
                
                if 'KNN Regression' in selected_models:
                    results['KNN Regression'], trained_models['KNN Regression'] = train_knn_regression(X, y, test_size)
                
                if 'Decision Tree' in selected_models:
                    results['Decision Tree'], trained_models['Decision Tree'] = train_decision_tree(X, y, test_size)
                
                if 'Random Forest' in selected_models:
                    results['Random Forest'], trained_models['Random Forest'] = train_random_forest(X, y, test_size)
                
                if 'XGBoost' in selected_models:
                    results['XGBoost'], trained_models['XGBoost'] = train_xgboost(X, y, test_size)
                
                # Display results
                st.markdown("### Model Performance Comparison")
                
                # Create results dataframe
                results_df = pd.DataFrame(results).T
                results_df = results_df.round(4)
                
                # Add ranking
                results_df['R2_Rank'] = results_df['R2_Score'].rank(ascending=False).astype(int)
                
                st.dataframe(
                    results_df[['MSE', 'RMSE', 'R2_Score', 'R2_Rank']].sort_values('R2_Score', ascending=False),
                    use_container_width=True
                )
                
                # Plot comparison
                fig_comparison = plot_model_comparison(results_df)
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Feature importance for tree-based models
                st.markdown("### Feature Importance Analysis")
                
                if 'Random Forest' in trained_models:
                    fig_importance = plot_feature_importance(
                        trained_models['Random Forest'], 
                        feature_cols,
                        "Random Forest"
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                
                # Store best model in session state
                best_model_name = results_df['R2_Score'].idxmax()
                st.success(f"🏆 Best Model: **{best_model_name}** with R² Score = {results_df.loc[best_model_name, 'R2_Score']:.4f}")
                
    else:
        st.warning(f"⚠️ Insufficient data for ML modeling. Need at least 100 records, found {len(df_ml)}.")
    
    st.divider()
    
    # ==================== CORRELATION ANALYSIS ====================
    st.subheader("📊 Feature Correlation Analysis")
    
    corr_cols = ['Temperature_C', 'Vibration_mps2', 'Voltage_V', 'Power_W', 
                 'Current_A', 'Humidity_percent', 'Ambient_Temperature_C', 
                 'Ambient_Humidity_percent', 'Fault_Status']
    
    available_corr_cols = [col for col in corr_cols if col in df_filtered.columns]
    
    if len(available_corr_cols) >= 2:
        fig_corr = plot_correlation_heatmap(df_filtered, available_corr_cols)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Insights
        st.markdown("**Key Insights:**")
        corr_matrix = df_filtered[available_corr_cols].corr()
        fault_corr = corr_matrix['Fault_Status'].drop('Fault_Status').sort_values(key=abs, ascending=False)
        
        for feature, corr_val in fault_corr.head(3).items():
            direction = "positive" if corr_val > 0 else "negative"
            st.write(f"- {feature}: {direction} correlation of {abs(corr_val):.3f} with failure status")
    
    st.divider()
    
    # ==================== EQUIPMENT SUMMARY ====================
    st.subheader("📋 Equipment Summary")
    
    equipment_summary = df_filtered.groupby('Equipment_ID').agg({
        'Fault_Status': ['mean', 'count'],
        'Temperature_C': 'mean',
        'Vibration_mps2': 'mean',
        'Power_W': 'mean'
    }).round(3)
    
    equipment_summary.columns = ['Failure_Rate', 'Record_Count', 'Avg_Temp_C', 'Avg_Vibration', 'Avg_Power_W']
    equipment_summary = equipment_summary.sort_values('Failure_Rate', ascending=False)
    
    st.dataframe(equipment_summary, use_container_width=True)
    
    # ==================== FOOTER ====================
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>🔧 Predictive Maintenance Dashboard | Built with Streamlit</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
