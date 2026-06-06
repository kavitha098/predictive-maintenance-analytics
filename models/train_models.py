"""
Machine learning models for failure prediction
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

import warnings
warnings.filterwarnings('ignore')


def train_linear_regression(X, y, test_size=0.25, random_state=42):
    """
    Train Linear Regression model
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion for test set
        random_state: Random seed
        
    Returns:
        Dictionary of metrics and trained model
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    
    results = {
        'MSE': mse,
        'RMSE': rmse,
        'R2_Score': r2,
        'CV_Mean': cv_scores.mean(),
        'CV_Std': cv_scores.std(),
        'MAE': mean_absolute_error(y_test, y_pred)
    }
    
    return results, model


def train_ridge_regression(X, y, test_size=0.25, alpha=1.0, random_state=42):
    """
    Train Ridge Regression model
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = Ridge(alpha=alpha, random_state=random_state)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    results = {
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2_Score': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred)
    }
    
    return results, model


def train_knn_regression(X, y, test_size=0.25, n_neighbors=5, random_state=42):
    """
    Train K-Nearest Neighbors Regression model
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    results = {
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2_Score': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred)
    }
    
    return results, model


def train_decision_tree(X, y, test_size=0.25, max_depth=10, random_state=42):
    """
    Train Decision Tree Regression model
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    results = {
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2_Score': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred)
    }
    
    return results, model


def train_random_forest(X, y, test_size=0.25, n_estimators=100, max_depth=10, random_state=42):
    """
    Train Random Forest Regression model
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    results = {
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2_Score': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred)
    }
    
    return results, model


def train_xgboost(X, y, test_size=0.25, random_state=42):
    """
    Train XGBoost Regression model
    """
    try:
        import xgboost as xgb
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        results = {
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2_Score': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred)
        }
        
        return results, model
        
    except ImportError:
        print("XGBoost not installed. Skipping...")
        return {'MSE': np.nan, 'RMSE': np.nan, 'R2_Score': np.nan, 'MAE': np.nan}, None


def train_gradient_boosting(X, y, test_size=0.25, random_state=42):
    """
    Train Gradient Boosting Regression model
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    results = {
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2_Score': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred)
    }
    
    return results, model


def train_svr(X, y, test_size=0.25, random_state=42):
    """
    Train Support Vector Regression model
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    results = {
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2_Score': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred)
    }
    
    return results, model


def evaluate_models(models_dict, X, y, test_size=0.25):
    """
    Evaluate multiple models and return comparison dataframe
    
    Args:
        models_dict: Dictionary of model name to model instance
        X: Feature matrix
        y: Target vector
        test_size: Test set proportion
        
    Returns:
        DataFrame with model comparisons
    """
    results = {}
    
    for name, model in models_dict.items():
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Scale for some models
            if name in ['Linear Regression', 'Ridge Regression', 'SVR', 'KNN Regression']:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            results[name] = {
                'MSE': mean_squared_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'R2_Score': r2_score(y_test, y_pred),
                'MAE': mean_absolute_error(y_test, y_pred)
            }
        except Exception as e:
            print(f"Error training {name}: {e}")
            results[name] = {'MSE': np.nan, 'RMSE': np.nan, 'R2_Score': np.nan, 'MAE': np.nan}
    
    return pd.DataFrame(results).T
