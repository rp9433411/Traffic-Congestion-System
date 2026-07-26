"""
Configuration settings for Traffic Congestion Prediction System.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'output'
NOTEBOOKS_DIR = BASE_DIR / 'notebooks'
SRC_DIR = BASE_DIR / 'src'
API_DIR = BASE_DIR / 'api'
STATIC_DIR = BASE_DIR / 'static'
TEMPLATES_DIR = API_DIR / 'templates'

# Ensure directories exist (skip on Vercel read-only filesystem)
_vercel = os.environ.get('VERCEL', '') == '1'
if not _vercel:
    for dir_path in [DATA_DIR, MODELS_DIR, OUTPUT_DIR, NOTEBOOKS_DIR,
                     SRC_DIR, API_DIR, STATIC_DIR, TEMPLATES_DIR,
                     STATIC_DIR / 'css', STATIC_DIR / 'js']:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError):
            pass

# Data Generation Settings
DATA_GENERATION = {
    'n_samples': 10000,
    'start_date': '2024-01-01',
    'end_date': '2024-12-31',
    'locations': [
        'Downtown', 'Highway_A', 'Highway_B', 'Residential_A',
        'Commercial_A', 'Airport_Road', 'Ring_Road', 'Bridge_1',
        'Bridge_2', 'Market_Area', 'Industrial_Zone', 'University_Area'
    ],
    'random_seed': 42
}

# Model Settings
MODEL_CONFIG = {
    'test_size': 0.2,
    'validation_size': 0.1,
    'random_state': 42,
    'target_column': 'congestion_level',
    'feature_columns': [
        'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday',
        'temperature', 'humidity', 'precipitation', 'wind_speed',
        'traffic_volume', 'avg_speed', 'weather_condition_encoded',
        'location_encoded', 'is_rush_hour', 'day_sin', 'day_cos',
        'month_sin', 'month_cos', 'hour_sin', 'hour_cos',
        'traffic_lag_1h', 'traffic_lag_2h', 'traffic_lag_3h',
        'traffic_rolling_mean_3h', 'speed_rolling_mean_3h'
    ]
}

# Random Forest Parameters
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'bootstrap': True,
    'random_state': 42,
    'n_jobs': -1
}

# XGBoost Parameters
XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1
}

# LSTM Parameters
LSTM_PARAMS = {
    'sequence_length': 24,  # 24 hours
    'n_features': None,  # Will be set dynamically
    'lstm_units': [64, 32],
    'dropout_rate': 0.2,
    'dense_units': [16, 8],
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50,
    'early_stopping_patience': 10
}

# Linear Regression (default params)

# API Settings
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True,
    'secret_key': 'traffic-prediction-secret-key-change-in-production'
}

# Congestion Levels Mapping
CONGESTION_LEVELS = {
    0: 'Low',
    1: 'Moderate',
    2: 'High',
    3: 'Severe'
}

# Thresholds for congestion classification
CONGESTION_THRESHOLDS = {
    'low': 0.25,
    'moderate': 0.50,
    'high': 0.75,
    'severe': 1.0
}

# Weather conditions mapping
WEATHER_CONDITIONS = {
    0: 'Clear',
    1: 'Cloudy',
    2: 'Rainy',
    3: 'Stormy',
    4: 'Foggy',
    5: 'Snowy'
}

</｜｜DSML｜｜parameter>
</create_file>
