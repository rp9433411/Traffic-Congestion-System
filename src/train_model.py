"""
Model Training Pipeline for Traffic Congestion Prediction.

Trains multiple models:
1. Random Forest Regressor
2. XGBoost Regressor
3. LSTM Neural Network
4. Linear Regression (baseline)
5. Ensemble (weighted average of all models)

Saves trained models, scalers, and performance metrics.
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import warnings
import joblib
from datetime import datetime
from typing import Dict, Tuple, Optional, Any

# Machine Learning
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, 
                             r2_score, accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.preprocessing import StandardScaler

# XGBoost
import xgboost as xgb

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (MODEL_CONFIG, RF_PARAMS, XGB_PARAMS, LSTM_PARAMS,
                    DATA_DIR, MODELS_DIR, OUTPUT_DIR, CONGESTION_LEVELS)
from src.data_generator import TrafficDataGenerator
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings


class ModelTrainer:
    """
    Orchestrates training of multiple models for traffic congestion prediction.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize the model trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or MODEL_CONFIG
        self.models = {}
        self.metrics = {}
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.data_generator = TrafficDataGenerator()
        
        # Ensure directories exist
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
    def prepare_data(self, n_samples: int = None) -> Tuple:
        """
        Generate and prepare data for training.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        print("=" * 60)
        print("STEP 1: Generating and Preparing Data")
        print("=" * 60)
        
        # Generate data
        if n_samples is None:
            n_samples = 10000
        data = self.data_generator.generate_dataset(n_samples)
        self.data_generator.save_dataset(data)
        
        # Preprocess data
        data = self.preprocessor.preprocess_pipeline(data, fit=True)
        
        # Feature engineering
        data = self.feature_engineer.engineer_features(data, fit=True)
        
        # Prepare features and target
        X, y = self.preprocessor.prepare_features_target(data)
        
        # Split into train/val/test
        X_train, X_val, X_test, y_train, y_val, y_test = \
            self.preprocessor.train_test_val_split(X, y)
        
        # Scale features
        X_train_scaled = self.preprocessor.scaler.fit_transform(X_train)
        X_val_scaled = self.preprocessor.scaler.transform(X_val)
        X_test_scaled = self.preprocessor.scaler.transform(X_test)
        
        # Save preprocessor
        self.preprocessor.save_preprocessor()
        
        # Save feature names
        self.feature_names = X_train.columns.tolist()
        
        print(f"\nData preparation complete!")
        print(f"  Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"  Validation: {X_val.shape[0]} samples")
        print(f"  Test: {X_test.shape[0]} samples")
        
        return (X_train_scaled, X_val_scaled, X_test_scaled, 
                X_train, X_val, X_test, y_train, y_val, y_test)
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> RandomForestRegressor:
        """
        Train Random Forest model.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            Trained RandomForestRegressor
        """
        print("\n" + "=" * 60)
        print("STEP 2: Training Random Forest Model")
        print("=" * 60)
        
        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(X_train, y_train)
        
        # Validate
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        print(f"Random Forest - Validation MAE: {mae:.4f}, R²: {r2:.4f}")
        
        self.models['random_forest'] = model
        return model
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray) -> xgb.XGBRegressor:
        """
        Train XGBoost model.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            Trained XGBRegressor
        """
        print("\n" + "=" * 60)
        print("STEP 3: Training XGBoost Model")
        print("=" * 60)
        
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Validate
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        print(f"XGBoost - Validation MAE: {mae:.4f}, R²: {r2:.4f}")
        
        self.models['xgboost'] = model
        return model
    
    def train_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray) -> Sequential:
        """
        Train LSTM Neural Network model.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            Trained LSTM model
        """
        print("\n" + "=" * 60)
        print("STEP 4: Training LSTM Neural Network")
        print("=" * 60)
        
        # Reshape data for LSTM [samples, timesteps, features]
        n_features = X_train.shape[1]
        seq_length = LSTM_PARAMS['sequence_length']
        
        # For LSTM, we need 3D input: (batch, timesteps, features)
        # Using a simple approach: treat each sample as sequence of 1
        X_train_lstm = X_train.reshape((X_train.shape[0], 1, n_features))
        X_val_lstm = X_val.reshape((X_val.shape[0], 1, n_features))
        
        # Build LSTM model
        model = Sequential([
            Input(shape=(1, n_features)),
            LSTM(LSTM_PARAMS['lstm_units'][0], return_sequences=True),
            Dropout(LSTM_PARAMS['dropout_rate']),
            LSTM(LSTM_PARAMS['lstm_units'][1], return_sequences=False),
            Dropout(LSTM_PARAMS['dropout_rate']),
            Dense(LSTM_PARAMS['dense_units'][0], activation='relu'),
            Dropout(LSTM_PARAMS['dropout_rate'] / 2),
            Dense(LSTM_PARAMS['dense_units'][1], activation='relu'),
            Dense(1, activation='linear')
        ])
        
        # Compile model
        optimizer = Adam(learning_rate=LSTM_PARAMS['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=LSTM_PARAMS['early_stopping_patience'], 
                         restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        ]
        
        # Train model
        history = model.fit(
            X_train_lstm, y_train,
            validation_data=(X_val_lstm, y_val),
            epochs=LSTM_PARAMS['epochs'],
            batch_size=LSTM_PARAMS['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Validate
        y_pred = model.predict(X_val_lstm, verbose=0).flatten()
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        print(f"LSTM - Validation MAE: {mae:.4f}, R²: {r2:.4f}")
        
        self.models['lstm'] = model
        return model
    
    def train_linear_regression(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray) -> LinearRegression:
        """
        Train Linear Regression baseline model.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            Trained LinearRegression
        """
        print("\n" + "=" * 60)
        print("STEP 5: Training Linear Regression (Baseline)")
        print("=" * 60)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Validate
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        print(f"Linear Regression - Validation MAE: {mae:.4f}, R²: {r2:.4f}")
        
        self.models['linear_regression'] = model
        return model
    
    def evaluate_models(self, X_test: np.ndarray, y_test: pd.Series) -> Dict:
        """
        Evaluate all trained models on test data.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("\n" + "=" * 60)
        print("STEP 6: Evaluating All Models on Test Data")
        print("=" * 60)
        
        metrics = {}
        
        for name, model in self.models.items():
            y_pred = self._predict_model(model, X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Classification metrics (rounded predictions)
            y_pred_class = np.round(y_pred).clip(0, 3).astype(int)
            y_test_class = y_test.values.astype(int)
            accuracy = accuracy_score(y_test_class, y_pred_class)
            
            metrics[name] = {
                'MAE': round(mae, 4),
                'MSE': round(mse, 4),
                'RMSE': round(rmse, 4),
                'R2': round(r2, 4),
                'Accuracy': round(accuracy, 4)
            }
            
            print(f"\n{name.upper()}:")
            print(f"  MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f} | Accuracy: {accuracy:.2%}")
            
        return metrics
    
    def _predict_model(self, model: Any, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using a model (handles different model types).
        
        Args:
            model: Trained model
            X: Features
            
        Returns:
            Predictions array
        """
        if isinstance(model, Sequential):
            # LSTM model needs 3D input
            X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
            return model.predict(X_reshaped, verbose=0).flatten()
        else:
            return model.predict(X)
    
    def save_models(self):
        """
        Save all trained models to disk.
        """
        print("\n" + "=" * 60)
        print("STEP 7: Saving Models")
        print("=" * 60)
        
        for name, model in self.models.items():
            filepath = str(MODELS_DIR / f'congestion_{name}.pkl')
            
            if isinstance(model, Sequential):
                # Save Keras model separately
                h5_path = str(MODELS_DIR / f'congestion_{name}.h5')
                model.save(h5_path)
                print(f"  {name} saved to: {h5_path}")
            else:
                joblib.dump(model, filepath)
                print(f"  {name} saved to: {filepath}")
    
    def create_ensemble_predictions(self, X_test: np.ndarray, 
                                   y_test: pd.Series) -> Dict:
        """
        Create ensemble predictions using weighted average.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Metrics for ensemble
        """
        print("\n" + "=" * 60)
        print("STEP 8: Creating Ensemble Predictions")
        print("=" * 60)
        
        # Collect predictions from all models
        predictions = []
        model_names = []
        
        for name, model in self.models.items():
            if name != 'linear_regression':  # Exclude baseline
                pred = self._predict_model(model, X_test)
                predictions.append(pred)
                model_names.append(name)
        
        # Weighted average (higher weight for better models)
        weights = {'random_forest': 0.30, 'xgboost': 0.35, 'lstm': 0.25, 'linear_regression': 0.10}
        available_weights = {k: v for k, v in weights.items() if k in model_names}
        
        # Normalize weights
        total_weight = sum(available_weights.values())
        normalized_weights = {k: v/total_weight for k, v in available_weights.items()}
        
        # Calculate weighted ensemble
        ensemble_pred = np.zeros_like(predictions[0])
        for i, name in enumerate(model_names):
            ensemble_pred += predictions[i] * normalized_weights.get(name, 1/len(model_names))
        
        # Evaluate ensemble
        mae = mean_absolute_error(y_test, ensemble_pred)
        mse = mean_squared_error(y_test, ensemble_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, ensemble_pred)
        
        y_pred_class = np.round(ensemble_pred).clip(0, 3).astype(int)
        y_test_class = y_test.values.astype(int)
        accuracy = accuracy_score(y_test_class, y_pred_class)
        
        ensemble_metrics = {
            'MAE': round(mae, 4),
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4),
            'R2': round(r2, 4),
            'Accuracy': round(accuracy, 4),
            'weights': normalized_weights
        }
        
        print(f"\nENSEMBLE (Weighted Average):")
        print(f"  Weights: {normalized_weights}")
        print(f"  MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f} | Accuracy: {accuracy:.2%}")
        
        self.models['ensemble'] = ensemble_pred
        return ensemble_metrics
    
    def save_metrics(self, metrics: Dict, ensemble_metrics: Dict = None):
        """
        Save evaluation metrics to JSON file.
        
        Args:
            metrics: Model metrics dictionary
            ensemble_metrics: Ensemble metrics (optional)
        """
        if ensemble_metrics:
            metrics['ensemble'] = ensemble_metrics
        
        self.metrics = metrics
        
        # Save to JSON
        filepath = str(MODELS_DIR / 'metrics.json')
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"\nMetrics saved to: {filepath}")
        
        # Also save as a readable report
        report_path = str(MODELS_DIR / 'model_report.txt')
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("TRAFFIC CONGESTION PREDICTION - MODEL REPORT\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            for model_name, model_metrics in metrics.items():
                f.write(f"\n{model_name.upper()}:\n")
                f.write("-" * 40 + "\n")
                for metric, value in model_metrics.items():
                    if isinstance(value, dict):
                        f.write(f"  {metric}: {value}\n")
                    else:
                        f.write(f"  {metric}: {value}\n")
        
        print(f"Report saved to: {report_path}")
    
    def train_all_models(self, n_samples: int = 10000) -> Dict:
        """
        Complete training pipeline for all models.
        
        Args:
            n_samples: Number of samples for training
            
        Returns:
            Performance metrics dictionary
        """
        print("\n" + "=" * 60)
        print("TRAFFIC CONGESTION PREDICTION - MODEL TRAINING")
        print("=" * 60 + "\n")
        
        # Prepare data
        (X_train_scaled, X_val_scaled, X_test_scaled,
         X_train_df, X_val_df, X_test_df, 
         y_train, y_val, y_test) = self.prepare_data(n_samples)
        
        # Train all models
        self.train_random_forest(X_train_scaled, y_train, X_val_scaled, y_val)
        self.train_xgboost(X_train_scaled, y_train, X_val_scaled, y_val)
        self.train_linear_regression(X_train_scaled, y_train, X_val_scaled, y_val)
        
        # LSTM requires scaling differently
        self.train_lstm(X_train_scaled, y_train, X_val_scaled, y_val)
        
        # Evaluate all models
        metrics = self.evaluate_models(X_test_scaled, y_test)
        
        # Create ensemble
        ensemble_metrics = self.create_ensemble_predictions(X_test_scaled, y_test)
        
        # Save models and metrics
        self.save_models()
        self.save_metrics(metrics, ensemble_metrics)
        
        print("\n" + "=" * 60)
        print("MODEL TRAINING COMPLETE!")
        print("=" * 60)
        
        return metrics
    
    def load_trained_models(self):
        """
        Load all trained models from disk.
        """
        model_files = {
            'random_forest': 'congestion_random_forest.pkl',
            'xgboost': 'congestion_xgboost.pkl',
            'linear_regression': 'congestion_linear_regression.pkl',
            'lstm': 'congestion_lstm.h5'
        }
        
        for name, filename in model_files.items():
            filepath = str(MODELS_DIR / filename)
            if os.path.exists(filepath):
                if filename.endswith('.h5'):
                    self.models[name] = load_model(filepath)
                else:
                    self.models[name] = joblib.load(filepath)
                print(f"Loaded {name} from {filepath}")
            else:
                print(f"Warning: {name} model not found at {filepath}")
        
        # Load preprocessor
        self.preprocessor.load_preprocessor()


if __name__ == "__main__":
    trainer = ModelTrainer()
    
    # Quick test with smaller dataset
    print("Starting model training with 5000 samples...")
    metrics = trainer.train_all_models(n_samples=5000)
    
    print("\nFinal Model Performance:")
    print("-" * 40)
    for model_name, model_metrics in metrics.items():
        print(f"{model_name}:")
        for metric, value in model_metrics.items():
            if not isinstance(value, dict):
                print(f"  {metric}: {value}")
        print()
