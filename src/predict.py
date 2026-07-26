"""
Prediction Module for Traffic Congestion Prediction.

Provides inference functions for trained models:
- Single prediction for real-time input
- Batch prediction for datasets
- Ensemble prediction combining multiple models
- Congestion level classification
- Confidence scoring
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import joblib
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (MODEL_CONFIG, MODELS_DIR, CONGESTION_LEVELS, 
                    WEATHER_CONDITIONS, DATA_DIR)
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer


class CongestionPredictor:
    """
    Makes congestion predictions using trained models.
    Supports both single and batch predictions.
    """
    
    def __init__(self, model_dir: str = None):
        """
        Initialize the predictor with trained models.
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = model_dir or str(MODELS_DIR)
        self.models = {}
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.metrics = {}
        self.is_loaded = False
        
    def load_models(self):
        """
        Load all trained models and preprocessor from disk.
        """
        print("Loading trained models...")
        
        # Load preprocessor
        preprocessor_path = os.path.join(self.model_dir, 'preprocessor.pkl')
        if os.path.exists(preprocessor_path):
            self.preprocessor.load_preprocessor(preprocessor_path)
        
        # Load models
        model_files = {
            'random_forest': 'congestion_random_forest.pkl',
            'xgboost': 'congestion_xgboost.pkl',
            'linear_regression': 'congestion_linear_regression.pkl',
            'lstm': 'congestion_lstm.h5'
        }
        
        for name, filename in model_files.items():
            filepath = os.path.join(self.model_dir, filename)
            if os.path.exists(filepath):
                try:
                    if filename.endswith('.h5'):
                        from tensorflow.keras.models import load_model
                        os.environ['TF_USE_LEGACY_KERAS'] = '1'
                        try:
                            self.models[name] = load_model(filepath, compile=False)
                        except Exception:
                            self.models[name] = load_model(filepath, compile=False, safe_mode=False)
                    else:
                        self.models[name] = joblib.load(filepath)
                    print(f"  ✓ Loaded {name}: {filepath}")
                except Exception as e:
                    print(f"  ✗ Failed to load {name}: {e}")
            else:
                print(f"  - Model {name} not found at {filepath}")
        
        # Load metrics
        metrics_path = os.path.join(self.model_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                self.metrics = json.load(f)
            print(f"  ✓ Loaded metrics from {metrics_path}")
        
        if self.models:
            self.is_loaded = True
            print(f"\nLoaded {len(self.models)} models successfully!")
        else:
            print("\n⚠ No models loaded. Using rule-based fallback prediction.")
    
    def preprocess_input(self, input_data: Dict) -> np.ndarray:
        """
        Preprocess input data for prediction.
        
        Args:
            input_data: Dictionary of input features
            
        Returns:
            Preprocessed feature array
        """
        # Convert input to DataFrame
        df = pd.DataFrame([input_data])
        
        # The scaler was trained on exactly these 13 features (in this order):
        # hour, day_of_week, month, is_weekend, is_holiday, temperature,
        # humidity, precipitation, wind_speed, traffic_volume, avg_speed,
        # location_encoded, is_rush_hour
        
        # Encode location as location_encoded using label encoder
        if 'location' in df.columns:
            if 'location' in self.preprocessor.label_encoders:
                le = self.preprocessor.label_encoders['location']
                known = list(le.classes_)
                loc_val = df['location'].iloc[0]
                if isinstance(loc_val, str) and loc_val in known:
                    df['location_encoded'] = le.transform([loc_val])[0]
                else:
                    df['location_encoded'] = -1
            else:
                df['location_encoded'] = hash(str(df['location'].iloc[0])) % 100
        
        # Define the exact feature order the scaler expects
        expected_features = [
            'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday',
            'temperature', 'humidity', 'precipitation', 'wind_speed',
            'traffic_volume', 'avg_speed', 'location_encoded', 'is_rush_hour'
        ]
        
        # Ensure all expected features exist with defaults
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0
        
        # Select features in the exact expected order
        X = df[expected_features].values.astype(np.float64)
        
        # Scale features if scaler is fitted
        if hasattr(self.preprocessor.scaler, 'mean_') and self.preprocessor.scaler.mean_ is not None:
            X = self.preprocessor.scaler.transform(X)
        
        return X
    
    def predict(self, input_data: Union[Dict, pd.DataFrame], 
               model_name: str = 'ensemble') -> Dict:
        """
        Make congestion prediction for a single input.
        
        Args:
            input_data: Input features dictionary or DataFrame
            model_name: Model to use ('random_forest', 'xgboost', 'lstm', 
                       'linear_regression', 'ensemble')
            
        Returns:
            Dictionary with prediction results
        """
        # Convert weather string to numeric code if needed
        if isinstance(input_data, dict):
            weather_map = {'clear': 0, 'cloudy': 1, 'rainy': 2, 'stormy': 3, 'foggy': 4, 'snowy': 5}
            if 'weather_condition' in input_data and isinstance(input_data['weather_condition'], str):
                wc = input_data['weather_condition'].lower().strip()
                input_data['weather_condition'] = weather_map.get(wc, 0)

        # Preprocess input
        if isinstance(input_data, dict):
            X = self.preprocess_input(input_data)
        else:
            X = input_data.values
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
        
        # Get predictions from each model
        predictions = {}
        
        for name, model in self.models.items():
            try:
                if name == 'lstm':
                    X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
                    pred = model.predict(X_reshaped, verbose=0).flatten()
                else:
                    pred = model.predict(X)
                predictions[name] = float(pred[0])
            except Exception as e:
                print(f"Warning: {name} prediction failed: {e}")
                predictions[name] = None
        
        # Calculate ensemble prediction
        if model_name == 'ensemble' and predictions:
            weights = {'random_forest': 0.30, 'xgboost': 0.35, 
                      'lstm': 0.25, 'linear_regression': 0.10}
            valid_preds = {k: v for k, v in predictions.items() if v is not None}
            if valid_preds:
                total_weight = sum(weights.get(k, 0.25) for k in valid_preds.keys())
                final_pred = sum(
                    valid_preds[k] * weights.get(k, 0.25) / total_weight 
                    for k in valid_preds.keys()
                )
            else:
                final_pred = 0.5
        elif model_name in predictions and predictions[model_name] is not None:
            final_pred = predictions[model_name]
        else:
            final_pred = self._heuristic_prediction(input_data)
        
        # Convert to congestion level
        congestion_value = np.clip(final_pred, 0, 3)
        congestion_level = int(round(congestion_value))
        congestion_label = CONGESTION_LEVELS.get(congestion_level, 'Unknown')
        
        # Calculate confidence score
        confidence = self._calculate_confidence(predictions, congestion_level)
        
        result = {
            'congestion_value': round(congestion_value, 2),
            'congestion_level': congestion_level,
            'congestion_label': congestion_label,
            'confidence': round(confidence, 2),
            'model_used': model_name,
            'model_predictions': predictions,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def predict_batch(self, df: pd.DataFrame, model_name: str = 'ensemble') -> pd.DataFrame:
        """
        Make predictions for a batch of inputs.
        
        Args:
            df: DataFrame with input features
            model_name: Model to use
            
        Returns:
            DataFrame with predictions added
        """
        results = df.copy()
        
        predictions_list = []
        for idx, row in df.iterrows():
            input_dict = row.to_dict()
            result = self.predict(input_dict, model_name)
            predictions_list.append(result)
        
        results['predicted_congestion'] = [r['congestion_value'] for r in predictions_list]
        results['predicted_level'] = [r['congestion_level'] for r in predictions_list]
        results['predicted_label'] = [r['congestion_label'] for r in predictions_list]
        results['confidence'] = [r['confidence'] for r in predictions_list]
        
        return results
    
    def _heuristic_prediction(self, input_data: Dict) -> float:
        """
        Rule-based fallback prediction when models are not available.
        
        Args:
            input_data: Input features
            
        Returns:
            Congestion value (0-3)
        """
        score = 0.5  # Base score
        
        # Time-based factors
        hour = input_data.get('hour', 12)
        is_weekend = input_data.get('is_weekend', 0)
        is_rush_hour = input_data.get('is_rush_hour', 0)
        
        if is_rush_hour:
            score += 0.8
        elif 10 <= hour <= 15:
            score += 0.3
        elif 20 <= hour or hour <= 5:
            score -= 0.3
        
        if is_weekend:
            score -= 0.2
        
        # Traffic volume factor
        traffic_volume = input_data.get('traffic_volume', 500)
        if traffic_volume > 1000:
            score += 0.5
        elif traffic_volume > 700:
            score += 0.2
        elif traffic_volume < 300:
            score -= 0.2
        
        # Average speed factor
        avg_speed = input_data.get('avg_speed', 50)
        if avg_speed < 20:
            score += 0.8
        elif avg_speed < 35:
            score += 0.3
        elif avg_speed > 60:
            score -= 0.2
        
        # Weather factors
        weather_condition = input_data.get('weather_condition', 0)
        precipitation = input_data.get('precipitation', 0)
        
        if weather_condition >= 3 or precipitation > 10:
            score += 0.4
        elif weather_condition >= 1:
            score += 0.1
        
        # Clip to 0-3 range
        return np.clip(score, 0, 3)
    
    def _calculate_confidence(self, predictions: Dict, congestion_level: int) -> float:
        """
        Calculate confidence score based on model agreement.
        
        Args:
            predictions: Model predictions dictionary
            congestion_level: Predicted congestion level
            
        Returns:
            Confidence score (0-1)
        """
        valid_preds = [v for v in predictions.values() if v is not None]
        
        if not valid_preds:
            return 0.3
        
        # Check agreement among models
        levels = [int(round(v)) for v in valid_preds]
        agreement = levels.count(congestion_level) / len(levels)
        
        # Check variance
        variance = np.var(valid_preds) if len(valid_preds) > 1 else 0
        
        # Confidence: higher agreement and lower variance = higher confidence
        confidence = agreement * (1 - min(variance, 1) * 0.5)
        
        return np.clip(confidence, 0, 1)
    
    def get_model_info(self) -> Dict:
        """
        Get information about loaded models and their performance.
        
        Returns:
            Dictionary with model info
        """
        info = {
            'models_loaded': list(self.models.keys()),
            'num_models': len(self.models),
            'performance': self.metrics,
            'preprocessor_fitted': hasattr(self.preprocessor.scaler, 'mean_') and 
                                   self.preprocessor.scaler.mean_ is not None
        }
        return info


# Utility function for quick prediction from command line
def quick_predict(hour: int = 14, day_of_week: int = 1, month: int = 6,
                  temperature: float = 25, humidity: float = 60,
                  precipitation: float = 0, wind_speed: float = 10,
                  traffic_volume: int = 800, avg_speed: int = 45,
                  weather_condition: int = 0, location: str = 'Downtown',
                  model_name: str = 'ensemble') -> Dict:
    """
    Quick prediction function with default values.
    
    Args:
        hour: Hour of day (0-23)
        day_of_week: Day of week (0-6, Mon-Sun)
        month: Month (1-12)
        temperature: Temperature in Celsius
        humidity: Humidity percentage
        precipitation: Precipitation in mm
        wind_speed: Wind speed in km/h
        traffic_volume: Number of vehicles
        avg_speed: Average speed in km/h
        weather_condition: Weather code (0-5)
        location: Location name
        model_name: Model to use
        
    Returns:
        Prediction result dictionary
    """
    predictor = CongestionPredictor()
    predictor.load_models()
    
    input_data = {
        'hour': hour,
        'day_of_week': day_of_week,
        'month': month,
        'is_weekend': 1 if day_of_week >= 5 else 0,
        'is_rush_hour': 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0,
        'temperature': temperature,
        'humidity': humidity,
        'precipitation': precipitation,
        'wind_speed': wind_speed,
        'traffic_volume': traffic_volume,
        'avg_speed': avg_speed,
        'weather_condition': weather_condition,
        'location': location
    }
    
    return predictor.predict(input_data, model_name)


if __name__ == "__main__":
    # Test prediction
    print("Testing CongestionPredictor...\n")
    
    # Test with sample input
    sample = {
        'hour': 17,
        'day_of_week': 1,
        'month': 6,
        'is_weekend': 0,
        'is_rush_hour': 1,
        'temperature': 32,
        'humidity': 65,
        'precipitation': 0.2,
        'wind_speed': 12,
        'traffic_volume': 850,
        'avg_speed': 35,
        'weather_condition': 2,
        'location': 'Downtown'
    }
    
    predictor = CongestionPredictor()
    predictor.load_models()
    
    if predictor.is_loaded:
        result = predictor.predict(sample)
        print("\nPrediction Result:")
        print(f"  Congestion Value: {result['congestion_value']}")
        print(f"  Congestion Level: {result['congestion_level']} ({result['congestion_label']})")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Model Used: {result['model_used']}")
        print(f"  Individual Model Predictions:")
        for model_name, pred in result['model_predictions'].items():
            print(f"    {model_name}: {pred:.4f}" if pred is not None else f"    {model_name}: N/A")
    else:
        print("No models loaded. Using heuristic prediction...")
        result = predictor.predict(sample)
        print(f"\nHeuristic Prediction: Level {result['congestion_level']} ({result['congestion_label']})")
        print(f"Confidence: {result['confidence']:.2%}")