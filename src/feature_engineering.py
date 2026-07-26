"""
Feature Engineering Module for Traffic Congestion Prediction.

Creates advanced features from raw data:
- Time-based cyclical features
- Lag features (traffic volume and speed)
- Rolling window statistics
- Interaction features
- Weather impact scores
- Rush hour indicators
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG


class FeatureEngineer:
    """
    Creates and transforms features for traffic congestion prediction.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize the feature engineer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or MODEL_CONFIG
        self.fitted = False
        
    def create_lag_features(self, df: pd.DataFrame, columns: List[str] = None,
                           lag_hours: List[int] = None) -> pd.DataFrame:
        """
        Create lag features for time series data.
        
        Args:
            df: Input DataFrame (must be sorted by timestamp)
            columns: Columns to create lags for
            lag_hours: List of lag periods in hours
            
        Returns:
            DataFrame with lag features added
        """
        df_feat = df.copy()
        
        if columns is None:
            columns = ['traffic_volume', 'avg_speed']
        if lag_hours is None:
            lag_hours = [1, 2, 3, 6, 12, 24]
            
        for col in columns:
            if col not in df_feat.columns:
                continue
            for lag in lag_hours:
                df_feat[f'{col}_lag_{lag}h'] = df_feat[col].shift(lag)
                
        print(f"Created lag features for columns: {columns}")
        return df_feat
    
    def create_rolling_features(self, df: pd.DataFrame, columns: List[str] = None,
                               windows: List[int] = None) -> pd.DataFrame:
        """
        Create rolling window statistics.
        
        Args:
            df: Input DataFrame
            columns: Columns to create rolling features for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features added
        """
        df_feat = df.copy()
        
        if columns is None:
            columns = ['traffic_volume', 'avg_speed']
        if windows is None:
            windows = [3, 6, 12, 24]
            
        for col in columns:
            if col not in df_feat.columns:
                continue
            for window in windows:
                # Rolling mean
                df_feat[f'{col}_rolling_mean_{window}h'] = df_feat[col].rolling(
                    window=window, min_periods=1
                ).mean()
                # Rolling std
                df_feat[f'{col}_rolling_std_{window}h'] = df_feat[col].rolling(
                    window=window, min_periods=1
                ).std().fillna(0)
                # Rolling max
                df_feat[f'{col}_rolling_max_{window}h'] = df_feat[col].rolling(
                    window=window, min_periods=1
                ).max()
                # Rolling min
                df_feat[f'{col}_rolling_min_{window}h'] = df_feat[col].rolling(
                    window=window, min_periods=1
                ).min()
                
        print(f"Created rolling features for columns: {columns}")
        return df_feat
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features added
        """
        df_feat = df.copy()
        
        # Traffic volume × weather condition interaction
        if 'traffic_volume' in df_feat.columns and 'weather_condition' in df_feat.columns:
            df_feat['volume_weather_interaction'] = (
                df_feat['traffic_volume'] * (df_feat['weather_condition'] + 1) / 1000
            )
            
        # Speed × volume ratio (congestion indicator)
        if all(col in df_feat.columns for col in ['avg_speed', 'traffic_volume']):
            df_feat['speed_volume_ratio'] = np.where(
                df_feat['traffic_volume'] > 0,
                df_feat['avg_speed'] / df_feat['traffic_volume'],
                0
            )
            
        # Rush hour × traffic volume
        if all(col in df_feat.columns for col in ['is_rush_hour', 'traffic_volume']):
            df_feat['rush_hour_volume'] = df_feat['is_rush_hour'] * df_feat['traffic_volume']
            
        # Weekend × traffic volume
        if all(col in df_feat.columns for col in ['is_weekend', 'traffic_volume']):
            df_feat['weekend_volume'] = df_feat['is_weekend'] * df_feat['traffic_volume']
            
        # Temperature × humidity (heat index proxy)
        if all(col in df_feat.columns for col in ['temperature', 'humidity']):
            df_feat['temp_humidity_index'] = (
                df_feat['temperature'] * 0.5 + df_feat['humidity'] * 0.5
            )
            
        # Precipitation × wind (storm severity)
        if all(col in df_feat.columns for col in ['precipitation', 'wind_speed']):
            df_feat['storm_severity'] = (
                df_feat['precipitation'] * 0.6 + df_feat['wind_speed'] * 0.4
            )
            
        print(f"Created interaction features")
        return df_feat
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional time-based features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional time features
        """
        df_feat = df.copy()
        
        # Time since midnight in minutes
        if 'hour' in df_feat.columns:
            df_feat['minutes_from_midnight'] = df_feat['hour'] * 60
            
        # Part of day
        if 'hour' in df_feat.columns:
            def get_part_of_day(hour):
                if 5 <= hour < 12:
                    return 0  # Morning
                elif 12 <= hour < 17:
                    return 1  # Afternoon
                elif 17 <= hour < 21:
                    return 2  # Evening
                else:
                    return 3  # Night
            df_feat['part_of_day'] = df_feat['hour'].apply(get_part_of_day)
            
        # Is business hours (9 AM - 5 PM on weekdays)
        if all(col in df_feat.columns for col in ['hour', 'is_weekend']):
            df_feat['is_business_hours'] = (
                (df_feat['hour'] >= 9) & 
                (df_feat['hour'] <= 17) & 
                (df_feat['is_weekend'] == 0)
            ).astype(int)
            
        # Days since start
        if 'month' in df_feat.columns and 'day_of_month' in df_feat.columns:
            df_feat['day_of_year'] = (
                df_feat['month'] * 30 + df_feat['day_of_month']
            )
            
        return df_feat
    
    def create_weather_impact_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite weather impact score.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with weather impact score
        """
        df_feat = df.copy()
        
        weather_cols = ['temperature', 'humidity', 'precipitation', 'wind_speed', 'weather_condition']
        available_cols = [col for col in weather_cols if col in df_feat.columns]
        
        if len(available_cols) >= 3:
            # Normalize components to 0-1 scale
            scores = []
            
            if 'precipitation' in df_feat.columns:
                precip_score = df_feat['precipitation'] / df_feat['precipitation'].max()
                scores.append(precip_score * 0.3)
                
            if 'wind_speed' in df_feat.columns:
                wind_score = df_feat['wind_speed'] / df_feat['wind_speed'].max()
                scores.append(wind_score * 0.2)
                
            if 'temperature' in df_feat.columns:
                # Extreme temperatures have higher impact
                temp_score = np.abs(df_feat['temperature'] - 25) / 25
                scores.append(temp_score * 0.2)
                
            if 'humidity' in df_feat.columns:
                humidity_score = df_feat['humidity'] / 100
                scores.append(humidity_score * 0.15)
                
            if 'weather_condition' in df_feat.columns:
                weather_score = df_feat['weather_condition'] / 5
                scores.append(weather_score * 0.15)
                
            if scores:
                df_feat['weather_impact_score'] = sum(scores)
                
        print("Created weather impact score")
        return df_feat
    
    def create_congestion_trend(self, df: pd.DataFrame, window: int = 6) -> pd.DataFrame:
        """
        Create congestion trend indicator (increasing, decreasing, stable).
        
        Args:
            df: Input DataFrame
            window: Window size for trend calculation
            
        Returns:
            DataFrame with congestion trend
        """
        df_feat = df.copy()
        
        if 'congestion_level' in df_feat.columns:
            # Rolling mean of congestion
            congestion_smooth = df_feat['congestion_level'].rolling(
                window=window, min_periods=1
            ).mean()
            
            # Trend direction
            congestion_shifted = congestion_smooth.shift(1)
            trend = congestion_smooth - congestion_shifted
            
            df_feat['congestion_trend'] = np.select(
                [trend > 0.1, trend < -0.1],
                [1, -1],
                default=0
            )
            
            # Congestion acceleration (rate of change)
            df_feat['congestion_acceleration'] = trend.diff().fillna(0)
            
        return df_feat
    
    def engineer_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit (True for training)
            
        Returns:
            DataFrame with engineered features
        """
        print("Running feature engineering pipeline...")
        
        # Apply all feature engineering steps
        df = self.create_time_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.create_interaction_features(df)
        df = self.create_weather_impact_score(df)
        df = self.create_congestion_trend(df)
        
        # Drop rows with NaN from lag/rolling features
        initial_rows = len(df)
        df = df.dropna()
        print(f"Dropped {initial_rows - len(df)} rows with NaN values from feature engineering")
        
        if fit:
            self.fitted = True
            
        print(f"Feature engineering complete. Total features: {df.shape[1]}")
        print(f"Columns: {list(df.columns)}")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of all engineered feature names.
        
        Returns:
            List of feature column names
        """
        return self.config.get('feature_columns', [])


if __name__ == "__main__":
    # Test feature engineering
    from src.data_generator import TrafficDataGenerator
    from src.preprocessing import DataPreprocessor
    
    generator = TrafficDataGenerator()
    data = generator.generate_dataset(1000)
    
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_pipeline(data, fit=True)
    
    engineer = FeatureEngineer()
    data_engineered = engineer.engineer_features(data)
    
    print(f"\nOriginal columns: {data.shape[1]}")
    print(f"Engineered columns: {data_engineered.shape[1]}")
    print(f"New features: {set(data_engineered.columns) - set(data.columns)}")