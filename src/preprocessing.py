"""
Data Preprocessing Module for Traffic Congestion Prediction.

Handles:
- Data loading and validation
- Missing value imputation
- Outlier detection and treatment
- Data scaling and normalization
- Train-test-validation splitting
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import warnings
import sys
import os
from typing import Tuple, Dict, Optional, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG, DATA_DIR, MODELS_DIR

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Preprocesses traffic data for ML model training and prediction.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize the preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or MODEL_CONFIG
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.fitted = False
        
    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """
        Load traffic dataset from CSV.
        
        Args:
            filepath: Path to CSV file (defaults to synthetic data)
            
        Returns:
            Loaded DataFrame
        """
        if filepath is None:
            filepath = str(DATA_DIR / 'synthetic_traffic_data.csv')
            
        if not os.path.exists(filepath):
            print(f"Data file not found at {filepath}")
            print("Generating synthetic data instead...")
            from src.data_generator import TrafficDataGenerator
            generator = TrafficDataGenerator()
            data = generator.generate_dataset()
            generator.save_dataset(data, filepath)
        else:
            data = pd.read_csv(filepath)
            print(f"Data loaded from {filepath}: {data.shape}")
            
        return data
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with imputed missing values
        """
        df_clean = df.copy()
        
        # Check for missing values
        missing_count = df_clean.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values. Imputing...")
            
            # Separate numeric and categorical columns
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
            
            # Impute numeric columns with median
            if len(numeric_cols) > 0:
                numeric_imputer = SimpleImputer(strategy='median')
                df_clean[numeric_cols] = numeric_imputer.fit_transform(df_clean[numeric_cols])
                
            # Impute categorical columns with mode
            if len(categorical_cols) > 0:
                categorical_imputer = SimpleImputer(strategy='most_frequent')
                df_clean[categorical_cols] = categorical_imputer.fit_transform(df_clean[categorical_cols])
        else:
            print("No missing values found.")
            
        return df_clean
    
    def detect_and_handle_outliers(self, df: pd.DataFrame, columns: List[str] = None, 
                                  method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect and handle outliers in numeric columns.
        
        Args:
            df: Input DataFrame
            columns: List of columns to check (defaults to all numeric)
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with handled outliers
        """
        df_clean = df.copy()
        
        if columns is None:
            columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude target variable and encoded features
            exclude_cols = ['congestion_level', 'location_encoded', 'weather_condition_encoded',
                           'is_weekend', 'is_holiday', 'is_rush_hour']
            columns = [col for col in columns if col not in exclude_cols]
        
        print(f"Checking outliers in {len(columns)} columns using {method} method...")
        
        for col in columns:
            if col not in df_clean.columns:
                continue
                
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
                if outliers > 0:
                    # Cap the outliers
                    df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
                    
            elif method == 'zscore':
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                z_scores = np.abs((df_clean[col] - mean) / std)
                
                outliers = (z_scores > threshold).sum()
                if outliers > 0:
                    # Replace outliers with median
                    median_val = df_clean[col].median()
                    df_clean.loc[z_scores > threshold, col] = median_val
        
        return df_clean
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                   columns: List[str] = None) -> pd.DataFrame:
        """
        Encode categorical features using Label Encoding.
        
        Args:
            df: Input DataFrame
            columns: Columns to encode (defaults to object/category columns)
            
        Returns:
            DataFrame with encoded features
        """
        df_encoded = df.copy()
        
        if columns is None:
            columns = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
            
        for col in columns:
            if col in df_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                        df_encoded[col].astype(str)
                    )
                else:
                    # Handle unseen categories
                    known_classes = list(self.label_encoders[col].classes_)
                    df_encoded[f'{col}_encoded'] = df_encoded[col].apply(
                        lambda x: known_classes.index(x) if x in known_classes else -1
                    )
                    
        return df_encoded
    
    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray, 
                      X_val: np.ndarray = None) -> Tuple[np.ndarray, ...]:
        """
        Scale features using StandardScaler.
        
        Args:
            X_train: Training features
            X_test: Test features
            X_val: Validation features (optional)
            
        Returns:
            Scaled feature arrays
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            return X_train_scaled, X_test_scaled, X_val_scaled
            
        return X_train_scaled, X_test_scaled
    
    def prepare_features_target(self, df: pd.DataFrame, 
                               target_col: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split DataFrame into features and target.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            X: Feature DataFrame
            y: Target Series
        """
        if target_col is None:
            target_col = self.config['target_column']
            
        # Get feature columns
        feature_cols = self.config['feature_columns']
        
        # Ensure all feature columns exist
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if len(available_cols) < len(feature_cols):
            missing = set(feature_cols) - set(available_cols)
            print(f"Warning: Missing feature columns: {missing}")
            
        X = df[available_cols].copy()
        y = df[target_col].copy() if target_col in df.columns else None
        
        print(f"Features: {X.shape[1]} columns, {X.shape[0]} rows")
        if y is not None:
            print(f"Target distribution:\n{y.value_counts().sort_index()}")
            
        return X, y
    
    def train_test_val_split(self, X: pd.DataFrame, y: pd.Series,
                            test_size: float = None, val_size: float = None,
                            random_state: int = None
                            ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                      pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, test, and validation sets.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Test set proportion
            val_size: Validation set proportion from training
            random_state: Random seed
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        if test_size is None:
            test_size = self.config['test_size']
        if val_size is None:
            val_size = self.config['validation_size']
        if random_state is None:
            random_state = self.config['random_state']
            
        # First split: separate test set
        # Use stratified bins for continuous target regression
        try:
            y_binned = pd.qcut(y.rank(method='first'), q=10, labels=False, duplicates='drop')
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y_binned
            )
            
            # Second split: separate validation from training
            val_relative_size = val_size / (1 - test_size)
            y_temp_binned = pd.qcut(y_temp.rank(method='first'), q=10, labels=False, duplicates='drop')
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_relative_size, 
                random_state=random_state, stratify=y_temp_binned
            )
        except Exception:
            # Fall back to non-stratified split if stratification fails
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            val_relative_size = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_relative_size, 
                random_state=random_state
            )
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessor(self, filepath: str = None):
        """
        Save the preprocessor (scaler, encoders) to disk.
        
        Args:
            filepath: Path to save (defaults to models/preprocessor.pkl)
        """
        if filepath is None:
            filepath = str(MODELS_DIR / 'preprocessor.pkl')
            
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'imputer': self.imputer,
            'config': self.config,
            'fitted': self.fitted
        }
        
        joblib.dump(preprocessor_data, filepath)
        print(f"Preprocessor saved to: {filepath}")
        
    def load_preprocessor(self, filepath: str = None):
        """
        Load preprocessor from disk.
        
        Args:
            filepath: Path to load from (defaults to models/preprocessor.pkl)
        """
        if filepath is None:
            filepath = str(MODELS_DIR / 'preprocessor.pkl')
            
        if os.path.exists(filepath):
            preprocessor_data = joblib.load(filepath)
            self.scaler = preprocessor_data['scaler']
            self.label_encoders = preprocessor_data['label_encoders']
            self.imputer = preprocessor_data['imputer']
            self.config = preprocessor_data['config']
            self.fitted = preprocessor_data['fitted']
            print(f"Preprocessor loaded from: {filepath}")
        else:
            print(f"Preprocessor file not found at {filepath}")
    
    def preprocess_pipeline(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Run complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit transformers (True for training, False for inference)
            
        Returns:
            Preprocessed DataFrame
        """
        print("Running preprocessing pipeline...")
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Handle outliers
        df = self.detect_and_handle_outliers(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        if fit:
            self.fitted = True
            
        print("Preprocessing complete.")
        return df


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = DataPreprocessor()
    data = preprocessor.load_data()
    data = preprocessor.preprocess_pipeline(data, fit=True)
    X, y = preprocessor.prepare_features_target(data)
    
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.train_test_val_split(X, y)
    
    # Save preprocessor
    preprocessor.save_preprocessor()
    
    print("\nPreprocessing test complete!")
