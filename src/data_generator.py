"""
Synthetic Traffic Data Generator for Traffic Congestion Prediction.

Generates realistic traffic data with:
- Time-based features (hour, day, month, weekend, holiday, rush hour)
- Weather conditions (temperature, humidity, precipitation, wind speed)
- Traffic metrics (volume, average speed)
- Location data
- Congestion levels (target variable: 0=Low, 1=Moderate, 2=High, 3=Severe)
"""

import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime, timedelta
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_GENERATION, DATA_DIR, CONGESTION_LEVELS


class TrafficDataGenerator:
    """
    Generates synthetic traffic congestion data for model training and demonstration.
    """

    def __init__(self, config: dict = None):
        """
        Initialize the data generator.

        Args:
            config: Configuration dictionary (defaults to DATA_GENERATION from config.py)
        """
        self.config = config or DATA_GENERATION
        self.rng = np.random.RandomState(self.config.get('random_seed', 42))
        self.locations = self.config.get('locations', ['Downtown'])
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        assert len(self.locations) > 0, "At least one location is required"
        assert self.config.get('n_samples', 0) > 0, "n_samples must be positive"

    def _generate_timestamps(self, n_samples: int) -> pd.Series:
        """
        Generate random timestamps within the configured date range.

        Args:
            n_samples: Number of timestamps to generate

        Returns:
            Series of datetime objects
        """
        start = pd.Timestamp(self.config.get('start_date', '2024-01-01'))
        end = pd.Timestamp(self.config.get('end_date', '2024-12-31'))
        total_seconds = int((end - start).total_seconds())
        random_seconds = self.rng.randint(0, total_seconds, size=n_samples)
        timestamps = start + pd.to_timedelta(random_seconds, unit='s')
        return pd.Series(timestamps, name='timestamp')

    def _generate_hourly_pattern(self, hour: int, is_weekend: bool) -> float:
        """
        Generate a base traffic multiplier based on hour of day.

        Args:
            hour: Hour of day (0-23)
            is_weekend: Whether it's a weekend

        Returns:
            Traffic multiplier (higher = more traffic)
        """
        if is_weekend:
            # Weekend: moderate traffic during day, low at night
            if 8 <= hour <= 11:
                return 0.7
            elif 12 <= hour <= 17:
                return 0.8
            elif 18 <= hour <= 22:
                return 0.6
            else:
                return 0.2
        else:
            # Weekday: rush hour peaks
            if 7 <= hour <= 9:     # Morning rush
                return 1.0 + 0.2 * np.sin(np.pi * (hour - 7) / 2)
            elif 10 <= hour <= 15: # Midday
                return 0.6 + 0.1 * np.sin(np.pi * (hour - 10) / 5)
            elif 16 <= hour <= 19: # Evening rush
                return 1.0 + 0.2 * np.sin(np.pi * (hour - 16) / 3)
            elif 20 <= hour <= 23: # Evening
                return 0.4
            else:                   # Late night / early morning
                return 0.1

    def _generate_weather_condition(self) -> int:
        """
        Generate a random weather condition code.

        Returns:
            Weather code: 0=Clear, 1=Cloudy, 2=Rainy, 3=Stormy, 4=Foggy, 5=Snowy
        """
        # Weighted distribution: mostly clear/cloudy, less stormy/snowy
        weights = [0.35, 0.25, 0.20, 0.08, 0.07, 0.05]
        return int(self.rng.choice(6, p=weights))

    def _generate_weather_data(self, n_samples: int) -> pd.DataFrame:
        """
        Generate realistic weather data.

        Args:
            n_samples: Number of samples

        Returns:
            DataFrame with weather columns
        """
        weather_conditions = [self._generate_weather_condition() for _ in range(n_samples)]

        temperature = []
        humidity = []
        precipitation = []
        wind_speed = []

        for wc in weather_conditions:
            if wc == 0:  # Clear
                temperature.append(self.rng.normal(28, 5))
                humidity.append(self.rng.normal(45, 10))
                precipitation.append(0)
                wind_speed.append(self.rng.normal(8, 4))
            elif wc == 1:  # Cloudy
                temperature.append(self.rng.normal(22, 4))
                humidity.append(self.rng.normal(60, 10))
                precipitation.append(self.rng.exponential(0.5))
                wind_speed.append(self.rng.normal(12, 5))
            elif wc == 2:  # Rainy
                temperature.append(self.rng.normal(18, 4))
                humidity.append(self.rng.normal(80, 8))
                precipitation.append(self.rng.exponential(5) + 1)
                wind_speed.append(self.rng.normal(18, 6))
            elif wc == 3:  # Stormy
                temperature.append(self.rng.normal(15, 3))
                humidity.append(self.rng.normal(90, 5))
                precipitation.append(self.rng.exponential(15) + 5)
                wind_speed.append(self.rng.normal(35, 10))
            elif wc == 4:  # Foggy
                temperature.append(self.rng.normal(12, 3))
                humidity.append(self.rng.normal(85, 8))
                precipitation.append(self.rng.exponential(1))
                wind_speed.append(self.rng.normal(5, 3))
            else:  # Snowy (5)
                temperature.append(self.rng.normal(-2, 4))
                humidity.append(self.rng.normal(75, 10))
                precipitation.append(self.rng.exponential(3) + 1)
                wind_speed.append(self.rng.normal(15, 7))

        return pd.DataFrame({
            'weather_condition': weather_conditions,
            'temperature': [max(-10, min(45, t)) for t in temperature],
            'humidity': [max(10, min(100, h)) for h in humidity],
            'precipitation': [max(0, p) for p in precipitation],
            'wind_speed': [max(0, w) for w in wind_speed]
        })

    def _compute_congestion_level(self, row: pd.Series) -> float:
        """
        Compute congestion level based on all features.

        Args:
            row: A single data row

        Returns:
            Congestion level (0.0 - 3.0)
        """
        score = 0.3  # Base congestion

        # Time factors
        hour = int(row['hour'])
        is_weekend = int(row['is_weekend'])
        is_rush_hour = int(row['is_rush_hour'])

        if is_rush_hour:
            score += 0.8
        elif 10 <= hour <= 15:
            score += 0.2
        elif 20 <= hour or hour <= 5:
            score -= 0.2

        if is_weekend:
            score -= 0.15

        # Traffic volume factor
        volume = row['traffic_volume']
        if volume > 1000:
            score += 0.5
        elif volume > 700:
            score += 0.25
        elif volume < 300:
            score -= 0.2

        # Speed factor (lower speed = more congestion)
        speed = row['avg_speed']
        if speed < 20:
            score += 0.7
        elif speed < 35:
            score += 0.3
        elif speed > 60:
            score -= 0.2

        # Weather factors
        weather = int(row['weather_condition'])
        precip = row['precipitation']

        if weather >= 3 or precip > 10:
            score += 0.5
        elif weather >= 2 or precip > 3:
            score += 0.2
        elif weather == 4:  # Foggy
            score += 0.3

        # Temperature extremes
        temp = row['temperature']
        if temp > 38 or temp < 0:
            score += 0.2
        elif temp > 33 or temp < 5:
            score += 0.1

        # Location-based variation
        location_idx = self.locations.index(row['location']) if row['location'] in self.locations else 0
        # Downtown, Highway, Bridge, Industrial have higher baseline congestion
        location_bias = {
            'Downtown': 0.3,
            'Highway_A': 0.2,
            'Highway_B': 0.2,
            'Commercial_A': 0.2,
            'Airport_Road': 0.15,
            'Ring_Road': 0.15,
            'Bridge_1': 0.25,
            'Bridge_2': 0.25,
            'Market_Area': 0.2,
            'Industrial_Zone': 0.15,
            'University_Area': 0.1,
            'Residential_A': -0.1
        }
        score += location_bias.get(row['location'], 0)

        # Add random noise
        noise = self.rng.normal(0, 0.15)
        score += noise

        return np.clip(score, 0.0, 3.0)

    def generate_dataset(self, n_samples: int = None) -> pd.DataFrame:
        """
        Generate a complete synthetic traffic dataset.

        Args:
            n_samples: Number of samples (defaults to config value)

        Returns:
            DataFrame with all features and target congestion level
        """
        if n_samples is None:
            n_samples = self.config.get('n_samples', 10000)

        print(f"Generating {n_samples} synthetic traffic samples...")

        # Generate timestamps
        timestamps = self._generate_timestamps(n_samples)

        # Extract time features
        df = pd.DataFrame({
            'timestamp': timestamps,
            'hour': timestamps.dt.hour.astype(int),
            'day_of_week': timestamps.dt.dayofweek.astype(int),
            'day_of_month': timestamps.dt.day.astype(int),
            'month': timestamps.dt.month.astype(int),
            'is_weekend': (timestamps.dt.dayofweek >= 5).astype(int),
            'is_holiday': np.zeros(n_samples, dtype=int),  # Simplified: no holidays
        })

        # Generate is_rush_hour
        df['is_rush_hour'] = (
            ((df['hour'] >= 7) & (df['hour'] <= 9)) |
            ((df['hour'] >= 17) & (df['hour'] <= 19))
        ).astype(int) & (1 - df['is_weekend'])

        # Generate traffic multiplier based on time
        traffic_mult = df.apply(
            lambda r: self._generate_hourly_pattern(r['hour'], r['is_weekend']),
            axis=1
        )

        # Generate base traffic volume and speed
        base_volume = self.rng.normal(500, 150, size=n_samples)
        base_speed = self.rng.normal(50, 10, size=n_samples)

        # Apply time multiplier with noise
        df['traffic_volume'] = (base_volume * traffic_mult + self.rng.normal(0, 50, n_samples)).clip(0, 2000).astype(int)
        df['avg_speed'] = (base_speed - traffic_mult * 20 + self.rng.normal(0, 5, n_samples)).clip(5, 120).astype(int)

        # Assign locations
        df['location'] = self.rng.choice(self.locations, size=n_samples)

        # Generate weather data
        weather_df = self._generate_weather_data(n_samples)
        df = pd.concat([df, weather_df], axis=1)

        # Compute congestion level
        df['congestion_level'] = df.apply(self._compute_congestion_level, axis=1)

        # Round congestion level
        df['congestion_level'] = df['congestion_level'].round(2)

        # Ensure consistent ordering
        column_order = [
            'timestamp', 'hour', 'day_of_week', 'day_of_month', 'month',
            'is_weekend', 'is_holiday', 'is_rush_hour',
            'location', 'weather_condition',
            'temperature', 'humidity', 'precipitation', 'wind_speed',
            'traffic_volume', 'avg_speed', 'congestion_level'
        ]
        available_cols = [c for c in column_order if c in df.columns]
        df = df[available_cols]

        print(f"Dataset generated: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Congestion distribution:\n{df['congestion_level'].describe()}")

        return df

    def save_dataset(self, data: pd.DataFrame, filepath: Optional[str] = None):
        """
        Save generated dataset to CSV.

        Args:
            data: DataFrame to save
            filepath: Path to save (defaults to data/synthetic_traffic_data.csv)
        """
        if filepath is None:
            filepath = str(DATA_DIR / 'synthetic_traffic_data.csv')

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        data.to_csv(filepath, index=False)
        print(f"Dataset saved to: {filepath}")
        print(f"File size: {os.path.getsize(filepath) / 1024:.1f} KB")

    def load_dataset(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load dataset from CSV.

        Args:
            filepath: Path to CSV file (defaults to data/synthetic_traffic_data.csv)

        Returns:
            Loaded DataFrame
        """
        if filepath is None:
            filepath = str(DATA_DIR / 'synthetic_traffic_data.csv')

        if os.path.exists(filepath):
            data = pd.read_csv(filepath)
            print(f"Dataset loaded from {filepath}: {data.shape}")
            return data
        else:
            print(f"Dataset not found at {filepath}. Generating new dataset.")
            data = self.generate_dataset()
            self.save_dataset(data, filepath)
            return data


if __name__ == "__main__":
    # Test data generation
    print("Testing TrafficDataGenerator...\n")

    generator = TrafficDataGenerator()
    data = generator.generate_dataset(1000)

    print(f"\nGenerated {len(data)} samples")
    print(f"Columns: {list(data.columns)}")
    print(f"\nFirst 5 rows:")
    print(data.head())

    print(f"\nCongestion Level Statistics:")
    print(data['congestion_level'].describe())

    # Save test data
    generator.save_dataset(data)
    print("\nData generation test complete!")

