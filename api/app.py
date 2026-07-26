"""
Flask API Server for Traffic Congestion Prediction System.

Provides:
- Web interface for congestion prediction
- REST API endpoints for model inference
- Interactive dashboard with visualizations
- Real-time prediction capabilities
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS

from config import API_CONFIG, CONGESTION_LEVELS, WEATHER_CONDITIONS, MODELS_DIR, OUTPUT_DIR, DATA_GENERATION
from src.predict import CongestionPredictor

STATIC_DIR = str(Path(__file__).resolve().parent.parent / 'static')
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='/static')
CORS(app)
app.config['SECRET_KEY'] = API_CONFIG.get('secret_key', 'traffic-prediction-key')

# Initialize predictor
predictor = CongestionPredictor()

# Try to load models, fall back to heuristic
try:
    predictor.load_models()
    print("✓ Models loaded successfully")
except Exception as e:
    print(f"⚠ Could not load models: {e}")
    print("  Using heuristic prediction fallback")


def _convert_weather(weather_str):
    """Convert weather string to numeric code."""
    weather_map = {
        'clear': 0, 'cloudy': 1, 'rainy': 2, 'stormy': 3, 'foggy': 4, 'snowy': 5
    }
    weather_lower = weather_str.lower().strip() if weather_str else ''
    return weather_map.get(weather_lower, 0)


# ============== ROUTES ==============

@app.route('/')
def home():
    """Home page."""
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    """Interactive dashboard."""
    return render_template('dashboard.html')


@app.route('/predict')
def predict_page():
    """Prediction page."""
    return render_template('predict.html', 
                         congestion_levels=CONGESTION_LEVELS,
                         weather_conditions=WEATHER_CONDITIONS)


@app.route('/advanced')
def advanced_page():
    """Advanced prediction and analysis page."""
    return render_template('advanced.html',
                         congestion_levels=CONGESTION_LEVELS,
                         weather_conditions=WEATHER_CONDITIONS,
                         locations=DATA_GENERATION.get('locations', [
                             'Downtown', 'Highway_A', 'Highway_B', 'Residential_A',
                             'Commercial_A', 'Airport_Road', 'Ring_Road', 'Bridge_1',
                             'Bridge_2', 'Market_Area', 'Industrial_Zone', 'University_Area'
                         ]))


# ============== API ENDPOINTS ==============

@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': predictor.is_loaded,
        'models_available': list(predictor.models.keys()) if predictor.models else []
    })


@app.route('/api/models', methods=['GET'])
def list_models():
    """List available models and their performance."""
    available_models = list(predictor.models.keys()) if predictor.models else ['heuristic']
    metrics = predictor.metrics if hasattr(predictor, 'metrics') else {}
    
    return jsonify({
        'models': available_models,
        'default': 'ensemble',
        'performance': metrics,
        'total_models': len(available_models)
    })


@app.route('/api/predict', methods=['POST'])
def predict_api():
    """
    Make congestion prediction.
    
    POST JSON body:
    {
        "hour": 17,
        "day_of_week": 1,
        "month": 6,
        "is_weekend": 0,
        "temperature": 32,
        "humidity": 65,
        "precipitation": 0.2,
        "wind_speed": 12,
        "traffic_volume": 850,
        "avg_speed": 35,
        "weather_condition": "Rainy",
        "location": "Downtown",
        "model": "ensemble"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract and convert input
        input_data = {
            'hour': int(data.get('hour', 12)),
            'day_of_week': int(data.get('day_of_week', 0)),
            'month': int(data.get('month', 1)),
            'is_weekend': int(data.get('is_weekend', 0)),
            'is_rush_hour': int(data.get('is_rush_hour', 0)),
            'temperature': float(data.get('temperature', 25)),
            'humidity': float(data.get('humidity', 60)),
            'precipitation': float(data.get('precipitation', 0)),
            'wind_speed': float(data.get('wind_speed', 10)),
            'traffic_volume': int(data.get('traffic_volume', 500)),
            'avg_speed': int(data.get('avg_speed', 50)),
            'weather_condition': _convert_weather(data.get('weather_condition', 'Clear')),
            'location': data.get('location', 'Downtown')
        }
        
        # Auto-detect rush hour if not explicitly provided
        hour = input_data['hour']
        is_weekend = input_data['is_weekend']
        if 'is_rush_hour' not in data or data.get('is_rush_hour') is None:
            input_data['is_rush_hour'] = 1 if (7 <= hour <= 9 or 17 <= hour <= 19) and not is_weekend else 0
        
        # Get model choice
        model_name = data.get('model', 'ensemble')
        
        # Make prediction
        result = predictor.predict(input_data, model_name)
        
        # Add input data for reference
        result['input_data'] = input_data
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/batch', methods=['POST'])
def predict_batch_api():
    """
    Make batch predictions.
    
    POST JSON body with list of inputs.
    """
    try:
        data = request.get_json()
        
        if not data or 'inputs' not in data:
            return jsonify({'error': 'No inputs provided'}), 400
        
        inputs = data['inputs']
        model_name = data.get('model', 'ensemble')
        
        results = []
        for input_data in inputs:
            result = predictor.predict(input_data, model_name)
            results.append(result)
        
        return jsonify({
            'predictions': results,
            'count': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sample-data', methods=['GET'])
def get_sample_data():
    """Get sample traffic data for dashboard visualization."""
    n_samples = int(request.args.get('n', 500))
    
    try:
        # Try to load existing data
        from config import DATA_DIR
        data_path = str(DATA_DIR / 'synthetic_traffic_data.csv')
        
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
        else:
            # Generate sample data
            from src.data_generator import TrafficDataGenerator
            generator = TrafficDataGenerator()
            df = generator.generate_dataset(n_samples)
        
        # Sample if too large
        if len(df) > n_samples:
            df = df.sample(n=n_samples, random_state=42)
        
        # Convert to JSON
        data_json = df.head(n_samples).to_dict(orient='records')
        
        # Calculate summary stats
        stats = {
            'total_records': len(df),
            'avg_congestion': float(df['congestion_level'].mean()),
            'avg_traffic_volume': float(df['traffic_volume'].mean()),
            'avg_speed': float(df['avg_speed'].mean()),
            'congestion_distribution': df['congestion_level'].value_counts().to_dict(),
            'locations': df['location'].nunique() if 'location' in df.columns else 0
        }
        
        return jsonify({
            'data': data_json,
            'stats': stats,
            'columns': list(df.columns)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/time-analysis', methods=['GET'])
def get_time_analysis():
    """Get time-based congestion analysis."""
    try:
        from config import DATA_DIR
        data_path = str(DATA_DIR / 'synthetic_traffic_data.csv')
        
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            
            # Hourly analysis
            hourly = df.groupby('hour')['congestion_level'].mean().to_dict()
            
            # Daily analysis
            daily = df.groupby('day_of_week')['congestion_level'].mean().to_dict()
            
            # Monthly analysis
            monthly = df.groupby('month')['congestion_level'].mean().to_dict()
            
            # Rush hour comparison
            rush_hour = df[df['is_rush_hour'] == 1]['congestion_level'].mean()
            non_rush = df[df['is_rush_hour'] == 0]['congestion_level'].mean()
            
            return jsonify({
                'hourly_avg': {str(k): float(v) for k, v in hourly.items()},
                'daily_avg': {str(k): float(v) for k, v in daily.items()},
                'monthly_avg': {str(k): float(v) for k, v in monthly.items()},
                'rush_hour_avg': float(rush_hour) if not pd.isna(rush_hour) else 0,
                'non_rush_avg': float(non_rush) if not pd.isna(non_rush) else 0
            })
        else:
            return jsonify({'error': 'No data available'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/weather-analysis', methods=['GET'])
def get_weather_analysis():
    """Get weather impact analysis."""
    try:
        from config import DATA_DIR
        data_path = str(DATA_DIR / 'synthetic_traffic_data.csv')
        
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            
            weather_impact = df.groupby('weather_condition')['congestion_level'].agg(['mean', 'count']).to_dict()
            
            return jsonify({
                'weather_impact': {
                    str(k): {
                        'avg_congestion': float(weather_impact['mean'][k]),
                        'samples': int(weather_impact['count'][k])
                    }
                    for k in weather_impact['mean'].keys()
                }
            })
        else:
            return jsonify({'error': 'No data available'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============== PREMIUM API ENDPOINTS ==============

@app.route('/api/location-ranking', methods=['GET'])
def get_location_ranking():
    """Get congestion ranking by location."""
    try:
        from config import DATA_DIR
        data_path = str(DATA_DIR / 'synthetic_traffic_data.csv')
        
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            
            ranking = df.groupby('location').agg(
                avg_congestion=('congestion_level', 'mean'),
                avg_volume=('traffic_volume', 'mean'),
                avg_speed=('avg_speed', 'mean'),
                sample_count=('congestion_level', 'count')
            ).reset_index()
            
            ranking = ranking.sort_values('avg_congestion', ascending=False)
            
            rankings_list = []
            for _, row in ranking.iterrows():
                level = int(round(row['avg_congestion']))
                rankings_list.append({
                    'location': row['location'],
                    'avg_congestion': round(float(row['avg_congestion']), 3),
                    'avg_traffic_volume': int(round(row['avg_volume'])),
                    'avg_speed': int(round(row['avg_speed'])),
                    'samples': int(row['sample_count']),
                    'congestion_level': level,
                    'congestion_label': CONGESTION_LEVELS.get(level, 'Unknown')
                })
            
            return jsonify({
                'rankings': rankings_list,
                'total_locations': len(rankings_list),
                'most_congested': rankings_list[0]['location'] if rankings_list else None,
                'least_congested': rankings_list[-1]['location'] if rankings_list else None
            })
        else:
            return jsonify({'error': 'No data available'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/optimal-times', methods=['GET'])
def get_optimal_times():
    """Get optimal travel times for each location."""
    try:
        from config import DATA_DIR
        data_path = str(DATA_DIR / 'synthetic_traffic_data.csv')
        
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            
            # Find best hour per location (lowest congestion)
            recommendations = []
            for loc in df['location'].unique():
                loc_data = df[df['location'] == loc]
                hourly = loc_data.groupby('hour')['congestion_level'].mean()
                best_hour = int(hourly.idxmin())
                worst_hour = int(hourly.idxmax())
                
                best_data = loc_data[loc_data['hour'] == best_hour]
                worst_data = loc_data[loc_data['hour'] == worst_hour]
                
                recommendations.append({
                    'location': loc,
                    'best_time': f"{best_hour:02d}:00 - {(best_hour+1)%24:02d}:00",
                    'worst_time': f"{worst_hour:02d}:00 - {(worst_hour+1)%24:02d}:00",
                    'congestion_level': float(hourly.min()),
                    'max_congestion': float(hourly.max()),
                    'traffic_volume': int(best_data['traffic_volume'].mean()),
                    'avg_speed': int(best_data['avg_speed'].mean()),
                    'improvement_pct': round((1 - hourly.min() / max(hourly.max(), 0.01)) * 100, 1)
                })
            
            return jsonify({
                'recommendations': sorted(recommendations, key=lambda x: x['improvement_pct'], reverse=True),
                'total_locations': len(recommendations)
            })
        else:
            return jsonify({'error': 'No data available'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/trends', methods=['GET'])
def get_trends():
    """Get congestion trends and predictions summary."""
    try:
        from config import DATA_DIR
        data_path = str(DATA_DIR / 'synthetic_traffic_data.csv')
        
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            
            current_hour = datetime.now().hour
            current_data = df[df['hour'] == current_hour]
            next_hour_data = df[df['hour'] == (current_hour + 1) % 24]
            
            current_avg = float(current_data['congestion_level'].mean()) if len(current_data) > 0 else 0
            next_avg = float(next_hour_data['congestion_level'].mean()) if len(next_hour_data) > 0 else 0
            
            # Hourly trend for the next 6 hours
            trend = []
            for h in range(current_hour, current_hour + 6):
                h_mod = h % 24
                hour_data = df[df['hour'] == h_mod]
                avg_cong = float(hour_data['congestion_level'].mean()) if len(hour_data) > 0 else 0
                trend.append({
                    'hour': h_mod,
                    'predicted_congestion': round(avg_cong, 3),
                    'label': CONGESTION_LEVELS.get(int(round(avg_cong)), 'Unknown')
                })
            
            return jsonify({
                'current_prediction': {
                    'hour': current_hour,
                    'avg_congestion': round(current_avg, 3),
                    'label': CONGESTION_LEVELS.get(int(round(current_avg)), 'Unknown')
                },
                'next_hour_prediction': {
                    'hour': (current_hour + 1) % 24,
                    'avg_congestion': round(next_avg, 3),
                    'label': CONGESTION_LEVELS.get(int(round(next_avg)), 'Unknown')
                },
                'trend': trend,
                'trend_direction': 'increasing' if next_avg > current_avg else 'decreasing' if next_avg < current_avg else 'stable'
            })
        else:
            return jsonify({'error': 'No data available'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Serve static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(str(Path(__file__).resolve().parent.parent / 'static'), filename)


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    host = API_CONFIG.get('host', '0.0.0.0')
    port = API_CONFIG.get('port', 5000)
    debug = API_CONFIG.get('debug', True)
    
    print(f"\n{'='*60}")
    print("TRAFFIC CONGESTION PREDICTION SYSTEM")
    print(f"{'='*60}")
    print(f"Web Interface: http://localhost:{port}")
    print(f"API Health:    http://localhost:{port}/api/health")
    print(f"Dashboard:     http://localhost:{port}/dashboard")
    print(f"{'='*60}\n")
    
    app.run(host=host, port=port, debug=debug)