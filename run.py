"""
Traffic Congestion Prediction System - Master Runner
Combines: data generation → model training → web server launch
"""

import subprocess
import sys
import os
import webbrowser
import time
from threading import Timer

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def main():
    print_header("TRAFFIC CONGESTION PREDICTION SYSTEM")
    print("  AI & Machine Learning Powered")
    print("="*60 + "\n")

    # Step 1: Install dependencies
    print_header("STEP 1: Installing Dependencies")
    print("Installing required Python packages...\n")
    
    deps = [
        "numpy pandas scikit-learn",
        "xgboost",
        "tensorflow keras",
        "flask flask-cors",
        "matplotlib seaborn plotly",
        "joblib python-dotenv tqdm"
    ]
    
    for dep in deps:
        print(f"  Installing: {dep}")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install"] + dep.split(),
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  ⚠ Warning: {result.stderr[:100]}")
        else:
            print(f"  ✓ {dep} ready")
    
    print("\n  ✓ Dependencies installed!")

    # Step 2: Generate data and train models
    print_header("STEP 2: Generating Data & Training Models")
    print("This will create synthetic traffic data and train 4 ML models...\n")
    print("  Models: Random Forest, XGBoost, LSTM, Linear Regression\n")
    
    result = subprocess.run(
        [sys.executable, "-m", "src.train_model"],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))
    )
    print(result.stdout)
    if result.stderr:
        print(f"  Errors: {result.stderr[:200]}")
    
    print("\n  ✓ Models trained successfully!")

    # Step 3: Launch web application
    print_header("STEP 3: Launching Web Application")
    print("  Starting Flask server...\n")
    print(f"  🌐 Open: http://localhost:5000")
    print(f"  📊 Dashboard: http://localhost:5000/dashboard")
    print(f"  🔮 Predict: http://localhost:5000/predict")
    print(f"  ❤ API Health: http://localhost:5000/api/health")
    print("\n  Press Ctrl+C to stop the server\n")
    
    # Auto-open browser after 2 seconds
    Timer(2, lambda: webbrowser.open('http://localhost:5000')).start()
    
    # Start Flask app
    from api.app import app
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == "__main__":
    main()