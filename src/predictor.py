import os
import joblib
import pandas as pd
from src.weather_api import fetch_live_weather
from src.fire_indices import calculate_fwi_indices

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "forest_fire_rf_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "standard_scaler.pkl")

def run_live_inference(lat, lon):
    """Orchestrates the entire real-time execution pipeline."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Trained pipeline artifacts missing in /models directory.")
        
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # Execute modular pipeline stages
    weather = fetch_live_weather(lat, lon)
    indices = calculate_fwi_indices(weather)
    
    # Format inputs for model matching training structure
    input_df = pd.DataFrame([{
        'day': 23, 'month': 6,
        'Temperature': weather["temp"], 'RH': weather["rh"], 'Ws': weather["ws"], 'Rain': weather["rain"],
        'FFMC': indices["FFMC"], 'DMC': indices["DMC"], 'DC': indices["DC"], 'ISI': indices["ISI"], 'FWI': indices["FWI"],
        'Region': 1
    }])
    
    input_df = input_df[scaler.feature_names_in_]
    input_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    
    return weather, indices, prediction, probabilities