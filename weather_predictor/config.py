import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY") or os.getenv("OPENWEATHER_API_KEY")

# Model configuration for enhanced ML model
ML_MODEL_PATH = os.getenv("ML_MODEL_PATH", "models/enhanced_weather_ml_model.pkl")