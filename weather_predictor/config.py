# Loads API keys and model configuration
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY") or os.getenv("OPENWEATHER_API_KEY")

ML_MODEL_PATH = os.getenv("ML_MODEL_PATH", "models/enhanced_weather_ml_model.pkl")
