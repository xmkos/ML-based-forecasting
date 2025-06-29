from .data_collector import WeatherDataCollector
from .ml_model import WeatherMLPredictor
from .config import API_KEY
from .exceptions import DataError
from typing import Dict, Optional
from datetime import datetime

class WeatherPredictor:
    def __init__(self, model_type='rf', historical_days=30):
        if not API_KEY:
            raise DataError("API_KEY is not set in environment variables or .env file.")
        self.collector = WeatherDataCollector(API_KEY)
        self.ml_model = WeatherMLPredictor(API_KEY, model_type=model_type)
        self.historical_days = historical_days

    def get_current_weather(self, city: str) -> dict:
        return self.collector.get_current(city)

    def train_model(self, city: str = "Wroclaw", days: Optional[int] = None, validate: bool = True) -> dict:
        """Train the enhanced ML model with current and historical forecast data"""
        if days is None:
            days = self.historical_days
            
        return self.ml_model.train_model(city, days=days, validate=validate)
    
    def predict_weather(self, city: str) -> dict:
        """Predict weather using enhanced ML model with integrated additional predictions"""
        try:
            # Get current weather with better error handling
            current_weather = self.collector.get_current(city)
            
            # Check for errors in the API response
            if isinstance(current_weather, dict) and current_weather.get('error', False):
                error_msg = current_weather.get('message', 'Unknown API error')
                print(f"Error: {error_msg}")
                return {
                    'city': city,
                    'error': error_msg,
                    'prediction_date': datetime.now().strftime('%Y-%m-%d'),
                    'status': 'failed'
                }
            
            # Get forecast data with error handling
            forecast_data = self.collector.get_forecast(city)
            if isinstance(forecast_data, dict) and forecast_data.get('error', False):
                error_msg = forecast_data.get('message', 'Unknown forecast API error')
                print(f"Error: {error_msg}")
                return {
                    'city': city,
                    'error': error_msg,
                    'prediction_date': datetime.now().strftime('%Y-%m-%d'),
                    'status': 'failed'
                }
            
            if not self.ml_model.is_trained:
                # Auto-train the model if not trained
                print("ML model not trained. Training now...")
                self.train_model(city)
            
            # Use try-except to catch prediction errors
            try:
                # Get basic prediction
                prediction_result = self.ml_model.predict_weather(current_weather, forecast_data)
                
                # Debug for troubleshooting
                print("Basic prediction keys:", list(prediction_result.keys()))
                
                # Instead of returning just the basic prediction, also get detailed prediction
                try:
                    # Directly generate full predictions to include all details
                    full_prediction = self.ml_model.predict_tomorrow_weather(current_weather)
                    
                    # Debug the full prediction values explicitly
                    print(f"HUMIDITY VALUE TYPE: {type(full_prediction.get('predicted_humidity'))}")
                    print(f"HUMIDITY VALUE: {full_prediction.get('predicted_humidity')}")
                    print(f"WIND VALUE TYPE: {type(full_prediction.get('predicted_wind'))}")
                    print(f"WIND VALUE: {full_prediction.get('predicted_wind')}")
                    
                    # Check if we have working auxiliary models
                    if self.ml_model.humidity_model is None:
                        print("Warning: Humidity model is missing - will try to re-train")
                        try:
                            # Re-initialize the humidity model
                            self.ml_model._initialize_model(self.ml_model.model_type)
                            # Try to load the saved models again
                            dummy_city = "Wroclaw"  # Default city
                            self.ml_model._load_model(dummy_city if city is None else city)
                        except Exception as reinit_err:
                            print(f"Failed to re-initialize models: {reinit_err}")
                    
                    # If values are None, provide fallback based on weather conditions
                    if full_prediction.get('predicted_humidity') is None:
                        # Simple fallback based on temperature and condition
                        temp = full_prediction['predicted_temp']
                        condition = full_prediction.get('predicted_condition', 'Clear')
                        
                        # Estimate humidity based on temperature and condition
                        if condition == 'Rain':
                            humidity_fallback = 85
                        elif condition == 'Snow':
                            humidity_fallback = 75
                        elif condition == 'Clouds':
                            humidity_fallback = 65
                        else:  # Clear
                            humidity_fallback = 50 - (temp - 20)/2  # Lower humidity when hotter
                            humidity_fallback = max(30, min(70, humidity_fallback))  # Keep in reasonable range
                            
                        full_prediction['predicted_humidity'] = humidity_fallback
                        print(f"Using fallback humidity value: {humidity_fallback}")
                        
                    if full_prediction.get('predicted_wind') is None:
                        # Simple fallback for wind based on condition
                        condition = full_prediction.get('predicted_condition', 'Clear')
                        
                        # Estimate wind based on condition
                        if condition == 'Rain':
                            wind_fallback = 4.5  # m/s
                        elif condition == 'Snow':
                            wind_fallback = 3.0  # m/s 
                        elif condition == 'Clouds':
                            wind_fallback = 2.5  # m/s
                        else:  # Clear
                            wind_fallback = 2.0  # m/s
                            
                        full_prediction['predicted_wind'] = wind_fallback
                        print(f"Using fallback wind value: {wind_fallback}")
                    
                    # Merge the detailed predictions with fallbacks
                    prediction_result.update({
                        'predicted_humidity': full_prediction.get('predicted_humidity'),
                        'predicted_wind': full_prediction.get('predicted_wind'),
                        'predicted_condition': full_prediction.get('predicted_condition'),
                        'model_used': full_prediction.get('model_used'),
                        'model_insights': full_prediction.get('model_insights')
                    })
                    
                    # Print the final result keys for debugging
                    print("Final prediction result keys:", list(prediction_result.keys()))
                    
                except Exception as detail_error:
                    import traceback
                    print(f"Warning: Could not generate detailed predictions: {detail_error}")
                    print(traceback.format_exc())
                    # Fall back to making direct predictions if needed
                    if self.ml_model.humidity_model is not None and self.ml_model.is_trained:
                        try:
                            # Get current weather data and transform it for prediction
                            # This is a simplified approach for fallback
                            features = self._extract_basic_features(current_weather)
                            if features is not None:
                                humidity_pred = self.ml_model.humidity_model.predict([features])[0]
                                prediction_result['predicted_humidity'] = humidity_pred
                                
                                if self.ml_model.wind_model is not None:
                                    wind_pred = self.ml_model.wind_model.predict([features])[0]
                                    prediction_result['predicted_wind'] = wind_pred
                        except Exception as e:
                            print(f"Fallback prediction failed: {e}")
                
                return prediction_result
            except Exception as e:
                error_msg = f"Prediction failed: {str(e)}"
                print(f"Error: {error_msg}")
                return {
                    'city': city,
                    'error': error_msg,
                    'prediction_date': datetime.now().strftime('%Y-%m-%d'),
                    'status': 'failed'
                }
            
        except Exception as e:
            # Provide more helpful error message
            error_msg = f"Failed to predict weather for {city}: {str(e)}"
            print(f"Error: {error_msg}")
            return {
                'city': city,
                'error': error_msg,
                'prediction_date': datetime.now().strftime('%Y-%m-%d'),
                'status': 'failed'
            }

    def _extract_basic_features(self, weather_data: dict) -> list | None:
        """Extract basic features for fallback prediction"""
        try:
            if 'main' not in weather_data or 'wind' not in weather_data:
                return None
            
            # Simple feature extraction - just basic values
            features = [
                weather_data['main']['temp'],
                weather_data['main']['humidity'],
                weather_data['main']['pressure'],
                weather_data['wind']['speed'],
                weather_data['clouds']['all'],
                # Add time features
                datetime.now().hour,
                datetime.now().day,
                datetime.now().month,
                datetime.now().timetuple().tm_yday,
                datetime.now().weekday()
            ]
            return features
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            return None
    
    def predict_weather_with_comparison(self, city: str) -> dict:
        """Get ML prediction with API comparison"""
        try:
            # Get current weather with error handling
            current_weather = self.collector.get_current(city)
            if isinstance(current_weather, dict) and current_weather.get('error', False):
                return {
                    'city': city,
                    'error': current_weather.get('message', 'Unknown API error'),
                    'status': 'failed'
                }
            
            # Get forecast with error handling
            forecast_data = self.collector.get_forecast(city)
            if isinstance(forecast_data, dict) and forecast_data.get('error', False):
                return {
                    'city': city,
                    'error': forecast_data.get('message', 'Unknown forecast API error'),
                    'status': 'failed'
                }
            
            # Train ML model if needed
            if not self.ml_model.is_trained:
                print("ML model not trained. Training now...")
                self.train_model(city)
            
            # Get ML prediction and comparison
            try:
                comparison = self.ml_model.compare_with_api_forecast(current_weather, forecast_data)
                
                # Apply fallbacks for missing humidity and wind in the ML prediction
                # This ensures consistency between predict_weather and predict_weather_with_comparison
                if 'ml_prediction' in comparison:
                    ml_pred = comparison['ml_prediction']
                    
                    # Add fallback for humidity if missing
                    if ml_pred.get('predicted_humidity') is None:
                        # Use same fallback logic as in predict_weather
                        temp = ml_pred.get('predicted_temp')
                        condition = ml_pred.get('predicted_condition', 'Clear')
                        
                        if condition == 'Rain':
                            humidity_fallback = 85
                        elif condition == 'Snow':
                            humidity_fallback = 75
                        elif condition == 'Clouds':
                            humidity_fallback = 65
                        else:  # Clear
                            humidity_fallback = 50 - (temp - 20)/2
                            humidity_fallback = max(30, min(70, humidity_fallback))
                        
                        ml_pred['predicted_humidity'] = humidity_fallback
                        print(f"Using fallback humidity value in comparison: {humidity_fallback}")
                    
                    # Add fallback for wind if missing
                    if ml_pred.get('predicted_wind') is None:
                        condition = ml_pred.get('predicted_condition', 'Clear')
                        
                        if condition == 'Rain':
                            wind_fallback = 4.5
                        elif condition == 'Snow':
                            wind_fallback = 3.0
                        elif condition == 'Clouds':
                            wind_fallback = 2.5
                        else:  # Clear
                            wind_fallback = 2.0
                        
                        ml_pred['predicted_wind'] = wind_fallback
                        print(f"Using fallback wind value in comparison: {wind_fallback}")
                
                # Include model performance metrics in the response
                performance = self.get_model_performance()
                
                # Add model type information
                model_info = self.ml_model.MODEL_TYPES.get(self.ml_model.model_type, ('Unknown', None))[0]
                if 'ml_prediction' in comparison:
                    comparison['ml_prediction']['model_used'] = model_info
                
                return {
                    'city': city,
                    'current_weather': current_weather,
                    'prediction_comparison': comparison,
                    'model_info': self.get_model_info(),
                    'model_performance': performance
                }
            except Exception as e:
                return {
                    'city': city,
                    'error': f"Prediction error: {str(e)}",
                    'status': 'failed'
                }
            
        except Exception as e:
            return {
                'city': city,
                'error': f"Failed to generate prediction: {str(e)}",
                'status': 'failed'
            }

    def is_model_trained(self) -> bool:
        """Check if the model is trained - for backward compatibility"""
        return self.ml_model.is_trained
    
    def get_model_info(self) -> str:
        """Get detailed model information - for backward compatibility"""
        info = self.ml_model.get_model_info()
        if isinstance(info, dict):
            return (f"Enhanced ML Model - Type: {info.get('model_type', 'Unknown')}, "
                   f"Trained: {info.get('is_trained', False)}, "
                   f"Features: {info.get('feature_count', 0)}, "
                   f"Training Data: {info.get('training_samples', 0)} samples, "
                   f"Algorithm: {info.get('algorithm', 'Unknown')}")
        return str(info)

    def get_model_performance(self) -> dict:
        """Get model performance metrics with proper type conversion"""
        return self.ml_model.model_performance if hasattr(self.ml_model, 'model_performance') else {}
    
    def change_model_type(self, model_type: str) -> bool:
        """Change the model type used for prediction"""
        return self.ml_model.change_model_type(model_type)
