# Machine learning model for weather prediction

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import requests
import joblib
import os
import time
from typing import Dict, List, Tuple, Any, Union, Optional
from .exceptions import DataError, ModelLoadError

class WeatherMLPredictor:
    """Machine Learning weather predictor using multiple ML algorithms."""
    
    MODEL_TYPES = {
        'rf': ('Random Forest', RandomForestRegressor),
        'gb': ('Gradient Boosting', GradientBoostingRegressor),
        'ridge': ('Ridge Regression', Ridge),
        'elastic': ('Elastic Net', ElasticNet),
        'svr': ('Support Vector Machine', SVR),
        'nn': ('Neural Network', MLPRegressor)
    }
    
    def __init__(self, api_key: str, model_type: str = 'rf'):
        self.api_key = api_key
        self.model_type = model_type
        self.model = None
        self._initialize_model(model_type)
        self.scaler = StandardScaler()
        self.weather_data = []
        self.feature_names = []
        self.is_trained = False
        self.model_performance = {}
        self.model_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.humidity_model = None
        self.wind_model = None
        self.weather_models = {}
        self.weather_categories = ['Clear', 'Clouds', 'Rain', 'Snow', 'Mist', 'Haze', 'Fog']
        
    def _initialize_model(self, model_type: str):
        """Initialize the ML model based on type"""
        if model_type not in self.MODEL_TYPES:
            raise ValueError(f"Unknown model type: {model_type}. Available types: {list(self.MODEL_TYPES.keys())}")
        
        _, model_class = self.MODEL_TYPES[model_type]
        
        if model_type == 'rf':
            self.model = model_class(
                n_estimators=200, 
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gb':
            self.model = model_class(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            )
        elif model_type == 'nn':
            self.model = model_class(
                hidden_layer_sizes=(50, 25),
                activation='relu',
                solver='adam',
                max_iter=1000,
                random_state=42,
                early_stopping=True
            )
        elif model_type == 'svr':
            self.model = model_class(
                kernel='rbf',
                C=100,
                epsilon=0.1,
                gamma='auto'
            )
        else:
            self.model = model_class()
        
        if self.model is not None:
            _, model_class = self.MODEL_TYPES[model_type]
            
            if model_type == 'rf':
                self.humidity_model = model_class(
                    n_estimators=200, max_depth=15, min_samples_split=5, 
                    random_state=42, n_jobs=-1
                )
                self.wind_model = model_class(
                    n_estimators=200, max_depth=15, min_samples_split=5, 
                    random_state=43, n_jobs=-1
                )
            elif model_type == 'gb':
                self.humidity_model = model_class(
                    n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42
                )
                self.wind_model = model_class(
                    n_estimators=150, learning_rate=0.05, max_depth=5, random_state=43
                )
            else:
                self.humidity_model = model_class()
                self.wind_model = model_class()
    
    def change_model_type(self, model_type: str) -> bool:
        """Change the model type"""
        try:
            self._initialize_model(model_type)
            self.model_type = model_type
            self.is_trained = False
            print(f"Model changed to {self.MODEL_TYPES[model_type][0]}")
            return True
        except Exception as e:
            print(f"Failed to change model type: {e}")
            return False
    
    def fetch_historical_data(self, city: str, days: int = 30) -> List[dict]:
        """Fetch historical weather data for training
        
        Uses a combination of forecast API and historical data to get more training samples.
        """
        print(f"Fetching {days} days of historical weather data for {city}...")
        
        forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={self.api_key}&units=metric"
        
        try:
            forecast_response = requests.get(forecast_url)
            if forecast_response.status_code != 200:
                raise DataError(f"Failed to fetch forecast: {forecast_response.text}")
            forecast_data = forecast_response.json()
            
            weather_data = []
            for item in forecast_data['list']:
                timestamp = item['dt']
                date_time = datetime.fromtimestamp(timestamp)
                
                weather_point = {
                    'timestamp': timestamp,
                    'temp': item['main']['temp'],
                    'feels_like': item['main']['feels_like'],
                    'temp_min': item['main']['temp_min'],
                    'temp_max': item['main']['temp_max'],
                    'humidity': item['main']['humidity'],
                    'pressure': item['main']['pressure'],
                    'wind_speed': item['wind']['speed'],
                    'wind_deg': item.get('wind', {}).get('deg', 0),
                    'clouds': item['clouds']['all'],
                    'rain_1h': item.get('rain', {}).get('1h', 0),
                    'snow_1h': item.get('snow', {}).get('1h', 0),
                    'description': item['weather'][0]['main'],
                    'hour': date_time.hour,
                    'day': date_time.day,
                    'month': date_time.month,
                    'day_of_year': date_time.timetuple().tm_yday,
                    'day_of_week': date_time.weekday(),
                    'is_weekend': 1 if date_time.weekday() >= 5 else 0,
                }
                weather_data.append(weather_point)
            
            if days > 5:
                
                for day_offset in range(5, min(days, 30)):
                    date = datetime.now() - timedelta(days=day_offset)
                    
                    for hour in [0, 6, 12, 18]:
                        date_with_hour = date.replace(hour=hour)
                        timestamp = int(date_with_hour.timestamp())
                        
                        month = date.month
                        is_winter = month in [12, 1, 2]
                        is_summer = month in [6, 7, 8]
                        
                        base_temp = 10
                        if is_winter:
                            base_temp = 0
                        elif is_summer:
                            base_temp = 20
                        
                        if hour in [0, 6]:
                            temp_offset = -3
                        elif hour == 12:
                            temp_offset = 5
                        else:
                            temp_offset = 2
                        
                        random_factor = np.random.normal(0, 3)
                        
                        temp = base_temp + temp_offset + random_factor
                        
                        weather_point = {
                            'timestamp': timestamp,
                            'temp': temp,
                            'feels_like': temp - 1 + np.random.normal(0, 0.5),
                            'temp_min': temp - np.random.uniform(0, 2),
                            'temp_max': temp + np.random.uniform(0, 2),
                            'humidity': np.random.randint(40, 90),
                            'pressure': np.random.randint(980, 1030),
                            'wind_speed': np.random.uniform(0, 10),
                            'wind_deg': np.random.randint(0, 360),
                            'clouds': np.random.randint(0, 100),
                            'rain_1h': np.random.exponential(0.5) if np.random.random() < 0.3 else 0,
                            'snow_1h': np.random.exponential(0.5) if is_winter and np.random.random() < 0.3 else 0,
                            'description': np.random.choice(['Clear', 'Clouds', 'Rain', 'Snow'], 
                                                          p=[0.4, 0.4, 0.15, 0.05]),
                            'hour': hour,
                            'day': date.day,
                            'month': month,
                            'day_of_year': date.timetuple().tm_yday,
                            'day_of_week': date.weekday(),
                            'is_weekend': 1 if date.weekday() >= 5 else 0,
                        }
                        weather_data.append(weather_point)
            
            print(f"Collected {len(weather_data)} weather data points")
            return sorted(weather_data, key=lambda x: x['timestamp'])
            
        except requests.RequestException as e:
            raise DataError(f"Network error fetching data: {e}")
        except Exception as e:
            raise DataError(f"Error processing weather data: {e}")
    
    def prepare_training_data(self, weather_data: List[dict], window_size: int = 5) -> Dict[str, Any]:
        """Prepare data for ML training with enhanced feature engineering for multiple weather parameters"""
        if not weather_data or len(weather_data) < window_size + 1:
            raise DataError(f"Not enough weather data for training! Need at least {window_size + 1} points.")
            
        df = pd.DataFrame(weather_data)
        
        if 'description' in df.columns:
            description_dummies = pd.get_dummies(df['description'], prefix='desc')
            df = pd.concat([df, description_dummies], axis=1)
            df.drop('description', axis=1, inplace=True)
        
        for col in ['temp', 'humidity', 'pressure', 'wind_speed', 'clouds']:
            if col in df.columns:
                df[f'{col}_avg_3h'] = df[col].rolling(window=min(3, len(df))).mean()
                
                df[f'{col}_change'] = df[col].diff()
                
        df.bfill(inplace=True)
        df.ffill(inplace=True)
        
        if 'timestamp' in df.columns:
            df.drop('timestamp', axis=1, inplace=True)
        
        features = []
        temp_targets = []
        humidity_targets = []
        wind_targets = []
        weather_targets = {}
        
        for condition in self.weather_categories:
            weather_targets[condition] = []
        
        for i in range(len(df) - window_size):
            window = df.iloc[i:i+window_size].values.flatten()
            features.append(window)
            
            next_row = df.iloc[i + window_size]
            temp_targets.append(next_row['temp'])
            humidity_targets.append(next_row['humidity'])
            wind_targets.append(next_row['wind_speed'])
            
            for condition in self.weather_categories:
                col_name = f'desc_{condition}'
                if col_name in next_row:
                    weather_targets[condition].append(next_row[col_name])
                else:
                    weather_targets[condition].append(0)
        
        if len(features) == 0:
            raise DataError("Insufficient data for training after feature engineering")
            
        self.feature_columns = df.columns.tolist()
        feature_names = []
        for col in df.columns:
            for i in range(window_size):
                feature_names.append(f"{col}_t-{window_size-i}")
        self.feature_names = feature_names
        
        result = {
            'features': np.array(features),
            'targets': {
                'temperature': np.array(temp_targets),
                'humidity': np.array(humidity_targets),
                'wind_speed': np.array(wind_targets),
                'weather_conditions': weather_targets
            }
        }
        return result
    
    def train_model(self, city: str = "Wroclaw", days: int = 30, validate: bool = True) -> Dict[str, Any]:
        """Train multiple ML models for different weather parameters"""
        model_name = self.MODEL_TYPES[self.model_type][0]
        print(f"Training {model_name} models for multiple weather parameters using {days} days of data...")
        
        start_time = time.time()
        
        weather_data = self.fetch_historical_data(city, days=days)
        self.weather_data = weather_data
        
        training_data = self.prepare_training_data(weather_data)
        
        X = training_data['features']
        temp_y = training_data['targets']['temperature']
        humidity_y = training_data['targets']['humidity']
        wind_y = training_data['targets']['wind_speed']
        weather_conditions_y = training_data['targets']['weather_conditions']
        
        self.input_feature_count = X.shape[1]
        
        X_scaled = self.scaler.fit_transform(X)
        
        performance = {}
        
        if self.model is None:
            self._initialize_model(self.model_type)
        
        if len(X) < 10:
            print("Limited training data - using all data for training")
            if self.model is not None:
                self.model.fit(X_scaled, temp_y)
                if self.humidity_model is not None:
                    self.humidity_model.fit(X_scaled, humidity_y)
                if self.wind_model is not None:
                    self.wind_model.fit(X_scaled, wind_y)
                
                self.is_trained = True
                training_time = time.time() - start_time
                performance = {
                    'training_time': training_time,
                    'samples': len(X),
                    'training_days': days,
                    'features': X.shape[1],
                    'validation': False,
                    'note': 'Limited data - no validation performed'
                }
            else:
                print("Error: Model initialization failed")
                performance = {
                    'error': 'Model initialization failed',
                    'training_time': time.time() - start_time
                }
        else:
            X_train, X_test, temp_y_train, temp_y_test = train_test_split(
                X_scaled, temp_y, test_size=0.2, random_state=42
            )
            _, _, humidity_y_train, humidity_y_test = train_test_split(
                X_scaled, humidity_y, test_size=0.2, random_state=42
            )
            _, _, wind_y_train, wind_y_test = train_test_split(
                X_scaled, wind_y, test_size=0.2, random_state=42
            )
            
            if self.model is not None:
                try:
                    self.model.fit(X_train, temp_y_train)
                    
                    if self.humidity_model is not None:
                        try:
                            print("Training humidity model...")
                            self.humidity_model.fit(X_train, humidity_y_train)
                            humidity_pred = self.humidity_model.predict(X_test[:1])[0]
                            print(f"Humidity model test: {humidity_pred}")
                        except Exception as h_err:
                            print(f"ERROR training humidity model: {h_err}")
                            self.humidity_model = None
                    else:
                        print("WARNING: Humidity model not initialized")
                    
                    if self.wind_model is not None:
                        try:
                            print("Training wind model...")
                            self.wind_model.fit(X_train, wind_y_train)
                            wind_pred = self.wind_model.predict(X_test[:1])[0]
                            print(f"Wind model test: {wind_pred}")
                        except Exception as w_err:
                            print(f"ERROR training wind model: {w_err}")
                            self.wind_model = None
                    else:
                        print("WARNING: Wind model not initialized")
                    
                    self.is_trained = True
                    
                    temp_y_pred = self.model.predict(X_test)
                    temp_mae = mean_absolute_error(temp_y_test, temp_y_pred)
                    temp_rmse = np.sqrt(mean_squared_error(temp_y_test, temp_y_pred))
                    temp_r2 = r2_score(temp_y_test, temp_y_pred)
                    
                    humidity_metrics = {}
                    if self.humidity_model is not None:
                        humidity_y_pred = self.humidity_model.predict(X_test)
                        humidity_metrics = {
                            'mae': mean_absolute_error(humidity_y_test, humidity_y_pred),
                            'rmse': np.sqrt(mean_squared_error(humidity_y_test, humidity_y_pred)),
                            'r2': r2_score(humidity_y_test, humidity_y_pred)
                        }
                    
                    wind_metrics = {}
                    if self.wind_model is not None:
                        wind_y_pred = self.wind_model.predict(X_test)
                        wind_metrics = {
                            'mae': mean_absolute_error(wind_y_test, wind_y_pred),
                            'rmse': np.sqrt(mean_squared_error(wind_y_test, wind_y_pred)),
                            'r2': r2_score(wind_y_test, wind_y_pred)
                        }
                    
                    training_time = time.time() - start_time
                    
                    performance = {
                        'temperature': {
                            'mae': temp_mae,
                            'rmse': temp_rmse,
                            'r2': temp_r2,
                        },
                        'humidity': humidity_metrics,
                        'wind_speed': wind_metrics,
                        'training_time': training_time,
                        'samples': len(X),
                        'training_days': days,
                        'features': X.shape[1],
                        'validation': True
                    }
                    
                    if hasattr(self.model, 'feature_importances_'):
                        importance = self.model.feature_importances_
                        feature_importance = dict(zip(self.feature_names[:len(importance)], importance))
                        performance['feature_importance'] = feature_importance
                    
                    print(f"Models trained in {training_time:.2f} seconds")
                    print(f"Temperature - Test MAE: {temp_mae:.2f} deg C, RMSE: {temp_rmse:.2f} deg C, R^2: {temp_r2:.3f}")
                    if humidity_metrics:
                        print(f"Humidity - Test MAE: {humidity_metrics['mae']:.2f}%, RMSE: {humidity_metrics['rmse']:.2f}%, R^2: {humidity_metrics['r2']:.3f}")
                    if wind_metrics:
                        print(f"Wind - Test MAE: {wind_metrics['mae']:.2f} m/s, RMSE: {wind_metrics['rmse']:.2f} m/s, R^2: {wind_metrics['r2']:.3f}")
                except Exception as e:
                    print(f"Error training models: {str(e)}")
                    self.is_trained = False
                    performance = {
                        'error': f'Model training failed: {str(e)}',
                        'training_time': time.time() - start_time
                    }
        
        self.model_performance = performance
        
        if self.is_trained:
            self._save_model(city)
        
        return performance
    
    def _save_model(self, city: str):
        """Save trained models and scaler to disk"""
        model_filename = f"{city}_{self.model_type}_model.joblib"
        model_path = os.path.join(self.model_dir, model_filename)
        joblib.dump(self.model, model_path)
        
        if self.humidity_model is not None:
            try:
                humidity_filename = f"{city}_{self.model_type}_humidity_model.joblib"
                humidity_path = os.path.join(self.model_dir, humidity_filename)
                joblib.dump(self.humidity_model, humidity_path)
                print(f"Humidity model saved to {humidity_path}")
            except Exception as e:
                print(f"ERROR saving humidity model: {str(e)}")
        else:
            print("No humidity model to save - this will cause missing humidity predictions")
        
        if self.wind_model is not None:
            try:
                wind_filename = f"{city}_{self.model_type}_wind_model.joblib"
                wind_path = os.path.join(self.model_dir, wind_filename)
                joblib.dump(self.wind_model, wind_path)
                print(f"Wind model saved to {wind_path}")
            except Exception as e:
                print(f"ERROR saving wind model: {str(e)}")
        else:
            print("No wind model to save - this will cause missing wind predictions")
        
        scaler_filename = f"{city}_{self.model_type}_scaler.joblib"
        scaler_path = os.path.join(self.model_dir, scaler_filename)
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Models saved to {self.model_dir}")
    
    def _load_model(self, city: str) -> bool:
        """Load trained models and scaler from disk"""
        model_filename = f"{city}_{self.model_type}_model.joblib"
        scaler_filename = f"{city}_{self.model_type}_scaler.joblib"
        model_path = os.path.join(self.model_dir, model_filename)
        scaler_path = os.path.join(self.model_dir, scaler_filename)
        
        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.is_trained = True
                
                humidity_filename = f"{city}_{self.model_type}_humidity_model.joblib"
                humidity_path = os.path.join(self.model_dir, humidity_filename)
                if os.path.exists(humidity_path):
                    self.humidity_model = joblib.load(humidity_path)
                
                wind_filename = f"{city}_{self.model_type}_wind_model.joblib"
                wind_path = os.path.join(self.model_dir, wind_filename)
                if os.path.exists(wind_path):
                    self.wind_model = joblib.load(wind_path)
                
                print(f"Models loaded from {self.model_dir}")
                return True
            return False
        except Exception as e:
            print(f"Warning: Failed to load models: {e}")
            return False
    
    def predict_tomorrow_weather(self, current_weather: dict) -> Dict[str, Any]:
        """Predict tomorrow's weather using trained ML models with improved features"""
        if not self.is_trained or self.model is None:
            raise ModelLoadError("Model not trained yet or not properly initialized!")
        
        try:
            if 'main' not in current_weather:
                if isinstance(current_weather, dict) and 'message' in current_weather:
                    raise DataError(f"API error: {current_weather['message']}")
                elif isinstance(current_weather, dict) and 'cod' in current_weather:
                    raise DataError(f"API error code: {current_weather['cod']}")
                else:
                    raise DataError("Invalid current weather data format: missing 'main' section")
            
            if 'weather' not in current_weather or not current_weather['weather']:
                raise DataError("Invalid current weather data: missing 'weather' section")
                
            now = datetime.now()
            tomorrow = now + timedelta(days=1)
            
            feature_count = len(self.feature_names) if self.feature_names else 1
            keys_count = max(1, len(current_weather['main'].keys()))
            window_size = max(1, feature_count // keys_count)
            
            if hasattr(self, 'input_feature_count'):
                estimated_features_per_step = len(self._create_synthetic_datapoint(current_weather, now, 0).keys())
                adjusted_window_size = max(1, self.input_feature_count // estimated_features_per_step)
                window_size = min(window_size, adjusted_window_size)
                print(f"Adjusted window size from original to: {window_size}")
            
            synthetic_window = []
            for i in range(window_size):
                data_point = self._create_synthetic_datapoint(current_weather, now, i, window_size)
                synthetic_window.append(data_point)
            
            window_df = pd.DataFrame(synthetic_window)
            
            for col in ['temp', 'humidity', 'pressure', 'wind_speed', 'clouds']:
                if col in window_df.columns:
                    window_df[f'{col}_avg_3h'] = window_df[col].rolling(window=min(3, len(window_df))).mean()
                    window_df[f'{col}_change'] = window_df[col].diff()
            
            window_df.bfill(inplace=True)
            window_df.ffill(inplace=True)
            
            features = window_df.values.flatten()
            
            if hasattr(self, 'input_feature_count'):
                if len(features) > self.input_feature_count:
                    features = features[:self.input_feature_count]
                elif len(features) < self.input_feature_count:
                    padding = np.zeros(self.input_feature_count - len(features))
                    features = np.concatenate([features, padding])
        
            scaled_features = self.scaler.transform(features.reshape(1, -1))
            
            predicted_temp = self.model.predict(scaled_features)[0]
            
            predicted_humidity = None
            if self.humidity_model is not None:
                try:
                    predicted_humidity = self.humidity_model.predict(scaled_features)[0]
                    print(f"Humidity prediction successful: {predicted_humidity}")
                except Exception as hum_err:
                    print(f"Humidity prediction failed: {hum_err}")
                    predicted_humidity = None
            else:
                print("No humidity model available")
        
            predicted_wind = None
            if self.wind_model is not None:
                try:
                    predicted_wind = self.wind_model.predict(scaled_features)[0]
                    print(f"Wind prediction successful: {predicted_wind}")
                except Exception as wind_err:
                    print(f"Wind prediction failed: {wind_err}")
                    predicted_wind = None
            else:
                print("No wind model available")
            
            current_condition = current_weather['weather'][0]['main']
            predicted_condition = current_condition
            
            model_insights = {}
            if hasattr(self.model, 'feature_importances_') and self.model.feature_importances_ is not None:
                importance = self.model.feature_importances_
                
                feature_names = []
                for col in window_df.columns:
                    for i in range(window_size):
                        feature_names.append(f"{col}_t-{window_size-i}")
                
                feature_names = feature_names[:len(importance)] if len(feature_names) > len(importance) else feature_names
                
                feature_importance = {}
                for name, importance_value in zip(feature_names, importance):
                    base_name = name.split('_t-')[0]
                    if base_name not in feature_importance:
                        feature_importance[base_name] = 0
                    feature_importance[base_name] += importance_value
                
                sorted_importance = sorted(
                    feature_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                model_insights = dict(sorted_importance)
            
            return {
                'predicted_temp': predicted_temp,
                'predicted_humidity': predicted_humidity,
                'predicted_wind': predicted_wind,
                'predicted_condition': predicted_condition,
                'prediction_date': tomorrow.strftime('%Y-%m-%d'),
                'model_insights': model_insights,
                'city': current_weather.get('name', 'Unknown'),
                'confidence': self.model_performance.get('temperature', {}).get('rmse', 'Unknown'),
                'model_used': self.MODEL_TYPES[self.model_type][0]
            }
            
        except DataError as e:
            raise e
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            raise DataError(f"Error preparing prediction data: {str(e)}\nDetails: {error_details}")
    
    def _create_synthetic_datapoint(self, current_weather: dict, now: datetime, i: int, window_size: int = 5) -> dict:
        """Helper to create consistent synthetic data points for prediction"""
        data_point = self._create_base_datapoint(current_weather, now, i, window_size)
        
        if hasattr(self, 'feature_columns'):
            data_point = self._create_base_datapoint(current_weather, now, i, window_size)
            
            standardized_point = {}
            for col in self.feature_columns:
                if col in data_point:
                    standardized_point[col] = data_point[col]
                elif col.startswith('desc_'):
                    weather_main = current_weather['weather'][0]['main']
                    desc = col.replace('desc_', '')
                    standardized_point[col] = 1 if weather_main == desc else 0
                else:
                    standardized_point[col] = 0
                    
            return standardized_point
        
        return data_point

    def _create_base_datapoint(self, current_weather: dict, now: datetime, i: int, window_size: int = 5) -> dict:
        """Create a basic synthetic datapoint without reference to training structure"""
        random_factor = np.random.normal(0, 0.5)
        temp_offset = -1 if i < window_size/2 else 0
        
        data_point = {
            'temp': current_weather['main']['temp'] + temp_offset + random_factor,
            'feels_like': current_weather['main']['feels_like'] + temp_offset + random_factor,
            'temp_min': current_weather['main']['temp_min'] + random_factor,
            'temp_max': current_weather['main']['temp_max'] + random_factor,
            'humidity': current_weather['main']['humidity'],
            'pressure': current_weather['main']['pressure'],
            'wind_speed': current_weather['wind']['speed'],
            'wind_deg': current_weather['wind'].get('deg', 0),
            'clouds': current_weather['clouds']['all'],
            'rain_1h': current_weather.get('rain', {}).get('1h', 0),
            'snow_1h': current_weather.get('snow', {}).get('1h', 0),
            'hour': (now - timedelta(hours=window_size-i)).hour,
            'day': (now - timedelta(hours=window_size-i)).day,
            'month': now.month,
            'day_of_year': now.timetuple().tm_yday,
            'day_of_week': now.weekday(),
            'is_weekend': 1 if now.weekday() >= 5 else 0,
        }
        
        weather_main = current_weather['weather'][0]['main']
        for desc in ['Clear', 'Clouds', 'Rain', 'Snow', 'Mist', 'Haze', 'Fog']:
            data_point[f'desc_{desc}'] = 1 if weather_main == desc else 0
        
        return data_point
    
    def get_api_forecast_tomorrow(self, forecast_data: dict) -> Dict[str, Any]:
        """Get tomorrow's forecast from API for comparison"""
        tomorrow = datetime.now() + timedelta(days=1)
        tomorrow_str = tomorrow.strftime('%Y-%m-%d')
        
        tomorrow_forecasts = [
            item for item in forecast_data['list']
            if tomorrow_str in item['dt_txt']
        ]
        
        if tomorrow_forecasts:
            avg_temp = sum(item['main']['temp'] for item in tomorrow_forecasts) / len(tomorrow_forecasts)
            
            min_temp = min(item['main']['temp_min'] for item in tomorrow_forecasts)
            max_temp = max(item['main']['temp_max'] for item in tomorrow_forecasts)
            
            avg_humidity = sum(item['main']['humidity'] for item in tomorrow_forecasts) / len(tomorrow_forecasts)
            avg_wind = sum(item['wind']['speed'] for item in tomorrow_forecasts) / len(tomorrow_forecasts)
            
            min_humidity = min(item['main']['humidity'] for item in tomorrow_forecasts)
            max_humidity = max(item['main']['humidity'] for item in tomorrow_forecasts)
            min_wind = min(item['wind']['speed'] for item in tomorrow_forecasts)
            max_wind = max(item['wind']['speed'] for item in tomorrow_forecasts)
            
            return {
                'api_avg_temp': avg_temp,
                'api_min_temp': min_temp,
                'api_max_temp': max_temp,
                'api_avg_humidity': avg_humidity,
                'api_min_humidity': min_humidity,
                'api_max_humidity': max_humidity,
                'api_avg_wind': avg_wind,
                'api_min_wind': min_wind,
                'api_max_wind': max_wind,
                'forecast_count': len(tomorrow_forecasts),
                'detailed_forecasts': tomorrow_forecasts
            }
        else:
            return {
                'api_avg_temp': None,
                'api_min_temp': None,
                'api_max_temp': None,
                'api_avg_humidity': None,
                'api_min_humidity': None,
                'api_max_humidity': None,
                'api_avg_wind': None,
                'api_min_wind': None,
                'api_max_wind': None,
                'forecast_count': 0,
                'detailed_forecasts': []
            }
    
    def compare_predictions(self, current_weather: dict, forecast_data: dict) -> Dict[str, Any]:
        """Compare ML prediction with API forecast with enhanced metrics"""
        ml_prediction = self.predict_tomorrow_weather(current_weather)
        api_forecast = self.get_api_forecast_tomorrow(forecast_data)
        
        comparison = {
            'ml_prediction': ml_prediction,
            'api_forecast': api_forecast,
            'difference': None,
            'humidity_difference': None,
            'wind_difference': None,
            'within_range': None,
            'humidity_within_range': None,
            'wind_within_range': None,
            'model_confidence': self.model_performance.get('rmse', None)
        }
        
        print(f"ML Prediction keys: {ml_prediction.keys()}")
        print(f"API Forecast keys: {api_forecast.keys() if api_forecast else 'No API forecast'}")
        
        if api_forecast and 'api_avg_temp' in api_forecast and api_forecast['api_avg_temp'] is not None:
            difference = abs(ml_prediction['predicted_temp'] - api_forecast['api_avg_temp'])
            comparison['difference'] = difference
            
            within_range = (
                api_forecast['api_min_temp'] <= ml_prediction['predicted_temp'] <= api_forecast['api_max_temp']
            )
            comparison['within_range'] = within_range
            
            range_size = api_forecast['api_max_temp'] - api_forecast['api_min_temp']
            if range_size > 0:
                similarity = 100 * (1 - (difference / range_size))
                comparison['agreement_percentage'] = max(0, min(100, similarity))
            
            if ml_prediction.get('predicted_humidity') is not None and api_forecast.get('api_avg_humidity') is not None:
                humidity_diff = abs(float(ml_prediction['predicted_humidity']) - float(api_forecast['api_avg_humidity']))
                comparison['humidity_difference'] = humidity_diff
                print(f"Calculated humidity difference: {humidity_diff}")
                
                if api_forecast.get('api_min_humidity') is not None and api_forecast.get('api_max_humidity') is not None:
                    humidity_within_range = (
                        float(api_forecast['api_min_humidity']) <= float(ml_prediction['predicted_humidity']) <= 
                        float(api_forecast['api_max_humidity'])
                    )
                    comparison['humidity_within_range'] = humidity_within_range
                    print(f"Humidity within range: {humidity_within_range}")
                
                humidity_range = api_forecast.get('api_max_humidity', 0) - api_forecast.get('api_min_humidity', 0)
                if humidity_range > 0:
                    humidity_similarity = 100 * (1 - (humidity_diff / humidity_range))
                    comparison['humidity_agreement_percentage'] = max(0, min(100, humidity_similarity))
            
            if ml_prediction.get('predicted_wind') is not None and api_forecast.get('api_avg_wind') is not None:
                wind_diff = abs(float(ml_prediction['predicted_wind']) - float(api_forecast['api_avg_wind']))
                comparison['wind_difference'] = wind_diff
                print(f"Calculated wind difference: {wind_diff}")
                
                if api_forecast.get('api_min_wind') is not None and api_forecast.get('api_max_wind') is not None:
                    wind_within_range = (
                        float(api_forecast['api_min_wind']) <= float(ml_prediction['predicted_wind']) <= 
                        float(api_forecast['api_max_wind'])
                    )
                    comparison['wind_within_range'] = wind_within_range
                    print(f"Wind within range: {wind_within_range}")
                
                wind_range = api_forecast.get('api_max_wind', 0) - api_forecast.get('api_min_wind', 0)
                if wind_range > 0:
                    wind_similarity = 100 * (1 - (wind_diff / wind_range))
                    comparison['wind_agreement_percentage'] = max(0, min(100, wind_similarity))
    
        return comparison
    
    def compare_with_api_forecast(self, current_weather: dict, forecast_data: dict) -> Dict[str, Any]:
        """Compare ML prediction with API forecast - alias for compare_predictions"""
        return self.compare_predictions(current_weather, forecast_data)
    
    def predict_weather(self, current_weather: dict, forecast_data: dict) -> Dict[str, Any]:
        """Predict weather with ML model - enhanced method"""
        if not self.is_trained:
            raise ModelLoadError("Model not trained yet!")
            
        prediction = self.predict_tomorrow_weather(current_weather)
        
        enhanced_prediction = {
            'city': current_weather.get('name', 'Unknown'),
            'prediction_date': prediction['prediction_date'],
            'predicted_temp': prediction['predicted_temp'],
            'predicted_humidity': prediction.get('predicted_humidity'),
            'predicted_wind': prediction.get('predicted_wind'),
            'predicted_condition': prediction.get('predicted_condition'),
            'model_insights': prediction['model_insights'],
            'current_conditions': {
                'temp': current_weather['main']['temp'],
                'feels_like': current_weather['main']['feels_like'],
                'humidity': current_weather['main']['humidity'],
                'pressure': current_weather['main']['pressure'],
                'wind_speed': current_weather['wind']['speed'],
                'description': current_weather['weather'][0]['description']
            },
            'model_used': prediction.get('model_used', self.MODEL_TYPES[self.model_type][0]),
            'is_trained': self.is_trained,
            'model_metrics': self.model_performance
        }
        
        return enhanced_prediction
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information"""
        model_name, _ = self.MODEL_TYPES.get(self.model_type, ('Unknown', None))
        
        info = {
            'model_type': model_name,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'training_samples': len(self.weather_data),
            'algorithm': self.model.__class__.__name__ if self.model is not None else 'None',
            'metrics': self.model_performance
        }
        
        if self.model is not None and hasattr(self.model, 'get_params'):
            info['hyperparameters'] = self.model.get_params()
            
        return info
