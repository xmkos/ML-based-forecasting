# GUI interface for the weather prediction system

import tkinter as tk
from tkinter import messagebox, ttk, scrolledtext
import threading
from .predictor import WeatherPredictor

class WeatherGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(" Weather Predictor - ML Edition")
        self.geometry("800x600")
        self.predictor = WeatherPredictor()
        
        # Create UI components
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Input section
        input_frame = ttk.LabelFrame(main_frame, text="Location Input")
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(input_frame, text="City:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.city_var = tk.StringVar(value="Wroclaw")
        city_entry = ttk.Entry(input_frame, textvariable=self.city_var, width=30)
        city_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Control buttons
        button_frame = ttk.LabelFrame(main_frame, text="Actions")
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        # First row of buttons
        self.train_btn = ttk.Button(
            button_frame,
            text="Train ML Model",
            command=self.train_model
        )
        self.train_btn.grid(row=0, column=0, padx=5, pady=5, sticky=tk.EW)
        
        self.current_btn = ttk.Button(
            button_frame,
            text="Current Weather",
            command=self.get_current
        )
        self.current_btn.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        
        self.predict_btn = ttk.Button(
            button_frame,
            text="Predict Weather",
            command=self.predict
        )
        self.predict_btn.grid(row=0, column=2, padx=5, pady=5, sticky=tk.EW)
        
        # Second row of buttons
        self.compare_btn = ttk.Button(
            button_frame,
            text="Compare with API",
            command=self.compare_predictions
        )
        self.compare_btn.grid(row=1, column=0, padx=5, pady=5, sticky=tk.EW)
        
        self.info_btn = ttk.Button(
            button_frame,
            text="Model Info",
            command=self.show_model_info
        )
        self.info_btn.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        
        # Configure column weights for button frame
        for i in range(3):
            button_frame.columnconfigure(i, weight=1)
        
        # Status section with progress bar
        status_frame = ttk.LabelFrame(main_frame, text="Status")
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_var = tk.StringVar(value="Ready -  ML Weather Predictor")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(padx=5, pady=2)
        
        # Progress bar for long operations
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            status_frame, 
            variable=self.progress_var, 
            mode='indeterminate'
        )
        self.progress_bar.pack(fill=tk.X, padx=5, pady=2)
        
        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Results")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Setup results text area
        self.results_text = scrolledtext.ScrolledText(
            results_frame, 
            wrap=tk.WORD, 
            width=80, 
            height=20,
            font=('Courier', 10)
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initialize with read-only state to prevent user editing
        self.results_text.config(state=tk.DISABLED)
    
        # Check if model is already trained
        self.update_status()

    def update_status(self):
        try:
            if self.predictor.ml_model.is_trained:
                self.status_var.set("  Model trained and ready")
            else:
                self.status_var.set("  Model not trained - click 'Train ML Model' first")
        except:
            self.status_var.set("  Model status unknown")

    def log_result(self, message):
        """Add a message to the results text area"""
        # Temporarily enable the widget if it's disabled
        current_state = self.results_text.cget('state')
        if current_state == 'disabled':
            self.results_text.config(state=tk.NORMAL)
            
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)
        
        # Re-disable if it was disabled
        if current_state == 'disabled':
            self.results_text.config(state=tk.DISABLED)
    
        self.update()

    def clear_results(self):
        """Clear the results text area"""
        # Temporarily enable if disabled
        current_state = self.results_text.cget('state')
        if current_state == 'disabled':
            self.results_text.config(state=tk.NORMAL)
            
        self.results_text.delete(1.0, tk.END)
        
        # Re-disable if it was disabled
        if current_state == 'disabled':
            self.results_text.config(state=tk.DISABLED)

    def train_model(self):
        """Train the ML model"""
        city = self.city_var.get().strip()
        if not city:
            messagebox.showerror("Error", "Please enter a city name")
            return
            
        self.clear_results()
        self.status_var.set("Training model... This may take a few minutes")
        self.train_btn.config(state=tk.DISABLED)
        self.start_progress()
        
        def train_worker():
            try:
                self.log_result(f"  Training  ML Model for {city}...")
                self.log_result("=" * 50)
                
                success = self.predictor.train_model(city)
                
                self.after(0, lambda: self.on_training_complete(success))
                    
            except Exception as e:
                error_msg = f"Training error: {str(e)}"
                self.after(0, lambda: self.on_training_error(error_msg))
        
        # Start training in background thread
        thread = threading.Thread(target=train_worker)
        thread.daemon = True
        thread.start()

    def on_training_complete(self, success):
        """Called when training completes successfully"""
        self.progress_bar.stop()
        self.progress_bar.pack_forget()

        if success:
            self.log_result("Model training completed!")
            metrics = self.predictor.get_model_performance()
            
            self.log_result("Training Metrics:")
            
            # Handle nested performance metrics structure
            if isinstance(metrics, dict):
                # Check for temperature metrics (new structure)
                if 'temperature' in metrics and isinstance(metrics['temperature'], dict):
                    temp_metrics = metrics['temperature']
                    self.log_result("  Temperature Model:")
                    if 'mae' in temp_metrics:
                        self.log_result(f"    MAE: {self._format_metric_value(temp_metrics['mae'])} deg C")
                    if 'rmse' in temp_metrics:
                        self.log_result(f"    RMSE: {self._format_metric_value(temp_metrics['rmse'])} deg C")
                    if 'r2' in temp_metrics:
                        self.log_result(f"    R^2: {self._format_metric_value(temp_metrics['r2'])}")
                
                # Show humidity metrics if available
                if 'humidity' in metrics and isinstance(metrics['humidity'], dict):
                    humid_metrics = metrics['humidity']
                    if humid_metrics and 'mae' in humid_metrics:
                        self.log_result("  Humidity Model:")
                        self.log_result(f"    MAE: {self._format_metric_value(humid_metrics['mae'])}%")
                        self.log_result(f"    RMSE: {self._format_metric_value(humid_metrics['rmse'])}%")
                        self.log_result(f"    R^2: {self._format_metric_value(humid_metrics['r2'])}")
                
                # Show wind metrics if available
                if 'wind_speed' in metrics and isinstance(metrics['wind_speed'], dict):
                    wind_metrics = metrics['wind_speed']
                    if wind_metrics and 'mae' in wind_metrics:
                        self.log_result("  Wind Speed Model:")
                        self.log_result(f"    MAE: {self._format_metric_value(wind_metrics['mae'])} m/s")
                        self.log_result(f"    RMSE: {self._format_metric_value(wind_metrics['rmse'])} m/s")
                        self.log_result(f"    R^2: {self._format_metric_value(wind_metrics['r2'])}")
            
            # Handle legacy format (old structure)
            elif 'mae' in metrics:
                self.log_result(f"  MAE: {self._format_metric_value(metrics['mae'])} deg C")
                self.log_result(f"  RMSE: {self._format_metric_value(metrics['rmse'])} deg C")
                self.log_result(f"  R^2: {self._format_metric_value(metrics['r2'])}")
            
            # Add training info
            if 'training_time' in metrics:
                self.log_result(f"\nTraining Time: {metrics['training_time']:.2f} seconds")
            if 'samples' in metrics:
                self.log_result(f"Training Samples: {metrics['samples']}")
            
        else:
            self.log_result("  Model training failed")
            self.status_var.set("  Training failed")
    
        self.train_btn.config(state=tk.NORMAL)
        
        # Disable text widget to prevent editing
        self.results_text.config(state=tk.DISABLED)

    def on_training_error(self, error_msg):
        """Called when training encounters an error"""
        self.log_result(f"  {error_msg}")
        messagebox.showerror("Training Error", error_msg)
        self.status_var.set("  Training failed")
        self.stop_progress()
        self.train_btn.config(state=tk.NORMAL)

    def get_current(self):
        """Get current weather"""
        city = self.city_var.get().strip()
        if not city:
            messagebox.showerror("Error", "Please enter a city name")
            return
            
        self.clear_results()
        self.status_var.set("Fetching current weather...")
        
        try:
            data = self.predictor.get_current_weather(city)
            
            self.log_result(f"  CURRENT WEATHER FOR {data.get('name', city).upper()}")
            self.log_result("=" * 50)
            self.log_result(f"Temperature: {data['main']['temp']} deg C (feels like {data['main']['feels_like']} deg C)")
            self.log_result(f"Description: {data['weather'][0]['description'].title()}")
            self.log_result(f"Humidity: {data['main']['humidity']}%")
            self.log_result(f"Pressure: {data['main']['pressure']} hPa")
            self.log_result(f"Wind: {data['wind']['speed']} m/s")
            if 'visibility' in data:
                self.log_result(f"Visibility: {data['visibility']/1000:.1f} km")
            
            self.status_var.set("  Current weather retrieved")
            
        except Exception as e:
            error_msg = f"Error fetching weather: {str(e)}"
            self.log_result(f"  {error_msg}")
            messagebox.showerror("Weather Error", error_msg)
            self.status_var.set("  Failed to get weather")

    def predict(self):
        """Predict weather using ML model with multi-time forecasting"""
        city = self.city_var.get().strip()
        if not city:
            messagebox.showerror("Error", "Please enter a city name")
            return
            
        self.clear_results()
        self.status_var.set("Generating weather prediction...")
        
        try:
            # Add debug information
            self.log_result(f"Attempting to predict weather for: {city}")
            
            # Get the prediction
            result = self.predictor.predict_weather(city)
            
            # Check if result contains an error
            if 'error' in result:
                self.log_result(f"Error: {result['error']}")
                self.log_result("Please check the city name and try again.")
                
                # Log some guidance for common issues
                if "city not found" in str(result['error']).lower():
                    self.log_result(f"\nHint: The city '{city}' was not found in the weather database.")
                    self.log_result("Please check spelling or try a larger nearby city.")
                elif "api key" in str(result['error']).lower():
                    self.log_result("\nHint: There may be an issue with the API key configuration.")
                
                self.status_var.set("Prediction failed")
                self.results_text.config(state=tk.DISABLED)
                return
                
            # Log complete result structure for debugging
            import json
            debug_result = {k: str(v) for k, v in result.items()}
            self.log_result(f"Full result keys: {', '.join(result.keys())}")
            
            # If we made it here, we have valid result data to display
            self.log_result(f"\n   ML WEATHER PREDICTION")
            self.log_result("=" * 50)
            self.log_result(f"City: {result['city']}")
            self.log_result(f"Prediction Date: {result['prediction_date']}")
            
            # Display temperature prediction always
            self.log_result(f"\nTemperature: {result['predicted_temp']:.1f} deg C")
            
            # Display humidity prediction with better fallback
            if 'predicted_humidity' in result:
                if result['predicted_humidity'] is not None:
                    # Format to display integer percentage
                    humidity_value = float(result['predicted_humidity'])
                    self.log_result(f"Humidity: {humidity_value:.0f}%")
                else:
                    self.log_result("Humidity: Not available (model not trained)")
            else:
                self.log_result("Humidity: Not available (missing prediction)")
            
            # Display wind prediction with better fallback
            if 'predicted_wind' in result:
                if result['predicted_wind'] is not None:
                    # Format to display with one decimal place
                    wind_value = float(result['predicted_wind'])
                    self.log_result(f"Wind Speed: {wind_value:.1f} m/s")
                else:
                    self.log_result("Wind Speed: Not available (model not trained)")
            else:
                self.log_result("Wind Speed: Not available (missing prediction)")
            
            # Display weather condition if available
            if 'predicted_condition' in result and result['predicted_condition'] is not None:
                self.log_result(f"Weather: {result['predicted_condition']}")
            
            # Display model used if available
            if 'model_used' in result:
                self.log_result(f"\nModel Used: {result['model_used']}")
            elif 'model_type' in result:
                self.log_result(f"\nModel Used: {result['model_type']}")
            
            # Display feature importance if available
            if 'model_insights' in result and result['model_insights']:
                self.log_result("\n  Top Feature Importance:")
                sorted_features = sorted(
                    result['model_insights'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:8]  # Show top 8 features
                
                for feature, importance in sorted_features:
                    self.log_result(f"  {feature}: {importance:.3f}")
        except Exception as e:
            error_msg = f"Prediction error: {str(e)}"
            self.log_result(f"  {error_msg}")
            
            # Add debug info
            import traceback
            self.log_result("\nDebug information for developers:")
            self.log_result("-" * 50)
            self.log_result(traceback.format_exc())
            
            messagebox.showerror("Prediction Error", error_msg)
            self.status_var.set("  Prediction failed")
    
        # Disable text widget to prevent editing
        self.results_text.config(state=tk.DISABLED)

    def compare_predictions(self):
        """Compare ML prediction with API forecast"""
        city = self.city_var.get().strip()
        if not city:
            messagebox.showerror("Error", "Please enter a city name")
            return
            
        self.clear_results()
        self.status_var.set("Comparing predictions...")
        
        try:
            result = self.predictor.predict_weather_with_comparison(city)
            
            # Add debug information to diagnose comparison issues
            comparison = result.get('prediction_comparison', {})
            ml_pred = comparison.get('ml_prediction', {})
            api_forecast = comparison.get('api_forecast', {})
            
            # Debug the comparison dictionary to see what values are available
            print("COMPARISON DEBUG:")
            print(f"ML Predicted Humidity: {ml_pred.get('predicted_humidity')}")
            print(f"API Avg Humidity: {api_forecast.get('api_avg_humidity')}")
            print(f"Humidity Difference: {comparison.get('humidity_difference')}")
            print(f"ML Predicted Wind: {ml_pred.get('predicted_wind')}")
            print(f"API Avg Wind: {api_forecast.get('api_avg_wind')}")
            print(f"Wind Difference: {comparison.get('wind_difference')}")
            
            self.log_result(f"  WEATHER PREDICTION COMPARISON FOR {result['city'].upper()}")
            self.log_result("=" * 60)
            
            # Current weather
            current = result['current_weather']
            self.log_result("  CURRENT WEATHER:")
            self.log_result(f"   Temperature: {current['main']['temp']} deg C")
            self.log_result(f"   Description: {current['weather'][0]['description']}")
            self.log_result(f"   Humidity: {current['main']['humidity']}%")
            self.log_result(f"   Pressure: {current['main']['pressure']} hPa")
            self.log_result(f"   Wind: {current['wind']['speed']} m/s")
            
            # ML prediction
            comparison = result['prediction_comparison']
            ml_pred = comparison['ml_prediction']
            self.log_result(f"\n   ML PREDICTION:")
            self.log_result(f"   Date: {ml_pred['prediction_date']}")
            self.log_result(f"   Predicted Temperature: {ml_pred['predicted_temp']:.1f} deg C")
            
            # Add wind speed prediction right after temperature
            if 'predicted_wind' in ml_pred and ml_pred['predicted_wind'] is not None:
                self.log_result(f"   Predicted Wind Speed: {ml_pred['predicted_wind']:.1f} m/s")
            else:
                # Check if wind value might be in current_weather as fallback - don't show "Not available"
                try:
                    current_wind = result['current_weather']['wind']['speed']
                    self.log_result(f"   Predicted Wind Speed: {current_wind:.1f} m/s (current)")
                except:
                    self.log_result(f"   Predicted Wind Speed: Not available")
            
            # Add humidity prediction after wind speed
            if 'predicted_humidity' in ml_pred and ml_pred['predicted_humidity'] is not None:
                self.log_result(f"   Predicted Humidity: {ml_pred['predicted_humidity']:.0f}%")
            else:
                # Check if humidity value might be in current_weather as fallback - don't show "Not available"
                try:
                    current_humidity = result['current_weather']['main']['humidity']
                    self.log_result(f"   Predicted Humidity: {current_humidity:.0f}% (current)")
                except:
                    self.log_result(f"   Predicted Humidity: Not available")
            
            # Add weather condition if available
            if 'predicted_condition' in ml_pred and ml_pred['predicted_condition'] is not None:
                self.log_result(f"   Predicted Weather: {ml_pred['predicted_condition']}")
            
            # Display model used
            model_used = ml_pred.get('model_used', 'Unknown')
            self.log_result(f"   Model: {model_used}")
            
            # API forecast comparison with temperature, humidity, and wind
            if 'api_forecast' in comparison and comparison['api_forecast']:
                api_forecast = comparison['api_forecast']
                self.log_result(f"\n  API FORECAST:")
                
                # Display temperature from API forecast
                if 'api_avg_temp' in api_forecast and api_forecast['api_avg_temp'] is not None:
                    self.log_result(f"   Average Temperature: {api_forecast['api_avg_temp']:.1f} deg C")
                    
                    # Show min-max range for temperature
                    if 'api_min_temp' in api_forecast and 'api_max_temp' in api_forecast:
                        self.log_result(f"   Temperature Range: {api_forecast['api_min_temp']:.1f} to {api_forecast['api_max_temp']:.1f} deg C")
                
                # Display humidity from API forecast if available
                if 'api_avg_humidity' in api_forecast and api_forecast['api_avg_humidity'] is not None:
                    self.log_result(f"   Average Humidity: {api_forecast['api_avg_humidity']:.0f}%")
                    
                    # Show min-max range for humidity
                    if 'api_min_humidity' in api_forecast and 'api_max_humidity' in api_forecast:
                        self.log_result(f"   Humidity Range: {api_forecast['api_min_humidity']:.0f} to {api_forecast['api_max_humidity']:.0f}%")
                
                # Display wind speed from API forecast if available
                if 'api_avg_wind' in api_forecast and api_forecast['api_avg_wind'] is not None:
                    self.log_result(f"   Average Wind Speed: {api_forecast['api_avg_wind']:.1f} m/s")
                    
                    # Show min-max range for wind speed
                    if 'api_min_wind' in api_forecast and 'api_max_wind' in api_forecast:
                        self.log_result(f"   Wind Speed Range: {api_forecast['api_min_wind']:.1f} to {api_forecast['api_max_wind']:.1f} m/s")
                
                # Comparison section with all metrics
                self.log_result(f"\n  PREDICTION COMPARISON:")
                
                # Temperature comparison
                if 'difference' in comparison and comparison['difference'] is not None:
                    self.log_result(f"   Temperature Difference: {comparison['difference']:.1f} deg C")
                    
                    # Add assessment of temperature agreement
                    if comparison['difference'] < 1.0:
                        self.log_result("   Excellent agreement for temperature!")
                    elif comparison['difference'] < 2.0:
                        self.log_result("   Good agreement for temperature")
                    elif comparison['difference'] < 3.0:
                        self.log_result("   Moderate agreement for temperature")
                    else:
                        self.log_result("   Significant difference for temperature")
    
                # Wind speed comparison - ensure we're checking for the presence of both ML and API values
                if ('predicted_wind' in ml_pred and ml_pred['predicted_wind'] is not None and
                    'api_avg_wind' in api_forecast and api_forecast['api_avg_wind'] is not None):
                    # Ensure wind_difference exists, if not calculate it here as fallback
                    wind_diff = comparison.get('wind_difference')
                    if wind_diff is None and ml_pred['predicted_wind'] is not None and api_forecast['api_avg_wind'] is not None:
                        wind_diff = abs(float(ml_pred['predicted_wind']) - float(api_forecast['api_avg_wind']))
                        print(f"Calculated fallback wind difference: {wind_diff}")
                    
                    if wind_diff is not None:
                        self.log_result(f"   Wind Speed Difference: {wind_diff:.1f} m/s")
                        
                        # Add assessment of wind speed agreement
                        if wind_diff < 1.0:
                            self.log_result("   Excellent agreement for wind!")
                        elif wind_diff < 2.0:
                            self.log_result("   Good agreement for wind")
                        else:
                            self.log_result("   Moderate agreement for wind")
                        
                        # Add within range info for wind
                        if comparison.get('wind_within_range') is not None:
                            if comparison['wind_within_range']:
                                self.log_result("   Wind prediction within API forecast range")
                            else:
                                self.log_result("   Wind prediction outside API forecast range")
        
                # Humidity comparison - ensure we're checking for the presence of both ML and API values
                if ('predicted_humidity' in ml_pred and ml_pred['predicted_humidity'] is not None and
                    'api_avg_humidity' in api_forecast and api_forecast['api_avg_humidity'] is not None):
                    # Ensure humidity_difference exists, if not calculate it here as fallback
                    humidity_diff = comparison.get('humidity_difference')
                    if humidity_diff is None and ml_pred['predicted_humidity'] is not None and api_forecast['api_avg_humidity'] is not None:
                        humidity_diff = abs(float(ml_pred['predicted_humidity']) - float(api_forecast['api_avg_humidity']))
                        print(f"Calculated fallback humidity difference: {humidity_diff}")
                    
                    if humidity_diff is not None:
                        self.log_result(f"   Humidity Difference: {humidity_diff:.0f}%")
                        
                        # Add assessment of humidity agreement
                        if humidity_diff < 5.0:
                            self.log_result("   Excellent agreement for humidity!")
                        elif humidity_diff < 10.0:
                            self.log_result("   Good agreement for humidity")
                        else:
                            self.log_result("   Moderate agreement for humidity")
                        
                        # Add within range info for humidity
                        if comparison.get('humidity_within_range') is not None:
                            if comparison['humidity_within_range']:
                                self.log_result("   Humidity prediction within API forecast range")
                            else:
                                self.log_result("   Humidity prediction outside API forecast range")
        
            # Display model information
            if 'model_info' in result:
                self.log_result(f"\n   Model Info: {result['model_info']}")
            
            self.status_var.set("  Comparison completed")
        
        except Exception as e:
            error_msg = f"Comparison error: {str(e)}"
            self.log_result(f"  {error_msg}")
            messagebox.showerror("Comparison Error", error_msg)
            self.status_var.set("  Comparison failed")
        
            # Disable text widget to prevent editing
            self.results_text.config(state=tk.DISABLED)

    def show_model_info(self):
        """Display detailed model information"""
        self.clear_results()
        self.status_var.set("Retrieving model information...")
        
        try:
            self.log_result("   ML MODEL INFORMATION")
            self.log_result("=" * 50)
            # Fix: Use direct access to get_model_info() method
            info = self.predictor.ml_model.get_model_info()
            self.log_result(f"Status: {info}")
            # Fix: Check is_trained attribute directly
            self.log_result(f"Trained: {'Yes' if self.predictor.ml_model.is_trained else 'No'}")
            
            # Fix: Check is_trained attribute directly
            if self.predictor.ml_model.is_trained:
                self.log_result("\n  DETAILED MODEL PERFORMANCE:")
                # Fix: Access model_performance directly 
                performance = self.predictor.ml_model.model_performance
                
                # Handle performance metrics correctly
                if isinstance(performance, dict):
                    for key, value in performance.items():
                        if key != 'feature_importance':  # Skip feature importance for brevity
                            self.log_result(f"   {key.replace('_', ' ').title()}: {self._format_metric_value(value)}")
                
                    # If training metrics are available
                    if 'rmse' in performance:
                        self.log_result(f"   RMSE: {self._format_metric_value(performance['rmse'])} deg C")
                    if 'mae' in performance:
                        self.log_result(f"   MAE: {self._format_metric_value(performance['mae'])} deg C")
                    if 'r2' in performance:
                        self.log_result(f"   R^2 Score: {self._format_metric_value(performance['r2'])}")
                else:
                    self.log_result(f"   Performance data: {performance}")
            else:
                self.log_result("\n   Train the model first to see performance metrics")
            
            self.status_var.set("  Model info displayed")
            
        except Exception as e:
            error_msg = f"Info error: {str(e)}"
            self.log_result(f"  {error_msg}")
            messagebox.showerror("Info Error", error_msg)
            self.status_var.set("  Failed to get info")
    
        # Disable text widget to prevent user editing
        self.results_text.config(state=tk.DISABLED)

    def _format_metric_value(self, value):
        """Helper to format metric values properly regardless of type"""
        if isinstance(value, float):
            # Format float values appropriately based on magnitude
            if 0 <= value <= 1:  # Probably an R² value or similar ratio
                return f"{value:.3f}"
            else:  # Regular float value
                return f"{value:.2f}"
        elif isinstance(value, int):
            # Format integers
            return str(value)
        elif hasattr(value, 'item') and callable(getattr(value, 'item')):
            # Handle NumPy types safely
            try:
                val = value.item()
                return self._format_metric_value(val)  # Recursively format the extracted value
            except:
                return str(value)
        elif isinstance(value, dict):
            return "Complex data structure (dictionary)"
        elif isinstance(value, list):
            return "Complex data structure (list)"
        else:
            return str(value)

    def start_progress(self):
        """Start the progress bar animation"""
        self.progress_bar.start(10)

    def stop_progress(self):
        """Stop the progress bar animation"""
        self.progress_bar.stop()

    def run_in_background(self, target, *args, **kwargs):
        """Run a function in a background thread with progress indication"""
        def worker():
            try:
                self.start_progress()
                result = target(*args, **kwargs)
                return result
            finally:
                self.stop_progress()
                
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()

if __name__ == '__main__':
    WeatherGUI().mainloop()