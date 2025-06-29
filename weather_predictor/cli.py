import argparse
from .predictor import WeatherPredictor
from .exceptions import DataError

class WeatherCLI:
    def __init__(self):
        self.predictor = WeatherPredictor()

    def run(self, argv=None) -> int:
        parser = argparse.ArgumentParser(prog="Unique_weather_predictor.py --mode cli")
        sub = parser.add_subparsers(dest='cmd')
        
        # Train model command
        train = sub.add_parser('train', help='Train the enhanced ML model')
        train.add_argument('--city', default='Wroclaw', help='City to train the model on')
        
        # Prediction commands
        predict = sub.add_parser('predict', help='Predict weather using enhanced ML model')
        predict.add_argument('--city', required=True, help='City to predict weather for')
        
        compare = sub.add_parser('compare', help='Compare ML prediction with API forecast')
        compare.add_argument('--city', required=True, help='City to compare predictions for')
        
        # Current weather command
        current = sub.add_parser('current', help='Get current weather for a city')
        current.add_argument('--city', required=True, help='City to get current weather for')
        
        # Model info command
        info = sub.add_parser('info', help='Get model information and performance metrics')
        
        args = parser.parse_args(argv)

        if args.cmd == 'train':
            try:
                print(f"Training enhanced ML model for {args.city}...")
                success = self.predictor.train_model(args.city)
                if success:
                    print("✅ Enhanced ML model training completed successfully!")
                    print("Model performance:")
                    performance = self.predictor.get_model_performance()
                    for model_name, metrics in performance.items():
                        print(f"  {model_name.title()}: MAE={metrics.get('mae', 'N/A'):.2f} deg C, R2={metrics.get('r2', 'N/A'):.3f}")
                return 0
            except Exception as e:
                print(f"Error training model: {e}")
                return 1

        if args.cmd == 'predict':
            try:
                result = self.predictor.predict_weather(args.city)
                
                # Check if we have multi-time forecast
                if 'multi_forecast' in result:
                    multi_forecast = result['multi_forecast']
                    print("🤖 ENHANCED ML WEATHER FORECAST:")
                    print(f"City: {multi_forecast['city']}")
                    print(f"Date: {multi_forecast['prediction_date']}")
                    print("\n📅 TOMORROW'S FORECAST:")
                    print("=" * 50)
                    
                    for time_label, forecast in multi_forecast['forecasts'].items():
                        print(f"\n🕐 {time_label}:")
                        temp = forecast['temperature'].get('value', 0)
                        wind = forecast['wind_speed'].get('value', 0)
                        rain = forecast['rain_probability'].get('value', 0)
                        
                        print(f"   Temperature: {temp:.1f} deg C")
                        print(f"   Wind Speed: {wind:.1f} m/s")
                        print(f"   Rain Probability: {rain:.0f}%")
                        print(f"   Summary: {forecast['weather_summary']}")
                else:
                    # Legacy format
                    print("🤖 ENHANCED ML PREDICTION:")
                    print(f"City: {result['city']}")
                    print(f"Date: {result['prediction_date']} at {result.get('prediction_time', 'N/A')}")
                    print(f"Predicted Temperature: {result['predicted_temp']:.1f} deg C")
                    print(f"Model Used: {result.get('model_used', 'Unknown')}")
                    print(f"Confidence: {result.get('prediction_confidence', 'N/A'):.1%}")
                    
                    if result.get('feature_importance'):
                        print("\nTop Feature Importance:")
                        sorted_features = sorted(result['feature_importance'].items(), 
                                               key=lambda x: x[1], reverse=True)[:5]
                        for feature, importance in sorted_features:
                            print(f"  {feature}: {importance:.3f}")
                return 0
            except Exception as e:
                print(f"Error predicting weather: {e}")
                return 1

        if args.cmd == 'compare':
            try:
                result = self.predictor.predict_weather_with_comparison(args.city)
                self.show_comparison(args.city)
                return 0
            except Exception as e:
                print(f"Error generating comparison: {e}")
                return 1

        if args.cmd == 'current':
            try:
                result = self.predictor.get_current_weather(args.city)
                print(f"📍 CURRENT WEATHER FOR {result.get('name', args.city).upper()}:")
                print(f"   Temperature: {result['main']['temp']} deg C (feels like {result['main']['feels_like']} deg C)")
                print(f"   Description: {result['weather'][0]['description'].title()}")
                print(f"   Humidity: {result['main']['humidity']}%")
                print(f"   Pressure: {result['main']['pressure']} hPa")
                print(f"   Wind: {result['wind']['speed']} m/s")
                if 'visibility' in result:
                    print(f"   Visibility: {result['visibility']/1000:.1f} km")
                return 0
            except Exception as e:
                print(f"Error fetching current weather: {e}")
                return 1

        if args.cmd == 'info':
            try:
                print("🔍 MODEL INFORMATION:")
                print(f"   Status: {self.predictor.get_model_info()}")
                print(f"   Trained: {'Yes' if self.predictor.is_model_trained() else 'No'}")
                
                if self.predictor.is_model_trained():
                    print("\n📊 MODEL PERFORMANCE:")
                    performance = self.predictor.get_model_performance()
                    for model_name, metrics in performance.items():
                        print(f"   {model_name.title()}:")
                        print(f"     MAE: {metrics.get('mae', 'N/A'):.2f} deg C")
                        print(f"     RMSE: {metrics.get('rmse', 'N/A'):.2f} deg C")
                        print(f"     R2: {metrics.get('r2', 'N/A'):.3f}")
                        print(f"     CV-MAE: {metrics.get('cv_mae', 'N/A'):.2f} deg C")
                return 0
            except Exception as e:
                print(f"Error getting model info: {e}")
                return 1

        parser.print_help()
        return 0

    def show_comparison(self, city: str):
        """Show comparison between ML prediction and API forecast"""
        print(f"Comparing ML prediction with API forecast for {city}...")
        
        try:
            # Get prediction comparison
            result = self.predictor.predict_weather_with_comparison(city)
            self.display_comparison(result)
        except Exception as e:
            print(f"Error: {str(e)}")
        
    def display_comparison(self, result: dict):
        """Display ML prediction compared to API forecast with enhanced parameters"""
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
            
        current_weather = result['current_weather']
        comparison = result['prediction_comparison']
        ml_pred = comparison['ml_prediction']
        api_forecast = comparison['api_forecast']
        
        # Display current weather
        print(f"\nCURRENT WEATHER IN {current_weather['name'].upper()}:")
        print(f"   Temperature: {current_weather['main']['temp']} deg C")
        print(f"   Description: {current_weather['weather'][0]['description']}")
        print(f"   Humidity: {current_weather['main']['humidity']}%")
        print(f"   Wind Speed: {current_weather['wind']['speed']} m/s")
        
        # Display ML prediction with all predicted metrics
        print(f"\nML MODEL PREDICTION FOR TOMORROW:")
        print(f"   Temperature: {ml_pred['predicted_temp']:.1f} deg C")
        
        if 'predicted_humidity' in ml_pred and ml_pred['predicted_humidity'] is not None:
            print(f"   Humidity: {ml_pred['predicted_humidity']:.0f}%")
            
        if 'predicted_wind' in ml_pred and ml_pred['predicted_wind'] is not None:
            print(f"   Wind Speed: {ml_pred['predicted_wind']:.1f} m/s")
        
        if 'predicted_condition' in ml_pred:
            print(f"   Weather: {ml_pred['predicted_condition']}")
        
        # Display API forecast with all metrics
        if api_forecast and 'api_avg_temp' in api_forecast and api_forecast['api_avg_temp'] is not None:
            print(f"\nAPI FORECAST FOR TOMORROW:")
            print(f"   Average Temperature: {api_forecast['api_avg_temp']:.1f} deg C")
            print(f"   Temperature Range: {api_forecast['api_min_temp']:.1f}-{api_forecast['api_max_temp']:.1f} deg C")
            
            # Show humidity data if available
            if 'api_avg_humidity' in api_forecast and api_forecast['api_avg_humidity'] is not None:
                print(f"   Average Humidity: {api_forecast['api_avg_humidity']:.0f}%")
                print(f"   Humidity Range: {api_forecast['api_min_humidity']:.0f}-{api_forecast['api_max_humidity']:.0f}%")
            
            # Show wind data if available
            if 'api_avg_wind' in api_forecast and api_forecast['api_avg_wind'] is not None:
                print(f"   Average Wind Speed: {api_forecast['api_avg_wind']:.1f} m/s")
                print(f"   Wind Speed Range: {api_forecast['api_min_wind']:.1f}-{api_forecast['api_max_wind']:.1f} m/s")
        
        # Display comparisons for all metrics
        print("\nCOMPARISON BETWEEN ML AND API FORECASTS:")
        
        # Temperature comparison
        if comparison.get('difference') is not None:
            print(f"   Temperature Difference: {comparison['difference']:.1f} deg C")
            if comparison.get('within_range'):
                print("   ✓ ML temperature prediction is within API forecast range")
            else:
                print("   ✗ ML temperature prediction is outside API forecast range")
            
            # Add assessment for temperature
            if comparison['difference'] < 1.0:
                print("   Temperature: Excellent agreement!")
            elif comparison['difference'] < 2.0:
                print("   Temperature: Good agreement")
            elif comparison['difference'] < 3.0:
                print("   Temperature: Moderate agreement")
            else:
                print("   Temperature: Significant difference")
        
        # Humidity comparison
        if comparison.get('humidity_difference') is not None:
            print(f"   Humidity Difference: {comparison['humidity_difference']:.0f}%")
            if comparison.get('humidity_within_range'):
                print("   ✓ ML humidity prediction is within API forecast range")
            else:
                print("   ✗ ML humidity prediction is outside API forecast range")
            
            # Add assessment for humidity
            if comparison['humidity_difference'] < 5.0:
                print("   Humidity: Excellent agreement!")
            elif comparison['humidity_difference'] < 10.0:
                print("   Humidity: Good agreement")
            else:
                print("   Humidity: Moderate agreement")
        
        # Wind speed comparison
        if comparison.get('wind_difference') is not None:
            print(f"   Wind Speed Difference: {comparison['wind_difference']:.1f} m/s")
            if comparison.get('wind_within_range'):
                print("   ✓ ML wind prediction is within API forecast range")
            else:
                print("   ✗ ML wind prediction is outside API forecast range")
            
            # Add assessment for wind
            if comparison['wind_difference'] < 1.0:
                print("   Wind Speed: Excellent agreement!")
            elif comparison['wind_difference'] < 2.0:
                print("   Wind Speed: Good agreement")
            else:
                print("   Wind Speed: Moderate agreement")
        
        # Display overall model info
        print(f"\nMODEL INFORMATION:")
        print(f"   {result['model_info']}")

if __name__ == '__main__':
    import sys
    exit_code = WeatherCLI().run(sys.argv[1:])
    sys.exit(exit_code)