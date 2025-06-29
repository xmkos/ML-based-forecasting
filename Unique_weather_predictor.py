# Main entry script providing CLI, GUI and demo modes

import sys
import os

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Weather Prediction System with Enhanced ML')
    parser.add_argument('--mode', choices=['cli','gui','demo'], default='gui', 
                      help='Choose interface mode: cli, gui, or demo (standalone ML demo)')
    parser.add_argument('--city', default='Wroclaw', 
                      help='City for weather prediction (used in demo mode)')
    parser.add_argument('--days', type=int, default=30,
                      help='Number of days of historical data to use for ML training (default: 30)')
    parser.add_argument('--model', choices=['rf', 'gb', 'nn'], default='rf',
                      help='ML model to use: Random Forest (rf), Gradient Boosting (gb), or Neural Network (nn)')
    parser.add_argument('--force-retrain', action='store_true',
                      help='Force retraining of the model even if already trained')
    args, extras = parser.parse_known_args()
    
    if args.mode == 'demo':
        run_ml_demo(args.city, training_days=args.days, model_type=args.model, 
                   force_retrain=args.force_retrain)
    elif args.mode == 'cli':
        from weather_predictor.cli import WeatherCLI
        WeatherCLI().run()
    else:
        from weather_predictor.gui import WeatherGUI
        WeatherGUI().mainloop()

def run_ml_demo(city="Wroclaw", training_days=30, model_type="rf", force_retrain=False):
    try:
        from weather_predictor import WeatherPredictor
        from weather_predictor.config import API_KEY
        import time
        
        if not API_KEY:
            print("Error: API_KEY not found!")
            print("Please set your OpenWeatherMap API key in the .env file or environment variables")
            return
        
        print("ENHANCED WEATHER ML PREDICTION DEMO")
        print("=" * 60)
        print(f"Analyzing weather for: {city}")
        print(f"Configuration: Training days: {training_days}, Model: {model_type}")
        print("=" * 60)
        
        predictor = WeatherPredictor(model_type=model_type)
        
        try:
            current_weather = predictor.get_current_weather(city)
            
            if 'main' not in current_weather:
                print(f"Error: Invalid weather data format for {city}")
                if 'message' in current_weather:
                    print(f"API message: {current_weather['message']}")
                return
                
            print(f"\nCURRENT WEATHER IN {city.upper()}:")
            print(f"   Temperature: {current_weather['main']['temp']} deg C")
            print(f"   Description: {current_weather['weather'][0]['description']}")
            print(f"   Humidity: {current_weather['main']['humidity']}%")
            print(f"   Pressure: {current_weather['main']['pressure']} hPa")
            print(f"   Wind Speed: {current_weather['wind']['speed']} m/s")
            print("-" * 50)
        except Exception as e:
            print(f"Error getting current weather: {str(e)}")
            print("Cannot proceed without valid weather data.")
            return
        if not predictor.is_model_trained() or force_retrain:
            print(f"Training ML model using {training_days} days of historical data...")
            print("This might take a few minutes for better accuracy...")
            
            start_time = time.time()
            validation_metrics = predictor.train_model(city, days=training_days, validate=True)
            training_time = time.time() - start_time
            
            print(f"Model training completed in {training_time:.2f} seconds!")
            
            print("\nVALIDATION METRICS:")
            if validation_metrics and isinstance(validation_metrics, dict):
                if 'mae' in validation_metrics:
                    print(f"   Mean Absolute Error: {validation_metrics['mae']:.2f} deg C")
                else:
                    print("   Mean Absolute Error: Not available")
                    
                if 'rmse' in validation_metrics:
                    print(f"   Root Mean Squared Error: {validation_metrics['rmse']:.2f} deg C")
                    rmse_value = validation_metrics['rmse']
                else:
                    print("   Root Mean Squared Error: Not available")
                    rmse_value = None
                    
                if 'r2' in validation_metrics:
                    print(f"   R^2 Score: {validation_metrics['r2']:.3f}")
                else:
                    print("   R^2 Score: Not available")
                
                print(f"   Available metrics: {list(validation_metrics.keys())}")
                
                if rmse_value is not None and rmse_value > 3.0:
                    print("Warning: Model accuracy is low. Consider using more training data")
                    print("   or trying a different model type (--model gb or --model nn)")
            else:
                print("   No validation metrics available or invalid format")
                print(f"   Received: {type(validation_metrics)} - {validation_metrics}")
        else:
            print("ML model already trained! Use --force-retrain to train again.")
        
        try:
            print("\nGENERATING PREDICTION COMPARISON...")
            result = predictor.predict_weather_with_comparison(city)
            
            if 'error' in result:
                print(f"Prediction error: {result['error']}")
                return
                
            comparison = result['prediction_comparison']
            if 'ml_prediction' not in comparison:
                print("Error: Missing ML prediction data in the results")
                return
                
            ml_pred = comparison['ml_prediction']
            
            print(f"\nML PREDICTION FOR TOMORROW:")
            print(f"   Date: {ml_pred['prediction_date']}")
            print(f"   Predicted Temperature: {ml_pred['predicted_temp']:.1f} deg C")
            
            if 'predicted_humidity' in ml_pred and ml_pred['predicted_humidity'] is not None:
                print(f"   Predicted Humidity: {ml_pred['predicted_humidity']:.0f}%")
            
            if 'predicted_wind' in ml_pred and ml_pred['predicted_wind'] is not None:
                print(f"   Predicted Wind Speed: {ml_pred['predicted_wind']:.1f} m/s")
                
            if 'predicted_condition' in ml_pred and ml_pred['predicted_condition'] is not None:
                print(f"   Predicted Weather: {ml_pred['predicted_condition']}")
            
            model_used = ml_pred.get('model_used', 'Unknown')
            print(f"   Model Used: {model_used}")
            
            if 'model_insights' in ml_pred and ml_pred['model_insights']:
                print(f"\nMODEL FEATURE IMPORTANCE:")
                for feature, importance in ml_pred['model_insights'].items():
                    print(f"   {feature}: {importance:.3f}")
            
            if 'api_forecast' in comparison and comparison['api_forecast']:
                api_forecast = comparison['api_forecast']
                if 'api_avg_temp' in api_forecast and api_forecast['api_avg_temp'] is not None:
                    print(f"\nAPI FORECAST FOR TOMORROW:")
                    print(f"   Average Temperature: {api_forecast['api_avg_temp']:.1f} deg C")
                    
                    difference = comparison.get('difference')
                    if difference is not None:
                        print(f"\nCOMPARISON:")
                        print(f"   Temperature Difference: {difference:.1f} deg C")
                        
                        if difference < 1.0:
                            print("   Excellent agreement between ML and API for temperature!")
                        elif difference < 2.0:
                            print("   Good agreement between ML and API for temperature")
                        elif difference < 3.0:
                            print("   Moderate agreement between ML and API for temperature")
                        else:
                            print("   Significant difference between ML and API for temperature")
                    
                    if 'predicted_humidity' in ml_pred and ml_pred['predicted_humidity'] is not None and \
                       'api_avg_humidity' in api_forecast and api_forecast['api_avg_humidity'] is not None:
                        print(f"   Average Humidity: {api_forecast['api_avg_humidity']:.0f}%")
                        
                        humidity_diff = comparison.get('humidity_difference')
                        if humidity_diff is not None:
                            print(f"   Humidity Difference: {humidity_diff:.0f}%")
                            
                            if humidity_diff < 5.0:
                                print("   Excellent agreement between ML and API for humidity!")
                            elif humidity_diff < 10.0:
                                print("   Good agreement between ML and API for humidity")
                            else:
                                print("   Moderate agreement between ML and API for humidity")
                    
                    if 'predicted_wind' in ml_pred and ml_pred['predicted_wind'] is not None and \
                       'api_avg_wind' in api_forecast and api_forecast['api_avg_wind'] is not None:
                        print(f"   Average Wind Speed: {api_forecast['api_avg_wind']:.1f} m/s")
                        
                        wind_diff = comparison.get('wind_difference')
                        if wind_diff is not None:
                            print(f"   Wind Speed Difference: {wind_diff:.1f} m/s")
                            
                            if wind_diff < 1.0:
                                print("   Excellent agreement between ML and API for wind!")
                            elif wind_diff < 2.0:
                                print("   Good agreement between ML and API for wind")
                            else:
                                print("   Moderate agreement between ML and API for wind")
                
                if 'detailed_forecasts' in api_forecast and api_forecast['detailed_forecasts']:
                    print(f"\nDETAILED API FORECASTS FOR TOMORROW:")
                    for forecast in api_forecast['detailed_forecasts'][:3]:
                        if all(key in forecast for key in ['dt_txt', 'main', 'weather']):
                            time = forecast['dt_txt']
                            temp = forecast['main']['temp']
                            desc = forecast['weather'][0]['description']
                            print(f"   {time}: {temp} deg C, {desc}")
        
        except Exception as e:
            print(f"Error generating prediction: {str(e)}")
            import traceback
            traceback.print_exc()
        if 'model_info' in result:
            print(f"\nMODEL INFORMATION:")
            print(f"   {result['model_info']}")
        
        print("\n" + "=" * 60)
        print("Demo completed! Try --mode cli or --mode gui for full interface")
        
    except Exception as e:
        print(f"Error in ML demo: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print("Full traceback:")
        import traceback
        traceback.print_exc()
        
        print("\nDEBUG INFORMATION:")
        print(f"City: {city}")
        print(f"Training days: {training_days}")
        print(f"Model type: {model_type}")
        print(f"Force retrain: {force_retrain}")
        
        try:
            if 'predictor' in locals():
                print(f"Predictor created: Yes")
                print(f"Model trained: {predictor.is_model_trained()}")
            else:
                print(f"Predictor created: No")
        except:
            print(f"Error checking predictor status")

if __name__ == '__main__':
    main()
