import requests
import json
import time
from typing import Dict, Any

class WeatherDataCollector:
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def get_current(self, city: str) -> dict:
        """Get current weather for a city with enhanced error handling and debugging"""
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.api_key}&units=metric"
        
        try:
            print(f"Fetching current weather for {city}...")
            response = requests.get(url, timeout=10)  # Add timeout
            
            # Debug the response
            print(f"API Response status: {response.status_code}")
            
            # Save response for debugging
            try:
                data = response.json()
                # Debug log the structure without exposing API key
                safe_data = self._sanitize_api_response(data)
                debug_msg = f"API Response data: {json.dumps(safe_data, indent=2)}"
                print(debug_msg[:500] + "..." if len(debug_msg) > 500 else debug_msg)
            except Exception as parse_error:
                print(f"Failed to parse response as JSON: {str(parse_error)}")
                print(f"Response text: {response.text[:500]}")
                return {
                    'error': True,
                    'message': f"Invalid API response: {str(parse_error)}",
                    'status_code': response.status_code
                }
            
            # Check for HTTP errors
            if response.status_code != 200:
                print(f"Error fetching weather for {city}: Status code {response.status_code}")
                return {
                    'error': True,
                    'status_code': response.status_code,
                    'message': f"API Error: {data.get('message', 'Unknown error')}"
                }
                
            # Create fallback data if main section is missing
            if 'main' not in data:
                print(f"Warning: API response for {city} is missing 'main' section")
                # Create fallback weather data based on average conditions
                fallback_data = self._create_fallback_weather(city)
                print(f"Using fallback weather data: {fallback_data}")
                return fallback_data
                
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"Network error getting weather for {city}: {e}")
            return {
                'error': True,
                'message': f"Network error: {str(e)}"
            }
        except Exception as e:
            print(f"Unexpected error getting weather for {city}: {e}")
            return {
                'error': True,
                'message': f"Unexpected error: {str(e)}"
            }
    
    def _sanitize_api_response(self, data: dict) -> dict:
        """Create a safe copy of the API response for debugging (remove API keys)"""
        if not isinstance(data, dict):
            return {'error': 'Response is not a dictionary'}
        
        # Make a copy to avoid modifying the original
        safe_data = data.copy()
        
        # Remove potentially sensitive information
        if 'appid' in safe_data:
            safe_data['appid'] = '[REDACTED]'
        
        return safe_data
    
    def _create_fallback_weather(self, city: str) -> dict:
        """Create fallback weather data when the API response is invalid"""
        import datetime
        import random
        
        # Get current time
        now = datetime.datetime.now()
        
        # Generate reasonable fallback data
        temp = 15 + random.uniform(-5, 5)  # Average temperature around 15°C
        
        # Create a complete fallback structure matching what the API would normally return
        return {
            'coord': {'lon': 0, 'lat': 0},
            'weather': [{
                'id': 800,
                'main': 'Clear',
                'description': 'clear sky (fallback data)',
                'icon': '01d'
            }],
            'base': 'fallback',
            'main': {
                'temp': temp,
                'feels_like': temp - 2,
                'temp_min': temp - 3,
                'temp_max': temp + 3,
                'pressure': 1013,
                'humidity': 70
            },
            'visibility': 10000,
            'wind': {
                'speed': 3.6,
                'deg': 220
            },
            'clouds': {
                'all': 20
            },
            'dt': int(now.timestamp()),
            'sys': {
                'type': 1,
                'id': 0,
                'country': 'XX',
                'sunrise': int((now.replace(hour=6, minute=0, second=0)).timestamp()),
                'sunset': int((now.replace(hour=18, minute=0, second=0)).timestamp())
            },
            'timezone': 0,
            'id': 0,
            'name': f"{city} (fallback data)",
            'cod': 200,
            '_fallback': True  # Flag to indicate this is fallback data
        }

    def get_forecast(self, city: str) -> dict:
        """Get weather forecast for a city with improved error handling"""
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={self.api_key}&units=metric"
        
        try:
            response = requests.get(url)
            
            if response.status_code != 200:
                print(f"Error fetching forecast for {city}: Status code {response.status_code}")
                return {
                    'error': True,
                    'status_code': response.status_code,
                    'message': f"API Error: {response.json().get('message', 'Unknown error')}"
                }
                
            data = response.json()
            
            # Validate API response
            if 'list' not in data or not data['list']:
                print(f"Warning: API forecast response for {city} is missing 'list' section or empty")
                fallback = self._create_fallback_forecast(city)
                return fallback
            
            return data
            
        except Exception as e:
            print(f"Error fetching forecast for {city}: {e}")
            return {
                'error': True,
                'message': f"Error fetching forecast: {str(e)}"
            }
    
    def _create_fallback_forecast(self, city: str) -> dict:
        """Create fallback forecast data when the API response is invalid"""
        import datetime
        import random
        
        # Get current time
        now = datetime.datetime.now()
        
        # Generate 5 days of forecast data (3-hour intervals)
        forecast_list = []
        base_temp = 15  # Start with an average temperature
        
        for day in range(5):
            for hour in [0, 3, 6, 9, 12, 15, 18, 21]:
                forecast_time = now + datetime.timedelta(days=day, hours=hour)
                
                # Vary temperature by time of day
                temp_offset = -5 if hour < 6 else (5 if 10 <= hour <= 16 else 0)
                temp = base_temp + temp_offset + random.uniform(-3, 3)
                
                # Determine weather condition (mostly clear or cloudy)
                weather_main = random.choice(['Clear', 'Clouds', 'Clouds', 'Rain']) 
                weather_desc = 'clear sky' if weather_main == 'Clear' else 'scattered clouds' if weather_main == 'Clouds' else 'light rain'
                
                forecast_list.append({
                    'dt': int(forecast_time.timestamp()),
                    'main': {
                        'temp': temp,
                        'feels_like': temp - 2,
                        'temp_min': temp - 2,
                        'temp_max': temp + 2,
                        'pressure': 1013,
                        'humidity': 70
                    },
                    'weather': [{
                        'id': 800 if weather_main == 'Clear' else 801,
                        'main': weather_main,
                        'description': weather_desc,
                        'icon': '01d' if weather_main == 'Clear' else '02d'
                    }],
                    'clouds': {'all': 20 if weather_main == 'Clear' else 60},
                    'wind': {'speed': 3.6, 'deg': 220},
                    'dt_txt': forecast_time.strftime('%Y-%m-%d %H:%M:%S')
                })
        
        return {
            'cod': '200',
            'message': 0,
            'cnt': len(forecast_list),
            'list': forecast_list,
            'city': {
                'id': 0,
                'name': f"{city} (fallback data)",
                'coord': {'lon': 0, 'lat': 0},
                'country': 'XX',
                'population': 0,
                'timezone': 0,
                'sunrise': int((now.replace(hour=6, minute=0, second=0)).timestamp()),
                'sunset': int((now.replace(hour=18, minute=0, second=0)).timestamp())
            },
            '_fallback': True  # Flag to indicate this is fallback data
        }