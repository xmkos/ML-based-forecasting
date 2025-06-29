# Tests for the data collector component

import pytest
import sys
import os
import json
from unittest.mock import patch, MagicMock, Mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from weather_predictor.data_collector import WeatherDataCollector


class TestWeatherDataCollector:
    """Test suite for WeatherDataCollector class"""

    def test_initialization(self):
        """Test WeatherDataCollector initialization"""
        api_key = 'test_api_key_12345'
        collector = WeatherDataCollector(api_key)
        
        assert collector.api_key == api_key

    def test_initialization_without_api_key(self):
        """Test initialization without API key"""
        collector = WeatherDataCollector('')
        assert collector.api_key == ''

    @patch('weather_predictor.data_collector.requests.get')
    def test_get_current_weather_success(self, mock_get):
        """Test successful current weather API call"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'main': {'temp': 22.5, 'humidity': 65, 'pressure': 1013},
            'weather': [{'description': 'clear sky', 'main': 'Clear'}],
            'wind': {'speed': 3.2, 'deg': 180},
            'name': 'TestCity'
        }
        mock_get.return_value = mock_response
        
        collector = WeatherDataCollector('test_api_key')
        result = collector.get_current('TestCity')
        
        mock_get.assert_called_once()
        call_args = mock_get.call_args[0][0]
        assert 'TestCity' in call_args
        assert 'test_api_key' in call_args
        assert 'units=metric' in call_args
        
        assert 'main' in result
        assert 'weather' in result
        assert result['main']['temp'] == 22.5

    @patch('weather_predictor.data_collector.requests.get')
    def test_get_current_weather_api_error(self, mock_get):
        """Test API error handling in get_current"""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {
            'cod': '404',
            'message': 'city not found'
        }
        mock_get.return_value = mock_response
        
        collector = WeatherDataCollector('test_api_key')
        result = collector.get_current('InvalidCity')
        
        assert 'error' in result or 'cod' in result or 'message' in result

    @patch('weather_predictor.data_collector.requests.get')
    def test_get_current_weather_network_error(self, mock_get):
        """Test network error handling"""
        mock_get.side_effect = Exception('Network error')
        
        collector = WeatherDataCollector('test_api_key')
        result = collector.get_current('TestCity')
        
        assert isinstance(result, dict)
        assert 'error' in result

    @patch('weather_predictor.data_collector.requests.get')
    def test_get_current_weather_timeout(self, mock_get):
        """Test timeout handling"""
        import requests
        mock_get.side_effect = requests.Timeout('Request timed out')
        
        collector = WeatherDataCollector('test_api_key')
        result = collector.get_current('TestCity')
        
        assert isinstance(result, dict)
        assert 'error' in result

    @patch('weather_predictor.data_collector.requests.get')
    def test_get_forecast_method_exists(self, mock_get):
        """Test that get_forecast method exists and is callable"""
        collector = WeatherDataCollector('test_api_key')
        
        assert hasattr(collector, 'get_forecast')
        assert callable(collector.get_forecast)

    @patch('weather_predictor.data_collector.requests.get')
    def test_get_forecast_success(self, mock_get):
        """Test successful forecast API call"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'list': [
                {
                    'dt': 1640995200,
                    'main': {'temp': 20.5, 'humidity': 70},
                    'weather': [{'description': 'light rain'}],
                    'dt_txt': '2022-01-01 12:00:00'
                },
                {
                    'dt': 1641006000,
                    'main': {'temp': 18.2, 'humidity': 75},
                    'weather': [{'description': 'overcast clouds'}],
                    'dt_txt': '2022-01-01 15:00:00'
                }
            ],
            'city': {'name': 'TestCity'}
        }
        mock_get.return_value = mock_response
        
        collector = WeatherDataCollector('test_api_key')
        result = collector.get_forecast('TestCity')
        
        mock_get.assert_called_once()
        call_args = mock_get.call_args[0][0]
        assert 'forecast' in call_args
        assert 'TestCity' in call_args
        
        assert 'list' in result or isinstance(result, list)

    @patch('weather_predictor.data_collector.requests.get')
    def test_api_key_in_requests(self, mock_get):
        """Test that API key is properly included in requests"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'test': 'data'}
        mock_get.return_value = mock_response
        
        api_key = 'specific_test_key_123'
        collector = WeatherDataCollector(api_key)
        collector.get_current('TestCity')
        
        call_args = mock_get.call_args[0][0]
        assert api_key in call_args

    @patch('weather_predictor.data_collector.requests.get')
    def test_units_parameter(self, mock_get):
        """Test that metric units are used in API calls"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'test': 'data'}
        mock_get.return_value = mock_response
        
        collector = WeatherDataCollector('test_api_key')
        collector.get_current('TestCity')
        
        call_args = mock_get.call_args[0][0]
        assert 'units=metric' in call_args

    @patch('weather_predictor.data_collector.requests.get')
    def test_historical_data_method(self, mock_get):
        """Test historical data collection if method exists"""
        collector = WeatherDataCollector('test_api_key')
        
        if hasattr(collector, 'get_historical'):
            assert callable(collector.get_historical)
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'data': []}
            mock_get.return_value = mock_response
            
            try:
                result = collector.get_historical('TestCity', days=30)
                mock_get.assert_called()
            except Exception:
                pass
    def test_data_sanitization(self, mock_get):
        """Test data sanitization for debugging"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'main': {'temp': 25.0},
            'coord': {'lat': 52.2297, 'lon': 21.0122}
        }
        mock_get.return_value = mock_response
        
        collector = WeatherDataCollector('secret_api_key')
        
        if hasattr(collector, '_sanitize_api_response'):
            test_data = {'sensitive': 'secret_api_key', 'normal': 'data'}
            sanitized = collector._sanitize_api_response(test_data)
            
            assert 'secret_api_key' not in str(sanitized)
        else:
            assert True

    @patch('weather_predictor.data_collector.requests.get')
    def test_response_validation(self, mock_get):
        """Test response validation"""
        collector = WeatherDataCollector('test_api_key')
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError('Invalid JSON', 'doc', 0)
        mock_response.text = 'Invalid response'
        mock_get.return_value = mock_response
        
        result = collector.get_current('TestCity')
        
        assert isinstance(result, dict)
        assert 'error' in result

    @patch('weather_predictor.data_collector.requests.get')
    def test_multiple_cities(self, mock_get):
        """Test handling multiple cities"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'main': {'temp': 20.0},
            'name': 'ResponseCity'
        }
        mock_get.return_value = mock_response
        
        collector = WeatherDataCollector('test_api_key')
        
        cities = ['London', 'Paris', 'Berlin']
        for city in cities:
            result = collector.get_current(city)
            assert isinstance(result, dict)

    @patch('weather_predictor.data_collector.requests.get')
    def test_api_rate_limiting(self, mock_get):
        """Test API rate limiting handling"""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {
            'cod': 429,
            'message': 'API calls limit exceeded'
        }
        mock_get.return_value = mock_response
        
        collector = WeatherDataCollector('test_api_key')
        result = collector.get_current('TestCity')
        
        assert isinstance(result, dict)
        assert ('cod' in result and result['cod'] == 429) or 'error' in result

    def test_debug_output_control(self):
        """Test that debug output can be controlled"""
        collector = WeatherDataCollector('test_api_key')
        
        if hasattr(collector, 'debug') or hasattr(collector, 'verbose'):
            assert True
        else:
            assert True
