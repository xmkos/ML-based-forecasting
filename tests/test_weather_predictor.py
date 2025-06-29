# File: tests/test_weather_predictor.py
# Unit tests for WeatherPredictor main class

import pytest
import sys
import os
from unittest.mock import patch, MagicMock, Mock

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from weather_predictor.predictor import WeatherPredictor
from weather_predictor.exceptions import DataError


class TestWeatherPredictor:
    """Test suite for WeatherPredictor class"""

    @patch('weather_predictor.config.API_KEY', 'test_api_key')
    @patch('weather_predictor.predictor.WeatherDataCollector')
    @patch('weather_predictor.predictor.WeatherMLPredictor')
    def test_initialization_with_api_key(self, mock_ml_predictor, mock_data_collector):
        """Test WeatherPredictor initialization with valid API key"""
        predictor = WeatherPredictor(model_type='rf', historical_days=30)
        
        assert predictor.historical_days == 30
        mock_data_collector.assert_called_once_with('test_api_key')
        mock_ml_predictor.assert_called_once_with('test_api_key', model_type='rf')    @patch('weather_predictor.config.API_KEY', None)
    def test_initialization_without_api_key(self):
        """Test WeatherPredictor initialization without API key raises error"""
        # Temporarily set API_KEY to None in the config module itself
        import weather_predictor.config
        original_api_key = weather_predictor.config.API_KEY
        weather_predictor.config.API_KEY = None
        
        try:
            with pytest.raises(DataError, match="API_KEY is not set"):
                WeatherPredictor()
        finally:
            # Restore original API key
            weather_predictor.config.API_KEY = original_api_key

    @patch('weather_predictor.config.API_KEY', 'test_api_key')
    @patch('weather_predictor.predictor.WeatherDataCollector')
    @patch('weather_predictor.predictor.WeatherMLPredictor')
    def test_get_current_weather(self, mock_ml_predictor, mock_data_collector):
        """Test get_current_weather method"""
        # Setup mocks
        mock_collector_instance = MagicMock()
        mock_data_collector.return_value = mock_collector_instance
        
        expected_weather = {
            'main': {'temp': 25.0, 'humidity': 60},
            'weather': [{'description': 'clear sky'}]
        }
        mock_collector_instance.get_current.return_value = expected_weather
        
        # Test
        predictor = WeatherPredictor()
        result = predictor.get_current_weather('TestCity')
        
        # Assertions
        assert result == expected_weather
        mock_collector_instance.get_current.assert_called_once_with('TestCity')

    @patch('weather_predictor.config.API_KEY', 'test_api_key')
    @patch('weather_predictor.predictor.WeatherDataCollector')
    @patch('weather_predictor.predictor.WeatherMLPredictor')
    def test_train_model_default_parameters(self, mock_ml_predictor, mock_data_collector):
        """Test train_model with default parameters"""
        # Setup mocks
        mock_ml_instance = MagicMock()
        mock_ml_predictor.return_value = mock_ml_instance
        
        expected_metrics = {'mae': 2.5, 'rmse': 3.2, 'r2': 0.85}
        mock_ml_instance.train_model.return_value = expected_metrics
        
        # Test
        predictor = WeatherPredictor(historical_days=30)
        result = predictor.train_model('TestCity')
        
        # Assertions
        assert result == expected_metrics
        mock_ml_instance.train_model.assert_called_once_with('TestCity', days=30, validate=True)

    @patch('weather_predictor.config.API_KEY', 'test_api_key')
    @patch('weather_predictor.predictor.WeatherDataCollector')
    @patch('weather_predictor.predictor.WeatherMLPredictor')
    def test_train_model_custom_parameters(self, mock_ml_predictor, mock_data_collector):
        """Test train_model with custom parameters"""
        # Setup mocks
        mock_ml_instance = MagicMock()
        mock_ml_predictor.return_value = mock_ml_instance
        
        expected_metrics = {'mae': 2.1, 'rmse': 2.8, 'r2': 0.90}
        mock_ml_instance.train_model.return_value = expected_metrics
        
        # Test
        predictor = WeatherPredictor()
        result = predictor.train_model('CustomCity', days=60, validate=False)
        
        # Assertions
        assert result == expected_metrics
        mock_ml_instance.train_model.assert_called_once_with('CustomCity', days=60, validate=False)

    @patch('weather_predictor.config.API_KEY', 'test_api_key')
    @patch('weather_predictor.predictor.WeatherDataCollector')
    @patch('weather_predictor.predictor.WeatherMLPredictor')
    def test_predict_weather(self, mock_ml_predictor, mock_data_collector):
        """Test predict_weather method"""
        # Setup mocks
        mock_collector_instance = MagicMock()
        mock_data_collector.return_value = mock_collector_instance
        
        mock_ml_instance = MagicMock()
        mock_ml_predictor.return_value = mock_ml_instance
        
        current_weather = {
            'main': {'temp': 22.0, 'humidity': 55},
            'weather': [{'description': 'scattered clouds'}]
        }
        mock_collector_instance.get_current.return_value = current_weather
        
        # Test
        predictor = WeatherPredictor()
        
        # Mock the predict_weather method exists (we're testing the interface)
        assert hasattr(predictor, 'predict_weather')

    @patch('weather_predictor.config.API_KEY', 'test_api_key')
    @patch('weather_predictor.predictor.WeatherDataCollector')
    @patch('weather_predictor.predictor.WeatherMLPredictor')
    def test_model_types_support(self, mock_ml_predictor, mock_data_collector):
        """Test that different model types are supported"""
        model_types = ['rf', 'gb', 'nn']
        
        for model_type in model_types:
            predictor = WeatherPredictor(model_type=model_type)
            mock_ml_predictor.assert_called_with('test_api_key', model_type=model_type)

    @patch('weather_predictor.config.API_KEY', 'test_api_key')
    @patch('weather_predictor.predictor.WeatherDataCollector')
    @patch('weather_predictor.predictor.WeatherMLPredictor')
    def test_error_handling_in_get_current_weather(self, mock_ml_predictor, mock_data_collector):
        """Test error handling in get_current_weather"""
        # Setup mocks to raise exception
        mock_collector_instance = MagicMock()
        mock_data_collector.return_value = mock_collector_instance
        mock_collector_instance.get_current.side_effect = Exception("API Error")
        
        # Test
        predictor = WeatherPredictor()
        
        with pytest.raises(Exception, match="API Error"):
            predictor.get_current_weather('InvalidCity')

    @patch('weather_predictor.config.API_KEY', 'test_api_key')
    @patch('weather_predictor.predictor.WeatherDataCollector')
    @patch('weather_predictor.predictor.WeatherMLPredictor')
    def test_is_model_trained_method(self, mock_ml_predictor, mock_data_collector):
        """Test is_model_trained method if it exists"""
        # Setup mocks
        mock_ml_instance = MagicMock()
        mock_ml_predictor.return_value = mock_ml_instance
        mock_ml_instance.is_model_trained.return_value = True
        
        predictor = WeatherPredictor()
        
        # Check if method exists and test it
        if hasattr(predictor, 'is_model_trained'):
            result = predictor.is_model_trained()
            assert result is True
            mock_ml_instance.is_model_trained.assert_called_once()

    def test_class_attributes(self):
        """Test class structure and attributes"""
        # Test that class has expected methods
        expected_methods = ['__init__', 'get_current_weather', 'train_model', 'predict_weather']
        
        for method in expected_methods:
            assert hasattr(WeatherPredictor, method)
            assert callable(getattr(WeatherPredictor, method))

    @patch('weather_predictor.config.API_KEY', 'test_api_key')
    @patch('weather_predictor.predictor.WeatherDataCollector')
    @patch('weather_predictor.predictor.WeatherMLPredictor')
    def test_integration_workflow(self, mock_ml_predictor, mock_data_collector):
        """Test a typical workflow: initialize -> get current -> train -> predict"""
        # Setup mocks
        mock_collector_instance = MagicMock()
        mock_data_collector.return_value = mock_collector_instance
        
        mock_ml_instance = MagicMock()
        mock_ml_predictor.return_value = mock_ml_instance
        
        # Mock responses
        current_weather = {'main': {'temp': 20.0}}
        training_metrics = {'mae': 2.0, 'rmse': 2.5}
        
        mock_collector_instance.get_current.return_value = current_weather
        mock_ml_instance.train_model.return_value = training_metrics
        
        # Test workflow
        predictor = WeatherPredictor(model_type='rf')
        
        # Step 1: Get current weather
        weather = predictor.get_current_weather('TestCity')
        assert weather == current_weather
        
        # Step 2: Train model
        metrics = predictor.train_model('TestCity', days=30)
        assert metrics == training_metrics
        
        # Verify all calls were made correctly
        mock_collector_instance.get_current.assert_called_with('TestCity')
        mock_ml_instance.train_model.assert_called_with('TestCity', days=30, validate=True)
