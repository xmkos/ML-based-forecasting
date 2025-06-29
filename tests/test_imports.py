# Tests for importability of modules

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_weather_predictor_import():
    """Test that WeatherPredictor can be imported and initialized"""
    try:
        from weather_predictor.predictor import WeatherPredictor
        assert hasattr(WeatherPredictor, '__init__')
        assert hasattr(WeatherPredictor, 'get_current_weather')
        assert hasattr(WeatherPredictor, 'train_model')
        assert hasattr(WeatherPredictor, 'predict_weather')
    except ImportError as e:
        pytest.fail(f"Failed to import WeatherPredictor: {e}")

def test_ml_model_import():
    """Test that WeatherMLPredictor can be imported"""
    try:
        from weather_predictor.ml_model import WeatherMLPredictor
        assert hasattr(WeatherMLPredictor, '__init__')
        assert hasattr(WeatherMLPredictor, 'train_model')
        assert hasattr(WeatherMLPredictor, 'predict_weather') or hasattr(WeatherMLPredictor, 'predict_tomorrow_weather')
        assert hasattr(WeatherMLPredictor, 'MODEL_TYPES')
        assert 'rf' in WeatherMLPredictor.MODEL_TYPES
        assert 'gb' in WeatherMLPredictor.MODEL_TYPES
        assert 'nn' in WeatherMLPredictor.MODEL_TYPES
    except ImportError as e:
        pytest.fail(f"Failed to import WeatherMLPredictor: {e}")

def test_data_collector_import():
    """Test that WeatherDataCollector can be imported"""
    try:
        from weather_predictor.data_collector import WeatherDataCollector
        assert hasattr(WeatherDataCollector, '__init__')
        assert hasattr(WeatherDataCollector, 'get_current')
        assert hasattr(WeatherDataCollector, 'get_forecast')
    except ImportError as e:
        pytest.fail(f"Failed to import WeatherDataCollector: {e}")

def test_cli_import():
    """Test that WeatherCLI can be imported"""
    try:
        from weather_predictor.cli import WeatherCLI
        assert hasattr(WeatherCLI, '__init__')
        assert hasattr(WeatherCLI, 'run')
    except ImportError as e:
        pytest.fail(f"Failed to import WeatherCLI: {e}")

def test_gui_import():
    """Test that WeatherGUI can be imported"""
    try:
        from weather_predictor.gui import WeatherGUI
        assert hasattr(WeatherGUI, '__init__')
    except ImportError as e:
        pytest.fail(f"Failed to import WeatherGUI: {e}")

def test_exceptions_import():
    """Test that custom exceptions can be imported"""
    try:
        from weather_predictor.exceptions import DataError, ModelLoadError, DownloadError
        assert issubclass(DataError, Exception)
        assert issubclass(ModelLoadError, Exception)
        assert issubclass(DownloadError, Exception)
    except ImportError as e:
        pytest.fail(f"Failed to import exceptions: {e}")

def test_config_import():
    """Test that config can be imported"""
    try:
        from weather_predictor.config import API_KEY
        assert API_KEY is None or isinstance(API_KEY, str)
    except ImportError as e:
        pytest.fail(f"Failed to import config: {e}")

def test_package_init():
    """Test that the main package can be imported"""
    try:
        import weather_predictor
        from weather_predictor import WeatherPredictor, WeatherMLPredictor, WeatherDataCollector
        assert WeatherPredictor is not None
        assert WeatherMLPredictor is not None
        assert WeatherDataCollector is not None
    except ImportError as e:
        pytest.fail(f"Failed to import weather_predictor package: {e}")

def test_required_dependencies():
    """Test that required dependencies are available"""
    required_modules = [
        'pandas', 'numpy', 'sklearn', 'requests', 
        'joblib', 'tkinter', 'geocoder', 'dotenv'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            if module == 'sklearn':
                import sklearn
            elif module == 'dotenv':
                import dotenv
            else:
                __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        pytest.fail(f"Missing required dependencies: {missing_modules}")

def test_ml_model_types():
    """Test that ML model types are properly defined"""
    from weather_predictor.ml_model import WeatherMLPredictor
    
    model_types = WeatherMLPredictor.MODEL_TYPES
    assert isinstance(model_types, dict)
    
    expected_types = ['rf', 'gb', 'ridge', 'elastic', 'svr', 'nn']
    for model_type in expected_types:
        assert model_type in model_types
        assert len(model_types[model_type]) == 2
        assert isinstance(model_types[model_type][0], str)
