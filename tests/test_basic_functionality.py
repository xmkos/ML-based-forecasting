# Basic sanity tests for public interfaces

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))



import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_imports_work():
    """Test that all main classes can be imported"""
    try:
        from weather_predictor.predictor import WeatherPredictor
        from weather_predictor.ml_model import WeatherMLPredictor
        from weather_predictor.data_collector import WeatherDataCollector
        from weather_predictor.cli import WeatherCLI
        from weather_predictor.exceptions import DataError, ModelLoadError, DownloadError
        
        assert all([
            WeatherPredictor, WeatherMLPredictor, WeatherDataCollector,
            WeatherCLI, DataError, ModelLoadError, DownloadError
        ])
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_weather_predictor_basic():
    """Test basic WeatherPredictor functionality"""
    from weather_predictor.config import API_KEY
    if not API_KEY:
        pytest.skip("No API key available")
    
    from weather_predictor.predictor import WeatherPredictor
    
    try:
        predictor = WeatherPredictor()
        assert hasattr(predictor, 'get_current_weather')
        assert hasattr(predictor, 'train_model')
        assert hasattr(predictor, 'predict_weather')
    except Exception as e:
        pytest.fail(f"WeatherPredictor basic test failed: {e}")


def test_ml_model_basic():
    """Test basic ML model functionality"""
    from weather_predictor.ml_model import WeatherMLPredictor
    
    assert hasattr(WeatherMLPredictor, 'MODEL_TYPES')
    model_types = WeatherMLPredictor.MODEL_TYPES
    assert 'rf' in model_types
    assert 'gb' in model_types
    
    with patch('weather_predictor.ml_model.requests'):
        try:
            predictor = WeatherMLPredictor('test_key', model_type='rf')
            assert predictor.model_type == 'rf'
            assert predictor.api_key == 'test_key'
        except Exception as e:
            pytest.fail(f"ML model basic test failed: {e}")


def test_main_script_structure():
    """Test main script structure"""
    script_path = os.path.join(os.path.dirname(__file__), '..', 'Unique_weather_predictor.py')
    assert os.path.exists(script_path)
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("main_script", script_path)
    main_module = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(main_module)
        assert hasattr(main_module, 'main')
        assert hasattr(main_module, 'run_ml_demo')
    except Exception as e:
        pytest.fail(f"Main script structure test failed: {e}")
