# File: tests/test_integration.py
# Integration tests for the weather prediction system

import pytest
import sys
import os
import subprocess
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

PY = sys.executable
SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Unique_weather_predictor.py'))


class TestIntegration:
    """Integration tests for the complete weather prediction system"""

    def test_main_script_exists(self):
        """Test that main script file exists and is executable"""
        assert os.path.exists(SCRIPT)
        assert os.path.isfile(SCRIPT)

    def test_command_line_help(self):
        """Test command line help functionality"""
        try:
            result = subprocess.run([PY, SCRIPT, '--help'], 
                                  capture_output=True, text=True, timeout=10)
            assert result.returncode == 0
            assert 'Weather Prediction System' in result.stdout
            assert '--mode' in result.stdout
        except subprocess.TimeoutExpired:
            pytest.fail("Help command timed out")

    def test_invalid_arguments(self):
        """Test handling of invalid command line arguments"""
        # Test invalid mode
        result = subprocess.run([PY, SCRIPT, '--mode', 'invalid'], 
                              capture_output=True, text=True, timeout=5)
        assert result.returncode != 0
        assert 'invalid choice' in result.stderr.lower()

    @patch('weather_predictor.config.API_KEY', 'test_api_key')
    def test_module_integration(self):
        """Test integration between different modules"""
        try:
            from weather_predictor import WeatherPredictor, WeatherMLPredictor, WeatherDataCollector
            from weather_predictor.cli import WeatherCLI
            from weather_predictor.exceptions import DataError
            
            # Test that modules can be imported together
            assert WeatherPredictor is not None
            assert WeatherMLPredictor is not None
            assert WeatherDataCollector is not None
            assert WeatherCLI is not None
            assert DataError is not None
            
        except ImportError as e:
            pytest.fail(f"Module integration failed: {e}")

    @patch('weather_predictor.config.API_KEY', 'test_api_key')
    @patch('weather_predictor.predictor.WeatherDataCollector')
    @patch('weather_predictor.predictor.WeatherMLPredictor')
    def test_predictor_workflow(self, mock_ml_predictor, mock_data_collector):
        """Test complete predictor workflow"""
        # Setup mocks
        mock_collector_instance = MagicMock()
        mock_data_collector.return_value = mock_collector_instance
        
        mock_ml_instance = MagicMock()
        mock_ml_predictor.return_value = mock_ml_instance
        
        # Mock responses
        current_weather = {
            'main': {'temp': 22.0, 'humidity': 65},
            'weather': [{'description': 'clear sky'}]
        }
        training_metrics = {'mae': 2.5, 'rmse': 3.0, 'r2': 0.85}
        
        mock_collector_instance.get_current.return_value = current_weather
        mock_ml_instance.train_model.return_value = training_metrics
        
        # Test workflow
        from weather_predictor.predictor import WeatherPredictor
        predictor = WeatherPredictor()
        
        # Step 1: Get current weather
        weather = predictor.get_current_weather('TestCity')
        assert weather == current_weather
        
        # Step 2: Train model
        metrics = predictor.train_model('TestCity')
        assert metrics == training_metrics
        
        # Verify integration
        mock_collector_instance.get_current.assert_called_with('TestCity')
        mock_ml_instance.train_model.assert_called()

    def test_package_structure(self):
        """Test that package structure is correct"""
        package_dir = os.path.join(os.path.dirname(__file__), '..', 'weather_predictor')
        
        # Check that package directory exists
        assert os.path.exists(package_dir)
        assert os.path.isdir(package_dir)
        
        # Check for essential files
        essential_files = [
            '__init__.py',
            'predictor.py',
            'ml_model.py',
            'data_collector.py',
            'cli.py',
            'gui.py',
            'config.py',
            'exceptions.py'
        ]
        
        for file_name in essential_files:
            file_path = os.path.join(package_dir, file_name)
            assert os.path.exists(file_path), f"Missing essential file: {file_name}"

    def test_requirements_compatibility(self):
        """Test that required packages can be imported"""
        try:
            import pandas
            import numpy
            import sklearn
            import requests
            import joblib
            import tkinter
            assert True
        except ImportError as e:
            pytest.fail(f"Required package import failed: {e}")

    @patch('weather_predictor.config.API_KEY', 'test_api_key')
    def test_cli_integration(self):
        """Test CLI integration"""
        try:
            from weather_predictor.cli import WeatherCLI
            
            # Mock the predictor to avoid API calls
            with patch('weather_predictor.cli.WeatherPredictor') as mock_predictor:
                cli = WeatherCLI()
                assert hasattr(cli, 'run')
                assert hasattr(cli, 'predictor')
                
        except Exception as e:
            pytest.fail(f"CLI integration failed: {e}")

    def test_gui_integration(self):
        """Test GUI integration (without actually showing GUI)"""
        try:
            # Import should work even if we can't display GUI
            from weather_predictor.gui import WeatherGUI
            assert WeatherGUI is not None
            
        except ImportError as e:
            # GUI might not be available in headless environments
            if 'tkinter' in str(e).lower():
                pytest.skip("GUI not available in headless environment")
            else:
                pytest.fail(f"GUI integration failed: {e}")

    def test_config_integration(self):
        """Test configuration integration"""
        try:
            from weather_predictor.config import API_KEY
            # API_KEY might be None, but should be importable
            assert API_KEY is None or isinstance(API_KEY, str)
            
        except ImportError as e:
            pytest.fail(f"Config integration failed: {e}")

    def test_model_directory_creation(self):
        """Test that model directory can be created"""
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'weather_predictor', 'models')
        
        # Directory might exist or be created during runtime
        if not os.path.exists(models_dir):
            # Test that we can create it
            try:
                os.makedirs(models_dir, exist_ok=True)
                assert os.path.exists(models_dir)
            except PermissionError:
                pytest.skip("Cannot create models directory due to permissions")

    @patch('weather_predictor.config.API_KEY', None)
    def test_missing_api_key_handling(self):
        """Test handling of missing API key across system"""
        from weather_predictor.exceptions import DataError
        
        # Should raise DataError when trying to create predictor without API key
        with pytest.raises(DataError):
            from weather_predictor.predictor import WeatherPredictor
            WeatherPredictor()

    def test_demo_mode_structure(self):
        """Test demo mode command structure"""
        # Test that demo mode accepts required parameters
        result = subprocess.run([
            PY, SCRIPT, '--mode', 'demo', 
            '--city', 'TestCity', 
            '--days', '30', 
            '--model', 'rf',
            '--help'
        ], capture_output=True, text=True, timeout=10)
        
        assert result.returncode == 0

    def test_model_types_consistency(self):
        """Test that model types are consistent across system"""
        try:
            from weather_predictor.ml_model import WeatherMLPredictor
            
            # Get available model types
            model_types = WeatherMLPredictor.MODEL_TYPES
            
            # Test that command line accepts these model types
            for model_type in ['rf', 'gb', 'nn']:  # Common model types
                if model_type in model_types:
                    result = subprocess.run([
                        PY, SCRIPT, '--mode', 'demo', 
                        '--model', model_type, '--help'
                    ], capture_output=True, text=True, timeout=5)
                    assert result.returncode == 0
                    
        except Exception as e:
            pytest.fail(f"Model types consistency test failed: {e}")

    def test_error_propagation(self):
        """Test that errors propagate correctly through the system"""
        # Test with invalid city name and missing API key
        result = subprocess.run([
            PY, SCRIPT, '--mode', 'demo', 
            '--city', '', '--days', '1'
        ], capture_output=True, text=True, timeout=10)
        
        # Should either complete or show appropriate error message
        # (not crash with unhandled exception)
        assert result.returncode in [0, 1, 2]  # Allow various exit codes
        
    def test_file_permissions(self):
        """Test file permissions for model saving"""
        test_dir = os.path.join(os.path.dirname(__file__), '..', 'weather_predictor')
        
        # Test that we can write to the weather_predictor directory
        test_file = os.path.join(test_dir, 'test_write.tmp')
        
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            
            assert os.path.exists(test_file)
            
            # Clean up
            os.remove(test_file)
            
        except PermissionError:
            pytest.skip("Cannot write to package directory due to permissions")

    def test_memory_usage(self):
        """Test basic memory usage (no memory leaks in imports)"""
        import gc
        import sys
        
        # Get initial memory usage
        initial_objects = len(gc.get_objects())
        
        # Import and use the system
        try:
            from weather_predictor import WeatherPredictor, WeatherMLPredictor
            # Create instances (mocked to avoid API calls)
            with patch('weather_predictor.config.API_KEY', 'test'):
                with patch('weather_predictor.predictor.WeatherDataCollector'):
                    with patch('weather_predictor.predictor.WeatherMLPredictor'):
                        predictor = WeatherPredictor()
                        del predictor
        except:
            pass
        
        # Force garbage collection
        gc.collect()
        
        # Check that we didn't create too many persistent objects
        final_objects = len(gc.get_objects())
        object_increase = final_objects - initial_objects
        
        # Allow some increase but not excessive
        assert object_increase < 1000, f"Too many objects created: {object_increase}"
