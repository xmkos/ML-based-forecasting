# File: tests/test_cli.py
# Unit tests for CLI functionality

import subprocess
import sys
import os
import pytest
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

PY = sys.executable
SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Unique_weather_predictor.py'))

def run_cmd(args, timeout=30):
    """Run command with timeout to prevent hanging"""
    try:
        result = subprocess.run([PY, SCRIPT] + args, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return -1, "Command timed out"

def test_help_command():
    """Test that help command works"""
    code, out = run_cmd(['--help'])
    assert code == 0
    assert 'Weather Prediction System' in out
    assert '--mode' in out
    assert '--city' in out
    assert '--days' in out
    assert '--model' in out

def test_invalid_mode():
    """Test that invalid mode shows error"""
    code, out = run_cmd(['--mode', 'invalid'])
    assert code != 0
    assert 'invalid choice' in out.lower()

def test_valid_modes():
    """Test that valid modes are accepted"""
    valid_modes = ['cli', 'gui', 'demo']
    for mode in valid_modes:
        # Just check that the mode is accepted (might fail due to missing API key)
        code, out = run_cmd(['--mode', mode, '--help'])
        # Help should work regardless of API key
        if '--help' in ' '.join(['--mode', mode, '--help']):
            assert code == 0

def test_demo_mode_parameters():
    """Test demo mode parameter validation"""
    # Test valid model types
    valid_models = ['rf', 'gb', 'nn']
    for model in valid_models:
        code, out = run_cmd(['--mode', 'demo', '--model', model, '--help'])
        assert code == 0

    # Test invalid model type
    code, out = run_cmd(['--mode', 'demo', '--model', 'invalid'])
    assert code != 0
    assert 'invalid choice' in out.lower()

def test_days_parameter():
    """Test days parameter validation"""
    # Test valid days
    code, out = run_cmd(['--mode', 'demo', '--days', '30', '--help'])
    assert code == 0
    
    # Test invalid days (not a number)
    code, out = run_cmd(['--mode', 'demo', '--days', 'invalid'])
    assert code != 0

@patch('weather_predictor.config.API_KEY', 'test_api_key')
def test_cli_mode_without_api():
    """Test CLI mode behavior when API is mocked"""
    from weather_predictor.cli import WeatherCLI
    
    cli = WeatherCLI()
    assert hasattr(cli, 'run')
    assert hasattr(cli, 'predictor')

@patch('weather_predictor.config.API_KEY', None)
def test_cli_mode_missing_api():
    """Test CLI mode behavior when API key is missing"""
    from weather_predictor.exceptions import DataError
    
    with pytest.raises(DataError):
        from weather_predictor.cli import WeatherCLI
        WeatherCLI()

def test_cli_commands_structure():
    """Test that CLI has proper command structure"""
    from weather_predictor.cli import WeatherCLI
    
    # Mock the API key to avoid initialization errors
    with patch('weather_predictor.config.API_KEY', 'test_key'):
        with patch('weather_predictor.predictor.WeatherPredictor'):
            cli = WeatherCLI()
            
            # Test that run method exists
            assert hasattr(cli, 'run')
            
            # Test run with help to see command structure
            try:
                exit_code = cli.run(['--help'])
                # Should return 0 for help or raise SystemExit
            except SystemExit as e:
                assert e.code == 0

def test_main_function_exists():
    """Test that main function exists and is callable"""
    # Import the main script
    import importlib.util
    spec = importlib.util.spec_from_file_location("main_script", SCRIPT)
    main_module = importlib.util.module_from_spec(spec)
    
    # Check if main function exists
    assert hasattr(main_module, 'main')
    assert callable(main_module.main)

def test_script_execution_modes():
    """Test that script accepts all documented modes"""
    # Test that script doesn't crash immediately with each mode
    modes = ['cli', 'gui', 'demo']
    
    for mode in modes:
        # Use very short timeout and expect either success or specific error
        code, out = run_cmd(['--mode', mode, '--help'], timeout=10)
        # Should either work (code 0) or fail with specific error (not crash)
        assert code in [0, 2]  # 0 = success, 2 = argument error

def test_force_retrain_flag():
    """Test force retrain flag"""
    code, out = run_cmd(['--mode', 'demo', '--force-retrain', '--help'])
    assert code == 0

def test_city_parameter():
    """Test city parameter"""
    code, out = run_cmd(['--mode', 'demo', '--city', 'TestCity', '--help'])
    assert code == 0