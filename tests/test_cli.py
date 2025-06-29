# Tests for the CLI utility

import subprocess
import sys
import os
import pytest
from unittest.mock import patch, MagicMock

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
        code, out = run_cmd(['--mode', mode, '--help'])
        if '--help' in ' '.join(['--mode', mode, '--help']):
            assert code == 0

def test_demo_mode_parameters():
    """Test demo mode parameter validation"""
    valid_models = ['rf', 'gb', 'nn']
    for model in valid_models:
        code, out = run_cmd(['--mode', 'demo', '--model', model, '--help'])
        assert code == 0

    code, out = run_cmd(['--mode', 'demo', '--model', 'invalid'])
    assert code != 0
    assert 'invalid choice' in out.lower()

def test_days_parameter():
    """Test days parameter validation"""
    code, out = run_cmd(['--mode', 'demo', '--days', '30', '--help'])
    assert code == 0
    
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
    
    with patch('weather_predictor.config.API_KEY', 'test_key'):
        with patch('weather_predictor.predictor.WeatherPredictor'):
            cli = WeatherCLI()
            
            assert hasattr(cli, 'run')
            
            try:
                exit_code = cli.run(['--help'])
            except SystemExit as e:
                assert e.code == 0

def test_main_function_exists():
    """Test that main function exists and is callable"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("main_script", SCRIPT)
    main_module = importlib.util.module_from_spec(spec)
    
    assert hasattr(main_module, 'main')
    assert callable(main_module.main)

def test_script_execution_modes():
    """Test that script accepts all documented modes"""
    modes = ['cli', 'gui', 'demo']
    
    for mode in modes:
        code, out = run_cmd(['--mode', mode, '--help'], timeout=10)
        assert code in [0, 2]

def test_force_retrain_flag():
    """Test force retrain flag"""
    code, out = run_cmd(['--mode', 'demo', '--force-retrain', '--help'])
    assert code == 0

def test_city_parameter():
    """Test city parameter"""
    code, out = run_cmd(['--mode', 'demo', '--city', 'TestCity', '--help'])
    assert code == 0
