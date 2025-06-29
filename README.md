# Weather Prediction System

## Project Description

### Goal
The goal of this project is to develop an advanced weather prediction tool that enables users to obtain accurate weather forecasts using machine learning algorithms and real-time weather data from APIs.

### Scope
The project encompasses the following aspects:
- Real-time weather data collection via OpenWeatherMap API
- Machine learning algorithms for weather prediction (Random Forest, Gradient Boosting, Ridge Regression, Elastic Net, Support Vector Machine, Neural Networks)
- Multiple user interfaces: CLI, GUI, and demo modes
- Data preprocessing and feature engineering
- Model training and evaluation
- Clear and precise prediction results

### Expected Results
By the end of this project, I expect to achieve:
- A fully functional weather prediction system with multiple ML models
- Accurate and efficient weather prediction results
- Intuitive and user-friendly interfaces (both command-line and graphical)
- Robust error handling and data validation
- Comprehensive testing coverage

### Technologies Used
- **Python 3.9+**, **scikit-learn**, **pandas**, **numpy**, **tkinter**, **requests**

## Installation and Setup

### Prerequisites
- Python 3.9+ and [OpenWeatherMap API key](https://openweathermap.org/api)

### Quick Setup
```powershell
# Create and activate virtual environment
python -m venv env
.\env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file with your API key
echo "API_KEY=your_api_key_here" > .env
```

### Usage
```powershell
# GUI Mode (default)
python Unique_weather_predictor.py

# CLI Mode
python Unique_weather_predictor.py --mode cli

# Demo Mode
python Unique_weather_predictor.py --mode demo --city "Warsaw" --model rf
```

**Models**: `rf` (Random Forest), `gb` (Gradient Boosting), `ridge`, `elastic`, `svr`, `nn`

## Testing

```powershell
# Quick test run
python run_tests.py

# Run specific tests
python -m pytest tests/test_ml_model.py -v
python -m pytest tests/ --cov=weather_predictor
```

**Coverage**: ~103 tests covering ML models, API integration, CLI/GUI, and error handling.

## Troubleshooting

```powershell
# API key issues - check .env file
# Module errors - reinstall: pip install -r requirements.txt
# Test failures - run: python run_tests.py
```


