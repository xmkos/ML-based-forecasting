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
- **Python 3.9+** - Main programming language
- **scikit-learn** - Machine learning framework for RF, GB, Ridge, Elastic Net, SVR, and Neural Network models
- **pandas & numpy** - Data manipulation and numerical computing
- **tkinter** - GUI framework
- **requests** - HTTP API communication
- **joblib** - Model serialization
- **python-dotenv** - Environment variable management
- **geocoder** - Location services

## Requirements Analysis

### Functional Requirements
- Fetch current weather data from OpenWeatherMap API
- Train machine learning models using historical weather data
- Predict weather conditions for specified locations
- Support multiple ML algorithms (Random Forest, Gradient Boosting, Ridge Regression, Elastic Net, Support Vector Machine, Neural Networks)
- Provide both CLI and GUI interfaces
- Save and load trained models
- Display prediction accuracy metrics
- Multi-parameter prediction (temperature, humidity, wind speed)
- Comprehensive error handling and fallback mechanisms

### Non-functional Requirements
- Response time under 5 seconds for predictions
- Support for multiple cities and locations
- Error handling for API failures and invalid inputs
- Data validation and preprocessing
- Modular code architecture
- Cross-platform compatibility (Windows, Linux, macOS)

### User Interface and System Functionality
- **CLI Mode**: Command-line interface for advanced users
- **GUI Mode**: Graphical interface with buttons and forms
- **Demo Mode**: Standalone ML demonstration with detailed output
- Real-time weather data visualization
- Model performance metrics display

## Installation and Usage Guide

### Prerequisites
1. Install Python 3.9.0 or higher from [Python Official Website](https://www.python.org/downloads/)
2. Obtain an API key from [OpenWeatherMap](https://openweathermap.org/api)

### Installation Steps

#### Method 1: Direct Installation
```powershell
pip install -r requirements.txt
```

#### Method 2: Using Virtual Environment (Recommended)
```powershell
# Create virtual environment
python -m venv env

# Activate virtual environment
.\env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration
1. Create a `.env` file in the project root directory
2. Add your OpenWeatherMap API key (replace with your actual key):
```
API_KEY=your_actual_api_key_without_brackets
```

### Running the Application

#### GUI Mode (Default)
```powershell
python Unique_weather_predictor.py
```

#### CLI Mode
```powershell
python Unique_weather_predictor.py --mode cli
```

#### Demo Mode with Custom Parameters
```powershell
python Unique_weather_predictor.py --mode demo --city "Warsaw" --days 30 --model rf
```

**Available model types:**
- `rf` - Random Forest (default)
- `gb` - Gradient Boosting
- `ridge` - Ridge Regression
- `elastic` - Elastic Net
- `svr` - Support Vector Machine
- `nn` - Neural Network (Multi-layer Perceptron)

### Usage Instructions
1. **GUI Mode**: 
   - Launch the application
   - Enter city name in the input field
   - Select prediction model from dropdown
   - Click "Predict Weather" button
   - View results and accuracy metrics

2. **CLI Mode**:
   - Follow on-screen prompts
   - Enter city name when requested
   - Choose prediction options
   - View detailed results in terminal

3. **Demo Mode**:
   - Automatically trains and evaluates the model
   - Displays comprehensive performance metrics
   - Shows prediction accuracy and model comparison

## Project Structure

```
weather-forecast/
├── weather_predictor/          # Main package
│   ├── __init__.py            # Package initialization
│   ├── predictor.py           # Main predictor class
│   ├── ml_model.py            # ML model implementations
│   ├── data_collector.py      # API data collection
│   ├── gui.py                 # GUI implementation
│   ├── cli.py                 # CLI implementation
│   ├── config.py              # Configuration settings
│   ├── exceptions.py          # Custom exceptions
│   └── models/                # Trained model storage
├── tests/                     # Unit tests
├── requirements.txt           # Dependencies
├── Unique_weather_predictor.py # Main entry point
└── README.md                  # This file
```
