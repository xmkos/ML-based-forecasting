# Packaging configuration for the project
from setuptools import setup, find_packages

setup(
    name='weather_predictor',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'nbeats-keras>=1.0.0',
        'tensorflow>=2.12',
        'pandas',
        'numpy',
        'scikit-learn',
        'requests',
        'python-dotenv',
        'geocoder'
    ]
)
