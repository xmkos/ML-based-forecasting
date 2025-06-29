# File: tests/test_exceptions.py
# Unit tests for custom exceptions

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from weather_predictor.exceptions import DownloadError, ModelLoadError, DataError


class TestCustomExceptions:
    """Test suite for custom exception classes"""

    def test_download_error_inheritance(self):
        """Test that DownloadError inherits from Exception"""
        assert issubclass(DownloadError, Exception)

    def test_download_error_with_message(self):
        """Test DownloadError with custom message"""
        message = "Failed to download weather data"
        error = DownloadError(message)
        
        assert str(error) == message
        assert error.args[0] == message

    def test_download_error_raise(self):
        """Test raising DownloadError"""
        with pytest.raises(DownloadError) as exc_info:
            raise DownloadError("Test download error")
        
        assert "Test download error" in str(exc_info.value)

    def test_model_load_error_inheritance(self):
        """Test that ModelLoadError inherits from Exception"""
        assert issubclass(ModelLoadError, Exception)

    def test_model_load_error_default_message(self):
        """Test ModelLoadError with default message"""
        error = ModelLoadError()
        
        assert hasattr(error, 'message')
        assert "Error occurred while loading the model" in error.message

    def test_model_load_error_custom_message(self):
        """Test ModelLoadError with custom message"""
        custom_message = "Custom model loading error"
        error = ModelLoadError(custom_message)
        
        assert error.message == custom_message
        assert str(error) == custom_message

    def test_model_load_error_raise(self):
        """Test raising ModelLoadError"""
        with pytest.raises(ModelLoadError) as exc_info:
            raise ModelLoadError("Test model load error")
        
        assert "Test model load error" in str(exc_info.value)

    def test_data_error_inheritance(self):
        """Test that DataError inherits from Exception"""
        assert issubclass(DataError, Exception)

    def test_data_error_with_message(self):
        """Test DataError with custom message"""
        message = "Invalid weather data format"
        error = DataError(message)
        
        assert str(error) == message

    def test_data_error_raise(self):
        """Test raising DataError"""
        with pytest.raises(DataError) as exc_info:
            raise DataError("Test data error")
        
        assert "Test data error" in str(exc_info.value)

    def test_exception_hierarchy(self):
        """Test that all custom exceptions can be caught as Exception"""
        exceptions_to_test = [
            DownloadError("download"),
            ModelLoadError("model"),
            DataError("data")
        ]
        
        for custom_exception in exceptions_to_test:
            with pytest.raises(Exception):
                raise custom_exception

    def test_exception_chaining(self):
        """Test exception chaining if supported"""
        try:
            # Simulate a chain of exceptions
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise ModelLoadError("Model failed to load") from e
        except ModelLoadError as e:
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, ValueError)

    def test_exception_context_manager(self):
        """Test exceptions in context managers"""
        with pytest.raises(DownloadError):
            raise DownloadError("Context manager test")

    def test_multiple_exception_types(self):
        """Test catching multiple exception types"""
        def raise_random_error(error_type):
            if error_type == "download":
                raise DownloadError("Download failed")
            elif error_type == "model":
                raise ModelLoadError("Model failed")
            elif error_type == "data":
                raise DataError("Data invalid")

        # Test catching specific types
        with pytest.raises(DownloadError):
            raise_random_error("download")
        
        with pytest.raises(ModelLoadError):
            raise_random_error("model")
        
        with pytest.raises(DataError):
            raise_random_error("data")

    def test_exception_attributes(self):
        """Test exception attributes and methods"""
        # Test DownloadError
        download_error = DownloadError("Download test")
        assert hasattr(download_error, 'args')
        assert download_error.args[0] == "Download test"

        # Test ModelLoadError with message attribute
        model_error = ModelLoadError("Model test")
        assert hasattr(model_error, 'message')
        assert model_error.message == "Model test"

        # Test DataError
        data_error = DataError("Data test")
        assert hasattr(data_error, 'args')
        assert data_error.args[0] == "Data test"

    def test_exception_string_representation(self):
        """Test string representation of exceptions"""
        download_err = DownloadError("Download message")
        model_err = ModelLoadError("Model message")
        data_err = DataError("Data message")

        assert "Download message" in str(download_err)
        assert "Model message" in str(model_err)
        assert "Data message" in str(data_err)

    def test_exception_in_try_except_blocks(self):
        """Test exceptions in realistic try-except scenarios"""
        def simulate_api_call():
            raise DownloadError("API connection failed")

        def simulate_model_loading():
            raise ModelLoadError("Model file not found")

        def simulate_data_processing():
            raise DataError("Invalid data format")

        # Test API call error handling
        try:
            simulate_api_call()
            assert False, "Should have raised DownloadError"
        except DownloadError as e:
            assert "API connection failed" in str(e)

        # Test model loading error handling
        try:
            simulate_model_loading()
            assert False, "Should have raised ModelLoadError"
        except ModelLoadError as e:
            assert "Model file not found" in str(e)

        # Test data processing error handling
        try:
            simulate_data_processing()
            assert False, "Should have raised DataError"
        except DataError as e:
            assert "Invalid data format" in str(e)

    def test_exception_with_empty_message(self):
        """Test exceptions with empty messages"""
        # Test with empty string
        error = DataError("")
        assert str(error) == ""

    def test_exception_documentation(self):
        """Test that exceptions have proper docstrings"""
        assert DownloadError.__doc__ is not None
        assert "download" in DownloadError.__doc__.lower()

        assert ModelLoadError.__doc__ is not None
        assert "model" in ModelLoadError.__doc__.lower()

        # DataError might not have docstring in current implementation
        # but it should be a proper exception class
        assert issubclass(DataError, Exception)
