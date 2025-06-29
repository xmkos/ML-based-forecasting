# Defines custom exception classes
class DownloadError(Exception):
    """Exception raised for errors in the download process."""
    def __init__(self, message: str):
        super().__init__(str(message))

class ModelLoadError(Exception):
    """Exception raised for errors in loading the model."""
    def __init__(self, message="Error occurred while loading the model"):
        self.message = message
        super().__init__(self.message)

class DataError(Exception):
    """Exception raised for errors in the data."""
    def __init__(self, message="Error occurred with the data"):
        self.message = message
        super().__init__(self.message)
