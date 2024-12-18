"""
# Custom Logging

The `custom_logging` module defines a custom logger class (`CustomLogger`) that 
expands Python's standard logging module. This module provides flexibility in 
configuring logging formats, levels, and handlers for different output demands.

## Classes

- MetaCustomLogger

    This is a metaclass used by `CustomLogger` to inherit attributes and methods 
    from the Python logging module. It allows dynamic access to logging attributes 
    when needed.

- CustomLogger

    This class implements a custom logging solution with additional configuration options.
    
- PyMARLLogger

    This class is a child class of `CustomLogger` that is specifically designed 
    for use with the PyMARL framework. It provides additional methods for logging 
    of information from MARL training, such as win rate and other metrics.

## Usage Example

A basic usage example is provided for testing the logger:

```python
if __name__ == '__main__':
    logger = CustomLogger("test_logger", level=logging.DEBUG)
    logger.info("Test info")
    logger.info(CustomLogger.DEBUG)
    logger.info(CustomLogger.getLevelNamesMapping())
```

This initializes a logger named "test_logger" with a logging level of DEBUG, and demonstrates logging informational messages.
"""

from .custom_logging import CustomLogger
from .pymarl_logging import PyMARLLogger

__all__ = ["CustomLogger", "PyMARLLogger"]
