import sys
import time
import logging

from pathlib import Path
from dataclasses import  dataclass, field, asdict, replace

from ..custom_repr_mixin import CustomConfigReprMixin


@dataclass
class CustomLoggerConfig(CustomConfigReprMixin):
    """
    A dataclass for storing custom logger configurations. This class contains
    configuration parameters for the logger, such as logging level, format,
    file logging settings, and more. It also includes methods for generating
    a dictionary representation of the configuration and handling post-initialization.

    Attributes:
        name (str): The name of the logger.
        level (int): The logging level for the logger (e.g., logging.INFO).
        fmt (str): The log message format.
        date_fmt (str): The format for the log timestamp.
        console_log (bool): Whether to enable logging to the console.
        console_log_level (int, optional): The logging level for the console log.
        file_log (bool): Whether to enable logging to a file.
        file_log_mode (str): The mode for opening the log file ('a' for append, 'w' for overwrite, etc.).
        file_log_level (int, optional): The logging level for the file log.
        file_log_encoding (str): The encoding for the log file.
        log_dir (Path): The directory to store the log files.
        log_file_name (str): The name of the log file, generated based on the current timestamp.
        propagate (bool): Whether to propagate log messages to higher-level loggers.

    Methods:
        dict: Returns a dictionary representation of the configuration.
   """
    name: str = "main"
    level: int = logging.INFO

    fmt: str = "{asctime} | {levelname:<8} | {name:<12} | {message}"
    date_fmt: str = "%Y-%m-%d_%H-%M-%S"

    console_log: bool = True
    console_log_level: int = None

    file_log: bool = False
    file_log_mode: str = "a"
    file_log_level: int = None
    file_log_encoding: str = "utf-8"
    log_dir: Path = "./log"
    log_file_name: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time())) + ".log"
    )

    propagate: bool = False

    @property
    def dict(self):
        return asdict(self)

    def __post_init__(self):
        # Set default console log level to the same as CustomLogger.
        if self.console_log_level is None:
            self.console_log_level = self.level

        # Set default file log level to the same as CustomLogger.
        if self.file_log_level is None:
            self.file_log_level = self.level

        # Set log directory to absolute path.
        self.log_dir = Path(self.log_dir).resolve()

        # Create log directory if it does not exist.
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True)

    def __repr__(self):
        return super().__repr__()


class MetaCustomLogger(type):
    """Metaclass for CustomLogger class.

    This metaclass is used to add custom attributes to the CustomLogger class.
    """
    @classmethod
    def __getattr__(cls, item):
        """Provide access to logging module attributes."""
        if item in cls.__dict__:
            return getattr(cls, item)
        else:
            return getattr(logging, item)

    @classmethod
    def fast_logger(cls, logger_name: str = "fast_logger") -> "CustomLogger":
        """
        Return a child logger with main settings or default settings to log messages
        anywhere in code without initializing a new logger.

        Returns:
            custom_logger: A child logger with default logger_name "fast_logger".
        """
        if CustomLogger.has_instance(logger_name):
            # fast_logger has already been initialized.
            return CustomLogger(logger_name)
        elif CustomLogger.has_instance("main"):
            # Inherits from main logger.
            return CustomLogger("main").get_child_logger(logger_name)
        else:
            # Create a new fast_logger with default settings.
            return CustomLogger(logger_name, CustomLogger.INFO)


class CustomLogger(metaclass=MetaCustomLogger):
    """Custom logger class to log messages with a specific format and handlers.

    Args:
        name (str): The name of the logger.
        level (int): The level of the logger (default is `logging.NOTSET`).
        **configs: The keyword arguments passed to the logger.

    Note:
        The following keyword arguments are accepted:\n
        - fmt (str): The format of the logger. Default is "{asctime} | {levelname:<8} | {name:<12} | {message}".
        - date_fmt (str): The format of the asctime in fmt. Default is "%Y-%m-%d_%H-%M-%S".
        - console_log (bool). Whether to log to console. Default is True.
        - console_log_level (int): Level of the console log handler. Default is the same as CustomLogger.
        - file_log (bool). Whether to log to file. Default is False.
        - file_log_mode (str): How to write the log file. Default is "a".
        - file_log_level (int): Level of the file log handler. Default is the same as CustomLogger.
        - log_dir (str): Directory for storaging log files. Default is "./log".
        - log_file_name (str): Name of the log file. Default is time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time())).
        - propagate (bool): Whether to propagate through child loggers. Default is False.
    """
    _instance = {}

    def __new__(cls, *args, **kwargs):
        # Get logger name.
        if len(args) > 0:
            # If name is provided in args, use it.
            logger_name = args[0]
        elif "name" in kwargs:
            # If name is provided in kwargs, use it.
            logger_name = kwargs["name"]
        else:
            # If name is not provided, raise an error.
            raise RuntimeError("No logger name was provided.")

        # Check if logger has already been initialized.
        if cls.has_instance(logger_name):
            # If logger has already been initialized, return the existing instance.
            instance = cls._instance[logger_name]
            if len(args) > 1 or (len(kwargs) > 0 and "name" not in kwargs.keys()):
                # Extra arguments are provided.
                # TODO: Update the logger with the new settings.
                instance.warning(
                    f"{str(instance.__class__.__name__)} object {instance} has already been initialized. "
                    "Update settings is not supported yet. Any settings would NOT be passed."
                )
        else:
            # If logger has not been initialized, create a new instance.
            cls._instance[logger_name] = super(CustomLogger, cls).__new__(cls)

        # Return the initialized logger.
        return cls._instance[logger_name]

    def __init__(
            self,
            name: str,
            level: int = logging.NOTSET,
            **configs
    ):
        if hasattr(self, "_initialized"):
            # If logger has already been initialized, do nothing.
            return

        self._logger = logging.getLogger(name)

        # Parameters init.
        valid_configs = {
            k: v
            for k, v in configs.items()
            if k in CustomLoggerConfig.__annotations__
        }
        self.configs = CustomLoggerConfig(level=level, **valid_configs)

        self.fmt = self.configs.fmt
        self.date_fmt = self.configs.date_fmt
        self.console_log = self.configs.console_log
        self.console_log_level = self.configs.console_log_level
        self.file_log = self.configs.file_log
        self.file_log_mode = self.configs.file_log_mode
        self.file_log_level = self.configs.file_log_level
        self.file_log_encoding = self.configs.file_log_encoding
        self.log_dir = self.configs.log_dir
        self.log_file_name = self.configs.log_file_name

        # logger initialization.
        self.level = level
        self.propagate = self.configs.propagate
        self.formatter = logging.Formatter(fmt=self.fmt, datefmt=self.date_fmt, style="{")

        if self.console_log is True:
            self._setup_console_logging()

        if self.file_log is True:
            self._setup_file_logging()

        self._initialized = True

    @property
    def logger(self) -> logging.Logger:
        if hasattr(self, "_logger"):
            return self._logger
        else:
            raise AttributeError(f"{self.__class__.__name__} object has no attribute '_logger'.")

    @logger.setter
    def logger(self, *args, **kwargs):
        raise RuntimeError(f"Property 'logger' should not be set. Use '{str(self.__class__)}.init_from_logger()' to get a new logger.")

    @property
    def main_logger(self) -> "CustomLogger":
        """Return the main logger."""
        if self.has_instance("main") is True:
            return self._instance["main"]
        else:
            return CustomLogger("main", CustomLogger.INFO)

    @property
    def level(self) -> int:
        return self._logger.level

    @level.setter
    def level(self, level: int):
        self._logger.setLevel(level)
        for handler in self._logger.handlers:
            handler.setLevel(level)

    @property
    def parent(self) -> logging.Logger:
        return self._logger.parent

    @parent.setter
    def parent(self, parent: logging.Logger):
        self._logger.parent = parent

    @property
    def propagate(self) -> bool:
        return self._logger.propagate

    @propagate.setter
    def propagate(self, propagate: bool):
        self._logger.propagate = propagate

    @classmethod
    def has_instance(cls, name: str) -> bool:
        return name in cls._instance.keys()

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)

    def fatal(self, msg, *args, **kwargs):
        self.logger.fatal(msg, *args, **kwargs)

    def get_child_logger(self, name: str, level: int = logging.NOTSET, propagate: bool = True, **kwargs) -> "CustomLogger":
        """
        Return a child logger whose parent is self.

        Args:
            name (str): Child logger name
            level (int): Child logger level
            propagate (bool): Whether to propagate through child loggers.

        Returns:
            A child logger whose parent is current logger.
        """
        if level == logging.NOTSET:
            level = self.level

        child_config = replace(self.configs, name=name, level=level, propagate=propagate, **kwargs)

        child_logger = self.__class__(**asdict(child_config))
        child_logger.parent = self._logger

        if propagate is True:
            child_logger.propagate = True
            child_logger.logger.handlers = []   # Avoid repeat logging.
        else:
            child_logger.propagate = False

        return child_logger

    def init_from_logger(self, ori_logger: logging.Logger, **kwargs) -> "CustomLogger":
        self.warning(f"Initializing {str(self.__class__)} object from logger {ori_logger.name}. This method hasn't been tested.")
        name = ori_logger.name
        level = ori_logger.level
        return self.__class__(name, level, **kwargs)

    def _setup_console_logging(self):
        """Set up console logging with different handlers for stdout and stderr."""
        # Log lower levels to stdout.
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(self.console_log_level)
        stdout_handler.setFormatter(self.formatter)
        stdout_handler.addFilter(lambda record: record.levelno <= logging.WARN)
        self.logger.addHandler(stdout_handler)

        # Log higher levels to stderr.
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(self.console_log_level)
        stderr_handler.setFormatter(self.formatter)
        stderr_handler.addFilter(lambda record: record.levelno > logging.WARN)
        self.logger.addHandler(stderr_handler)

    def _setup_file_logging(self):
        """Set up file logging."""
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.error(f"Could not create log directory {self.log_dir}: {e}. Initialization of file handler failed.")
            raise RuntimeError(f"Log directory creation failed: {e}")

        file_handler = logging.FileHandler(
            filename=self.log_dir / self.log_file_name,
            mode=self.file_log_mode,
            encoding=self.file_log_encoding,
        )
        file_handler.setLevel(self.file_log_level)
        file_handler.setFormatter(self.formatter)
        self._logger.addHandler(file_handler)

    def __getattr__(self, item):
        """Provide access to logging.Logger attributes."""
        if "_logger" in self.__dict__:
            return getattr(self._logger, item)
        else:
            raise AttributeError(f"{self.__class__.__name__} object has no attribute '{item}'")

    def __str__(self):
        return f"<{self.__class__.__name__} {self.name} ({logging.getLevelName(self.level)})>"

    def __repr__(self):
        # TODO Should display handler, formatter, etc.
        return self.__str__()


if __name__ == '__main__':
    logger = CustomLogger("main", CustomLogger.DEBUG, console_log=True, file_log=False)
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    CustomLogger.fast_logger("Test_fast_logging").debug("This is a debug message")
    CustomLogger.fast_logger("Test_fast_logging").info("This is an info message")
    CustomLogger.fast_logger("Test_fast_logging").info("This is an info message")
    CustomLogger.fast_logger("Test_fast_logging").error("This is an error message")
    CustomLogger.fast_logger("Test_fast_logging").critical("This is a critical message")

    child = logger.get_child_logger("child", CustomLogger.DEBUG)
    child.debug("This is a debug message from child logger")
    child.info("This is an info message from child logger")
    child.warning("This is a warning message from child logger")
    child.error("This is an error message from child logger")
    child.critical("This is a critical message from child logger")
