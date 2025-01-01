import sys
import time
import logging
from pathlib import Path


class MetaCustomLogger(type):
    """Metaclass for inheriting attributes and methods from python logging module."""
    @classmethod
    def __getattr__(cls, item):
        if item in cls.__dict__:
            return getattr(cls, item)
        else:
            return getattr(logging, item)


class CustomLogger(metaclass=MetaCustomLogger):
    """Custom logger class to log messages with a specific format and handlers.
    TODO: Implement Class level function like logging.info().

    Args:
        name (str): The name of the logger.
        level (int): The level of the logger (defaults to `logging.NOTSET`).
        *args: The positional arguments passed to the logger.
        **kwargs: The keyword arguments passed to the logger.

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
        if "name" in kwargs:    # TODO: Implement a logger like root logger to enable directly logging without initializing a logger object.
            logger_name = kwargs["name"]
        elif len(args) > 0:
            logger_name = args[0]
        else:
            raise RuntimeError("No logger name was provided.")

        # Check if logger has already been initialized.
        if logger_name in cls._instance.keys():
            if len(args) > 1 or (len(kwargs) > 0 and "name" not in kwargs.keys()):
                print(
                    f"{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))} "
                    f"| WARNING  | LOGGINGINIT    | "
                    f"{str(cls.name)} object \"{logger_name}\" has already been initialized. Any settings would NOT be passed."
                )
        else:
            cls._instance[logger_name] = super(CustomLogger, cls).__new__(cls)

        # Return the initialized logger.
        return cls._instance[logger_name]

    def __init__(
            self,
            name: str,
            level: int = logging.NOTSET,
            **kwargs
    ):
        if getattr(self, "_initialized", False):
            return
        # Parameters init.
        self.kwargs: dict = kwargs  # Store all keyword arguments.

        self.fmt: str = kwargs.get("fmt", "{asctime} | {levelname:<8} | {name:<12} | {message}")
        self.date_fmt: str = kwargs.get("date_fmt", "%Y-%m-%d_%H-%M-%S")

        self.console_log: bool = kwargs.get("console_log", True)
        self.console_log_level: int = kwargs.get("console_log_level", level)

        self.file_log: bool = kwargs.get("file_log", False)
        self.file_log_mode: str = kwargs.get("file_log_mode", "a")
        self.file_log_level: int = kwargs.get("file_log_level", level)
        self.file_log_encoding: str = kwargs.get("file_log_encoding", "utf-8")
        self.log_dir: Path = Path(kwargs.get("log_dir", "./log")).resolve()
        self.log_file_name: str = kwargs.get(
            "log_file_name",
            time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time())) + ".log",
        )

        # logger initialization.
        self.formatter = logging.Formatter(fmt=self.fmt, datefmt=self.date_fmt, style="{")

        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.propagate = kwargs.get("propagate", False)

        if self.console_log is True:
            # Setup console logging.
            self._setup_console_logging()

        if self.file_log is True:
            # Setup file logging
            self._setup_file_logging()

        self._initialized = True

    @property
    def logger(self) -> logging.Logger:
        if hasattr(self, "_logger"):
            return self._logger
        else:
            raise AttributeError

    @logger.setter
    def logger(self, *args, **kwargs):
        raise RuntimeError(f"Property 'logger' should not be set. Use '{str(self.__class__)}.init_from_logger()' to get a new logger.")

    @property
    def name(self) -> str:
        return self._logger.name

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
    def root(self) -> logging.Logger:
        return self._logger.root

    @property
    def propagate(self) -> bool:
        return self._logger.propagate

    @propagate.setter
    def propagate(self, propagate: bool):
        self._logger.propagate = propagate

    @property
    def filters(self) -> list[logging.Filter]:
        return self._logger.filters

    @property
    def handlers(self) -> list[logging.Handler]:
        return self._logger.handlers

    def get_child_logger(self, name: str, level: int = logging.NOTSET) -> "CustomLogger":
        """
        Return a child logger whose parent is self.

        Args:
            name (str): Child logger name
            level (int): Child logger level

        Returns:
            A child logger whose parent is self
        """
        if level == logging.NOTSET:
            level = self.level

        child_logger = self.__class__(
            name=f"{name}",
            level=level,
            **self.kwargs
        )
        child_logger.parent = self._logger
        child_logger.propagate = True
        child_logger._logger.handlers = []   # Avoid repeat logging.

        return child_logger

    def init_from_logger(self, ori_logger: logging.Logger, *args, **kwargs) -> "CustomLogger":
        self.warning(f"Initializing {str(self.__class__)} object from logger {ori_logger.name}. This method hasn't been tested.")
        name = ori_logger.name
        level = ori_logger.level
        return self.__class__(name, level, *args, **kwargs)

    def _setup_console_logging(self):
        """Set up console logging with different handlers for stdout and stderr."""
        # Log lower levels to stdout.
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(self.console_log_level)
        stdout_handler.setFormatter(self.formatter)
        stdout_handler.addFilter(lambda record: record.levelno <= logging.WARN)
        self._logger.addHandler(stdout_handler)

        # Log higher levels to stderr.
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(self.console_log_level)
        stderr_handler.setFormatter(self.formatter)
        stderr_handler.addFilter(lambda record: record.levelno > logging.WARN)
        self._logger.addHandler(stderr_handler)

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
    logger = CustomLogger("test_logger", level=CustomLogger.DEBUG)
    logger.info("Test info")
    logger.info(CustomLogger.DEBUG)
    logger.info(CustomLogger.getLevelNamesMapping())
