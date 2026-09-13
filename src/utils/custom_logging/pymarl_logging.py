import logging

from collections import defaultdict
from dataclasses import  dataclass
from types import NoneType
from pathlib import Path

import torch
import numpy

from utils.custom_logging.custom_logging import CustomLogger, CustomLoggerConfig


def _use_tensorboard_without_tensorflow():
    """Force TensorBoard's stub TF API so the real TensorFlow package is not imported.

    ``torch.utils.tensorboard.SummaryWriter`` goes through ``tensorboard.compat.tf``.
    If TensorFlow is installed, that loads the real TF CUDA runtime, which
    segfaults with pygame/mpe2 (and with PyTorch). Pre-registering the empty
    ``tensorboard.compat.notf`` module makes TensorBoard use ``tensorflow_stub``
    instead; event files are unchanged.
    """
    import sys
    import types

    sys.modules.setdefault(
        "tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf")
    )


@dataclass
class PyMARLLoggerConfig(CustomLoggerConfig):
    """
    Configuration class for PyMARLLogger, which extends the base logging
    configuration from `CustomLoggerConfig`. This class adds additional
    configuration options for integrating TensorBoard, Sacred, and Weights and Biases (wandb)
    logging frameworks.

    In addition to the parameters defined in `CustomLoggerConfig`, the following parameters are introduced:
        - use_tensorboard (bool): Whether to enable TensorBoard logging. Defaults to False.
        - use_sacred (bool): Whether to use Sacred for experiment tracking. Defaults to False.
        - use_wandb (bool): Whether to enable Weights and Biases (wandb) logging. Defaults to False.

    """
    use_tensorboard: bool = False
    use_sacred: bool = False
    use_wandb: bool = False

    def __repr__(self):
        return super().__repr__()

    def __post_init__(self):
        if self.use_sacred is True:
            self.use_sacred = False
            PyMARLLogger.fast_logger.info(f"Sacred logging is removed in new train structure.")

        if self.use_wandb is True:
            self.use_wandb = False
            PyMARLLogger.fast_logger.warning(f"Wandb logging is not implemented yet.")
            raise NotImplementedError("wandb logging is not implemented yet.")


class PyMARLLogger(CustomLogger):
    """
    Custom logger for PyMARL Structured MARL training process logging.

    Args:
        name (str): The name of the logger.
        level (int): The level of the logger. Default is logging.NOTSET.
        **configs: The keyword arguments passed to the logger.

    Attributes:
        stats (defaultdict): A default dict of empty list to storage stats fot plotting.

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

    def __init__(self, name, level: int = logging.NOTSET, **configs):
        if hasattr(self, "_initialized"):
            return
        super().__init__(name, level, **configs)

        valid_configs = {
            k: v
            for k, v in configs.items()
            if k in PyMARLLoggerConfig.__annotations__
        }
        self.configs = PyMARLLoggerConfig(level=level, **valid_configs)

        # PyMarl Structure.
        self.use_tensorboard = self.configs.use_tensorboard
        self.use_sacred = self.configs.use_sacred
        self.use_wandb = self.configs.use_wandb

        self.tb_writer = None

        # sacred info.
        self._run_obj = None
        self.sacred_info = None

        # Stats storage.
        self.stats = defaultdict(lambda: [])

        self.args_storage = None   # TODO: Temporary solution for args sharing. Will be removed after Config is implemented.

        # TODO: Due to the params passing method, loggers need to be initialized out of inti.
        #  Consider refactoring after Config is implemented.

    def setup_tensorboard_logging(self, directory_name: Path):
        """Initialize a SummaryWriter to log tensorboard data."""
        _use_tensorboard_without_tensorflow()
        from torch.utils.tensorboard import SummaryWriter

        self.tb_writer = SummaryWriter(log_dir=directory_name)
        self.use_tensorboard = True

        self.info(f"Initialized tensorboard writer with logging dir: \"{directory_name}\"")

    def setup_sacred_logging(self, sacred_run_dict):
        """Initialize sacred variables"""
        self._run_obj = sacred_run_dict
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    @staticmethod
    def setup_wandb_logging(*args, **kwargs):
        raise NotImplementedError("wandb logging is not implemented yet.")

    def log_scalar(self, key, value, t):
        """Add a scalar to initialized log tools."""
        if isinstance(value, torch.Tensor):
            value = value.item()

        self.stats[key].append((t, value))

        if self.use_tensorboard:
            self.tb_writer.add_scalar(key, value, t)

        if self.use_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

            self._run_obj.log_scalar(key, value, t)

    def log_histogram(self, key, value, t):
        """Add a histogram to tensorboard."""
        self.tb_writer.add_histogram(key, value, t)

    def log_embedding(self, key, value):
        """Add an embedding to tensorboard."""
        # TODO Add pre-transition to embeddings.
        self.tb_writer.add_embedding(key, value)

    def print_recent_stats(self):
        """Log recent stats stored in self.stats."""
        # First line with environment step and episode info
        log_str = f"t_env: {self.stats['running/episode'][-1][0]}                     "
        log_str += f"Episode: {self.stats['running/episode'][-1][1]}\n"

        # Hanging indent for all subsequent variable lines
        indent = " " * 8  # 8 spaces for hanging indent

        for k, v in sorted(self.stats.items()):
            if k == "running/episode":
                continue
                
            # Determine window size for averaging
            window = 10 if k != "running/episode" else 1
            
            # Calculate mean value with error handling
            try:
                item = "{:.4f}".format(numpy.mean([x[1] for x in self.stats[k][-window:]]))
            except AttributeError:
                item = "{:.4f}".format(numpy.mean([x[1].item() for x in self.stats[k][-window:]]))
            
            # Add each variable on its own line with hanging indent
            log_str += f"{indent}{k:<23}: {item:>8}\n"
        
        # Remove the trailing newline and log the result
        self.info(log_str.rstrip())

    def finish(self, args):
        """Log final metrics in the training process."""
        if self.use_tensorboard:
            hparam_dict = {}
            for key, value in vars(args).items():
                # TODO Use a param to control hparams to be logged.
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if not isinstance(sub_value, (bool, str, float, int, NoneType, torch.Tensor)):
                            continue
                        hparam_dict[f"{key}.{sub_key}"] = sub_value
                else:
                    if isinstance(value, (bool, str, float, int, NoneType, torch.Tensor)):
                        hparam_dict[key] = value

            metric_dict = {}
            for key, value in self.stats.items():
                # TODO Use a param to control metrics to be logged.
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        metric_dict[f"{key}.{sub_key}"] = torch.tensor(sub_value[-1][-1])
                metric_dict[key] = torch.tensor(value[-1][-1])

            self.tb_writer.add_hparams(
                hparam_dict=hparam_dict,
                metric_dict=metric_dict,
                run_name=".",
                global_step=max(values[-1][0] for values in self.stats.values()),
            )
            self.tb_writer.close()

    # Supply compatibility for origin PyMARL logging.
    @property
    def console_logger(self) -> logging.Logger:
        return self._logger

    def setup_tb(self, *args, **kwargs):
        self.setup_tensorboard_logging(*args, **kwargs)

    def setup_sacred(self, *args, **kwargs):
        self.setup_sacred_logging(*args, **kwargs)

    def setup_wandb(self, *args, **kwargs):
        self.setup_wandb_logging(*args, **kwargs)

    def log_stat(self, *args, **kwargs):
        self.log_scalar(*args, **kwargs)

if __name__ == '__main__':
    # Test PyMARLLogger.
    logger = PyMARLLogger("test_logger", use_tensorboard=True, level=logging.DEBUG)
    logger.info("Test info message.")
    logger.debug("Test debug message.")
    logger.warning("Test warning message.")
    logger.error("Test error message.")
    logger.critical("Test critical message.")
