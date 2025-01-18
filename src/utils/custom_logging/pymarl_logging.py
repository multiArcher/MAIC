import logging

import torch
import numpy
from collections import defaultdict

from utils.custom_logging import CustomLogger


class PyMARLLogger(CustomLogger):
    """
    Custom logger for PyMARL Structured MARL training process logging.

    Args:
        name (str): The name of the logger.
        level (int): The level of the logger.
        *args: The positional arguments passed to the logger.
        **kwargs: The keyword arguments passed to the logger.

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
    def __init__(self, name, level: int = logging.NOTSET, **kwargs):
        if hasattr(self, "_initialized"):
            return
        super().__init__(name, level, **kwargs)

        # PyMarl Structure.
        self.use_tensorboard: bool = kwargs.get("use_tensorboard", False)
        self.use_sacred: bool = kwargs.get("use_sacred", False)
        self.use_wandb: bool = kwargs.get("use_wandb", False)

        self.tb_writer = None

        # sacred info.
        self._run_obj = None
        self.sacred_info = None

        # Stats storage.
        self.stats = defaultdict(lambda: [])

        self.args_storage = None   # TODO: Temporary solution for args sharing. Will be removed after Config is implemented.

    def setup_tensorboard_logging(self, directory_name: str):
        """Initialize a SummaryWriter to log tensorboard data."""
        from torch.utils.tensorboard import SummaryWriter

        self.tb_writer = SummaryWriter(log_dir=directory_name)
        self.use_tensorboard = True

        self.info(f"Tensorboard logging dir: \"{directory_name}\"")

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
        log_str = "t_env: {} | Episode: {}\n".format(*self.stats["episode"][-1])
        log_str += " " * 33
        i = 0
        for k, v in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            try:
                item = "{:.4f}".format(numpy.mean([x[1] for x in self.stats[k][-window:]]))
            except AttributeError:
                item = "{:.4f}".format(numpy.mean([x[1].item() for x in self.stats[k][-window:]]))
            log_str += "{:<23}{:>8}".format(k + ":", item)
            log_str += ("\n" + " " * 44) if i % 3 == 0 else "\t"

        self.info(log_str)

    def finish(self, args):
        """Log final metrics in the training process."""
        if self.use_tensorboard:
            import torch
            hparam_dict = {}
            for key, value in vars(args).items():
                # TODO Use a param to control hparams to be logged.
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if not isinstance(sub_value, (bool, str, float, int, type(None), torch.Tensor)):
                            continue
                        hparam_dict[f"{key}.{sub_key}"] = sub_value
                else:
                    if isinstance(value, (bool, str, float, int, type(None), torch.Tensor)):
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
                run_name=args.name
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
