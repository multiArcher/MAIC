import os
import sys
import json
import logging
import datetime

from tqdm import tqdm
from hashlib import sha256
from collections import defaultdict

import numpy as np


class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_wandb = False
        self.use_sacred = False
        self.use_hdf = False

        self.stats = defaultdict(lambda: [])
        self.tb_writer = None

    def setup_tb(self, directory_name):
        # Import here so it doesn't have to be installed if you don't use it
        from torch.utils.tensorboard import SummaryWriter

        self.tb_writer = SummaryWriter(log_dir=directory_name)
        self.use_tb = True

        # self.console_logger.info("*" * 80)
        self.console_logger.info(f"Tensorboard logging dir: \"{directory_name}\"")
        # self.console_logger.info("*" * 80)

    def setup_wandb(self, config, team_name, project_name, mode):
        import wandb

        assert (
            team_name is not None and project_name is not None
        ), "W&B logging requires specification of both `wandb_team` and `wandb_project`."
        assert (
            mode in ["offline", "online"]
        ), f"Invalid value for `wandb_mode`. Received {mode} but only 'online' and 'offline' are supported."

        self.use_wandb = True

        alg_name = config["name"]
        env_name = config["env"]
        if "map_name" in config["env_args"]:
            env_name += "_" + config["env_args"]["map_name"]
        elif "key" in config["env_args"]:
            env_name += "_" + config["env_args"]["key"]

        non_hash_keys = ["seed"]
        self.config_hash = sha256(
            json.dumps(
                {k: v for k, v in config.items() if k not in non_hash_keys},
                sort_keys=True,
            ).encode("utf8")
        ).hexdigest()[-10:]

        group_name = "_".join([alg_name, env_name, self.config_hash])

        self.wandb = wandb.init(
            entity=team_name,
            project=project_name,
            config=config,
            group=group_name,
            mode=mode,
        )

        self.console_logger.info("*******************")
        self.console_logger.info("WANDB RUN ID:")
        self.console_logger.info(f"{self.wandb.id}")
        self.console_logger.info("*******************")

        # accumulate data at same timestep and only log in one batch once
        # all data has been gathered
        self.wandb_current_t = -1
        self.wandb_current_data = {}

    def setup_sacred(self, sacred_run_dict):
        self._run_obj = sacred_run_dict
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def log_stat(self, key, value, t, to_sacred=True):
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_writer.add_scalar(key, value, t)

        if self.use_wandb:
            if self.wandb_current_t != t and self.wandb_current_data:
                # self.console_logger.info(
                #     f"Logging to WANDB: {self.wandb_current_data} at t={self.wandb_current_t}"
                # )
                self.wandb.log(self.wandb_current_data, step=self.wandb_current_t)
                self.wandb_current_data = {}
            self.wandb_current_t = t
            self.wandb_current_data[key] = value

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

            self._run_obj.log_scalar(key, value, t)

    def log_histogram(self, key, value, t):
        self.tb_writer.add_histogram(key, value, t)

    def log_embedding(self, key, value):
        self.tb_writer.add_embedding(key, value)

    def print_recent_stats(self):
        log_prefix = f"{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | STATSLOG | "
        log_str = "t_env: {} | Episode: {}\n".format(
            *self.stats["episode"][-1]
        )
        log_str += " " * 33
        i = 0
        for k, v in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            try:
                item = "{:.4f}".format(np.mean([x[1] for x in self.stats[k][-window:]]))
            except:
                item = "{:.4f}".format(
                    np.mean([x[1].item() for x in self.stats[k][-window:]])
                )
            log_str += "{:<23}{:>8}".format(k + ":", item)
            log_str += ("\n" + " " * 33) if i % 3 == 0 else "\t"

        # Use tqdm.write to avoid interrupting the progress bar.
        # self.console_logger.info(log_str)
        tqdm.write(log_prefix + log_str)
        tqdm.write(" " * 33 + "*" * 41 + "STATS FINISHED" + "*" * 41)

    def finish(self):
        if self.use_wandb:
            if self.wandb_current_data:
                self.wandb.log(self.wandb_current_data, step=self.wandb_current_t)
            self.wandb.finish()


# set up a custom logger
def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler(stream=sys.stdout)

    # modified logging formatter
    logging_fmt = "{asctime} | {levelname:<8} | {name:<8} |{message}"
    logging_date_fmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=logging_fmt, datefmt=logging_date_fmt, style="{")

    # ori formatter.
    # formatter = logging.Formatter(
    #     "[%(levelname)s %(asctime)s] %(name)s %(message)s", "%H:%M:%S"
    # )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel("DEBUG")

    return logger
