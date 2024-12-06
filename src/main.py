import logging
import os
import sys
import yaml
import datetime
from copy import deepcopy
from os.path import dirname, abspath

import numpy
import torch
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

try:
    from collections import Mapping    # until python 3.10
except ImportError:
    from collections.abc import Mapping    # from python 3.10

from run import run
# from utils.pymarl_logging import get_logger
from utils.marl_logging import PyMARLLogger

# set to "no" if you want to see stdout/stderr in console "sys" / "no" / "fd"
SETTINGS["CAPTURE_MODE"] = "no"


ex = Experiment("pymarl", save_git_info=False)
# logger = get_logger()
# ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    numpy.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    config["env_args"]["seed"] = config["seed"]  # Set the seed in SCII game.

    # run the framework
    run(_run, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(
            os.path.join(
                os.path.dirname(__file__),
                "config",
                subfolder,
                "{}.yaml".format(config_name),
            ),
            "r",
        ) as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == "__main__":
    # region read_parameters
    # Params order is command line -> algs -> envs -> defaults
    params = deepcopy(sys.argv)
    # th.set_num_threads(1)

    # Get the defaults from default.yaml
    with open(
        os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r"
    ) as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # endregion

    # Generate unique token.
    # get map_name from yaml
    try:
        map_name = config_dict["env_args"]["map_name"]
    except KeyError:
        map_name = config_dict["env_args"]["key"]
    # Get experiment name from yaml
    experiment_name = config_dict["name"]

    # Update map name and experiment name from command line parameters.
    for param in params:
        if param.startswith("env_args.map_name"):
            map_name = param.split("=")[1]
        elif param.startswith("env_args.key"):
            map_name = param.split("=")[1]
        elif param.startswith("name"):
            experiment_name = param.split("=")[1]

    unique_token = (
        f"{experiment_name}_{map_name}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}"
    )

    config_dict.update({"unique_token": unique_token})

    logger = PyMARLLogger(
        name="main",
        level=logging.DEBUG,
        file_log=True,
        log_dir="./results/log",
        log_file_name=unique_token + ".log"
    )
    ex.logger = logger.logger

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in \"./results/sacred\".")
    file_obs_path = os.path.join(results_path, f"sacred/{unique_token}")

    ex.observers.append(
        FileStorageObserver(file_obs_path, copy_artifacts=False, copy_sources=False)
    )
    # ex.observers.append(MongoObserver(db_name="marlbench")) #url='172.31.5.187:27017'))
    # ex.observers.append(MongoObserver())

    # Run experiment.
    ex.add_config(config_dict)
    ex.run_commandline(params)
