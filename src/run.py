import datetime
import os
import sys
import pprint
import shutil
import tqdm
import threading
import psutil
import time
from os.path import dirname, abspath
from types import SimpleNamespace as SN

import torch as th

from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from utils.general_reward_support import test_alg_config_supports_reward
from utils.logging import Logger
from runners import get_runner
from controllers import get_controller
from learners import get_learner


def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    # args.device = th.device(args.device)
    assert test_alg_config_supports_reward(
        args
    ), "The specified algorithm does not support the general reward setup. Please choose a different algorithm or set `common_reward=True`."

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, underscore_numbers=True)
    _log.info("\n" + experiment_params + "\n")

    # configure tensorboard logger
    if args.use_tensorboard:
        tb_logs_dir = os.path.join(
            dirname(dirname(abspath(__file__))), "results", "tensorboard_logs"
        )
        tb_exp_dir = os.path.join(tb_logs_dir, "{}").format(_config["unique_token"])
        logger.setup_tb(tb_exp_dir)

    if args.use_wandb:
        logger.setup_wandb(
            _config, args.wandb_team, args.wandb_project, args.wandb_mode
        )

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Finish logging
    logger.finish(args)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    # os._exit(os.EX_OK)


def evaluate_sequential(args, runner):
    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, logger):
    # Init runner so we can get env info
    # runner = r_REGISTRY[args.runner](args=args, logger=logger)
    runner = get_runner(args.runner, args=args, logger=logger)
    logger.console_logger.debug(f"Running with {runner.__class__.__name__}.")

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    # For individual rewards in gymmai reward is of shape (1, n_agents)
    if args.common_reward:
        scheme["reward"] = {"vshape": (1,)}
    else:
        scheme["reward"] = {"vshape": (args.n_agents,)}
    groups = {"agents": args.n_agents}
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )

    # Setup multiagent controller here
    # mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    mac = get_controller(args.mac, buffer.scheme, groups, args)
    logger.console_logger.debug(f"Running with {mac.__class__.__name__}.")

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    # learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    learner = get_learner(args.learner, mac, buffer.scheme, logger, args)
    logger.console_logger.debug(f"Running with {learner.__class__.__name__}.")

    if args.use_cuda:
        learner.to(args.device)

    if args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(args.checkpoint_path)
            )
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            runner.log_train_stats_t = runner.t_env
            evaluate_sequential(args, runner)
            logger.log_stat("episode", runner.t_env, runner.t_env)
            logger.print_recent_stats()
            logger.console_logger.info("Finished Evaluation")
            return

    # start training
    episode = 0
    last_test_t = -args.test_interval - 1
    last_log_t = 0
    model_save_time = 0

    logger.console_logger.info(f"Beginning training for {args.t_max} timesteps")
    logger.console_logger.info("*" * 38 + "TRAINING START" + "*" * 38)

    # Delay init tqdm bar
    progress_bar = None
    tqdm_output = open("/dev/tty", "w") if sys.platform.startswith('linux') else sys.stderr

    while runner.t_env <= args.t_max:
        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            for _ in range(args.batch_size_run):
                episode_sample = buffer.sample(args.batch_size)

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_t) / args.test_interval >= 1.0:
            # logger.console_logger.info(
            #     "t_env: {} / {}".format(runner.t_env, args.t_max)
            # )
            # logger.console_logger.info(
            #     "Estimated time left: {}. Time passed: {}".format(
            #         time_left(last_time, last_test_t, runner.t_env, args.t_max),
            #         time_str(time.time() - start_time),
            #     )
            # )
            # last_time = time.time()

            last_test_t = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        # Save models to unique token directory
        if args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval
            or model_save_time == 0
        ):
            model_save_time = runner.t_env
            save_path = os.path.join(
                args.local_results_path, "models", args.unique_token, str(runner.t_env)
            )
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

            if args.use_wandb and args.wandb_save_model:
                wandb_save_dir = os.path.join(
                    logger.wandb.dir, "models", args.unique_token, str(runner.t_env)
                )
                os.makedirs(wandb_save_dir, exist_ok=True)
                for f in os.listdir(save_path):
                    shutil.copyfile(
                        os.path.join(save_path, f), os.path.join(wandb_save_dir, f)
                    )

        episode += args.batch_size_run

        if (runner.t_env - last_log_t) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_t = runner.t_env

        # display process
        if progress_bar is None:
            logger.console_logger.info("Train process started")
            progress_bar = tqdm.tqdm(
                total=args.t_max,
                mininterval=3,
                unit="step",
                bar_format="{desc}{bar:9}| {n_fmt}/{total_fmt} steps{percentage:3.0f}% [{elapsed}<{remaining} {rate_fmt}]{postfix}",
                desc=f"{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")} | TRAINING | ",
                postfix={"episode": episode},
                file=tqdm_output
            )

        # Watch CPU usage.
        memory_info = psutil.virtual_memory()
        total_memory = memory_info.total / 1024 ** 3  # 总内存，单位GB
        free_memory = memory_info.free / 1024 ** 3  # 可用内存，单位GB
        used_memory = memory_info.used / 1024 ** 3  # 已用内存，单位GB

        # Watch GPU usage.
        gpu_available_memory, gpu_total_memory = th.cuda.mem_get_info()
        gpu_available_memory = gpu_available_memory / 1024 ** 3
        gpu_total_memory = gpu_total_memory / 1024 ** 3
        gpu_memory_allocated = th.cuda.memory_allocated() / 1024 ** 3
        gpu_memory_reserved = th.cuda.memory_reserved() / 1024 ** 3

        progress_bar.set_postfix({
                    "episode": episode,
                    "memory": f"{used_memory:2.1f}/{free_memory:2.1f}/{total_memory:2.1f} GB",
                    "gpu": f"{gpu_memory_allocated:2.1f}/{gpu_memory_reserved:2.1f}/{gpu_available_memory:2.1f}/{gpu_total_memory:2.1f} GB"
                })
        progress_bar.set_description_str(f"{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")} | TRAINING | ")
        update_steps = runner.t_env - progress_bar.n
        progress_bar.update(update_steps)
        sys.stdout.flush()

    progress_bar.close()
    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config
