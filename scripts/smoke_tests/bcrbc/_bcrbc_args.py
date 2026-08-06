"""Shared args factory for BC-RBC smoke tests.

The BC-RBC modules read every hyperparameter directly from the run config with no
code-side default (research strictness: a missing key must raise). These smoke
tests therefore need to supply the full key set, exactly as ``bcrbc_qmix.yaml``
does for a real run. This helper centralizes that full set so each test only has
to override what it actually varies (n_agents, dims, loss weights, comm flags).
"""

from types import SimpleNamespace as SN

import torch


def make_bcrbc_args(**overrides):
    """Return a SimpleNamespace with the full BC-RBC arg set, applying overrides."""
    defaults = dict(
        device=torch.device("cpu"),
        use_cuda=False,
        common_reward=True,
        agent_output_type="q",
        action_selector="epsilon_greedy",
        epsilon_start=1.0,
        epsilon_finish=0.05,
        epsilon_anneal_time=100,
        evaluation_epsilon=0.0,
        # --- BC-RBC model (mirrors bcrbc_qmix.yaml) ---
        bcrbc_d_model=32,
        bcrbc_agent_output_dim=32,
        bcrbc_z_dim=16,
        bcrbc_num_z_tokens=1,
        bcrbc_depth=1,
        bcrbc_heads=4,
        bcrbc_dropout=0.0,
        bcrbc_q_hidden_dim=32,
        bcrbc_max_delay=8,
        bcrbc_context_window=0,  # 0 = unbounded (legacy) unless a test overrides
        bcrbc_kv_cache=False,    # across-step persistent KV cache (eval rollout); off by default
        bcrbc_retro_max_replay_len=0,
        # comm pathway (on by default — BC-RBC is a communication method)
        bcrbc_use_comm=True,
        comm_gaussian_delay_mean=1.0,
        comm_gaussian_delay_std=1.0,
        # generative eval world model
        bcrbc_generative_eval=False,
        bcrbc_flow_steps=8,
        # --- loss weights ---
        td_loss_weight=1.0,
        rec_loss_weight=0.0,
        retro_loss_weight=0.0,
        msg_rec_loss_weight=0.0,
        flow_loss_weight=0.0,
        # --- learner / mixer ---
        target_type="td",
        standardise_returns=False,
        standardise_rewards=False,
        gamma=0.99,
        optimizer="adamW",
        lr=0.001,
        grad_norm_clip=10,
        double_q=True,
        target_update_interval_or_tau=200,
        learner_log_interval=999999,
        mixer="new_qmix_mixer",
        mixing_embed_dim=8,
        hypernet_embed=16,
    )
    defaults.update(overrides)
    return SN(**defaults)


def delay_scheme(n_actions=None):
    """The three obs-delay scheme keys that run.py adds to every real run.

    BC-RBC reads these unconditionally, so any batch fed to the MAC/learner in a
    smoke test must carry them (matching the production buffer scheme).
    """
    return {
        "obs_delay": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "obs_gen_t": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "obs_fresh_mask": {"vshape": (1,), "group": "agents", "dtype": torch.float32},
    }


def zero_delay_fill(batch_size, time_steps, n_agents):
    """Zero-delay (timely, fresh) values for the obs-delay keys over a full episode."""
    time_ids = torch.arange(time_steps).view(1, time_steps, 1, 1).expand(batch_size, -1, n_agents, -1)
    return {
        "obs_delay": torch.zeros(batch_size, time_steps, n_agents, 1, dtype=torch.long),
        "obs_gen_t": time_ids.long().clone(),
        "obs_fresh_mask": torch.ones(batch_size, time_steps, n_agents, 1),
    }
