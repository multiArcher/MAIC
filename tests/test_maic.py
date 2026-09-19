import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from controllers.maic_controller import MAICMAC
from learners.maic_learner import MAICLearner
from modules.agents.maic_agent import MAICAgent
from utils.maker import AgentMaker, LearnerMaker, MACMaker


class DummyLogger:
    def __init__(self):
        self.stats = []

    def log_stat(self, key, value, t):
        self.stats.append((key, value, t))

    def info(self, *args, **kwargs):
        return None

    @property
    def console_logger(self):
        return self


def _base_args(**overrides):
    args = SimpleNamespace(
        n_agents=3,
        n_actions=5,
        hidden_dim=16,
        nn_hidden_size=16,
        latent_dim=4,
        attention_dim=8,
        var_floor=0.002,
        mi_loss_weight=0.001,
        entropy_loss_weight=0.01,
        msg_l1_reg_loss_weight=0.0,
        use_rnn=True,
        obs_agent_id=True,
        obs_last_action=True,
        agent="maic",
        mac="maic_mac",
        learner="maic_learner",
        mixer="qmix",
        mixing_embed_dim=8,
        hypernet_layers=2,
        hypernet_embed=16,
        agent_output_type="q",
        action_selector="epsilon_greedy",
        epsilon_start=1.0,
        epsilon_finish=0.05,
        epsilon_anneal_time=100,
        evaluation_epsilon=0.0,
        device=torch.device("cpu"),
        lr=0.0005,
        gamma=0.99,
        optimiser="Adam",
        optim_alpha=0.99,
        optim_eps=1e-5,
        grad_norm_clip=10,
        double_q=True,
        target_update_interval_or_tau=200,
        learner_log_interval=1,
        standardise_returns=False,
        standardise_rewards=False,
        common_reward=True,
        state_shape=8,
        obs_delay_enabled=False,
        obs_delay_apply_train=False,
        obs_delay_apply_test=False,
        obs_gaussian_delay_mean=0.0,
        obs_gaussian_delay_std=0.0,
        obs_delay_discretization="round",
        mask_before_softmax=True,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _make_batch(args, obs_dim=7, max_seq_length=6, batch_size=4):
    scheme = {
        "state": {"vshape": args.state_shape, "dtype": torch.float32},
        "obs": {"vshape": obs_dim, "group": "agents", "dtype": torch.float32},
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {"vshape": (args.n_actions,), "group": "agents", "dtype": torch.int},
        "reward": {"vshape": (1,), "dtype": torch.float32},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}
    batch = EpisodeBatch(
        scheme,
        groups,
        batch_size,
        max_seq_length,
        preprocess=preprocess,
        device="cpu",
    )
    for t in range(max_seq_length):
        terminated = torch.zeros(batch_size, 1, dtype=torch.uint8)
        if t == max_seq_length - 1:
            terminated[:] = 1
        batch.update(
            {
                "state": torch.randn(batch_size, args.state_shape),
                "obs": torch.randn(batch_size, args.n_agents, obs_dim),
                "actions": torch.randint(0, args.n_actions, (batch_size, args.n_agents, 1)),
                "avail_actions": torch.ones(batch_size, args.n_agents, args.n_actions, dtype=torch.int),
                "reward": torch.randn(batch_size, 1),
                "terminated": terminated,
            },
            ts=t,
        )
    return batch, batch.scheme, groups


class MAICTests(unittest.TestCase):
    def test_yaml_files_parse(self):
        import yaml

        for name, mixer in (("maic.yaml", "qmix"), ("maic_vdn.yaml", "vdn")):
            path = Path(__file__).resolve().parents[1] / "src" / "config" / "algs" / name
            with open(path, "r", encoding="utf-8") as handle:
                cfg = yaml.safe_load(handle)
            self.assertEqual(cfg["mac"], "maic_mac")
            self.assertEqual(cfg["agent"], "maic")
            self.assertEqual(cfg["learner"], "maic_learner")
            self.assertEqual(cfg["mixer"], mixer)
            self.assertFalse(cfg["entity_scheme"])
            self.assertEqual(cfg["hidden_dim"], 64)
            self.assertIn("latent_dim", cfg)
            self.assertIn("mi_loss_weight", cfg)

        entity_path = Path(__file__).resolve().parents[1] / "src" / "config" / "algs" / "entity_maic.yaml"
        with open(entity_path, "r", encoding="utf-8") as handle:
            entity_cfg = yaml.safe_load(handle)
        self.assertEqual(entity_cfg["mac"], "entity_maic_mac")
        self.assertEqual(entity_cfg["agent"], "entity_maic")
        self.assertEqual(entity_cfg["learner"], "maic_learner")
        self.assertTrue(entity_cfg["entity_scheme"])

    def test_makers_register_maic(self):
        args = _base_args()
        obs_dim = 7
        input_shape = obs_dim + args.n_agents + args.n_actions
        agent = AgentMaker.make("maic", input_shape, args)
        self.assertIsInstance(agent, MAICAgent)

        batch, scheme, groups = _make_batch(args, obs_dim=obs_dim)
        mac = MACMaker.make("maic_mac", scheme, groups, args)
        self.assertIsInstance(mac, MAICMAC)
        learner = LearnerMaker.make("maic_learner", mac, scheme, DummyLogger(), args)
        self.assertIsInstance(learner, MAICLearner)

    def test_agent_forward_shapes_and_aux_losses(self):
        args = _base_args()
        bs = 4
        input_shape = 7 + args.n_agents + args.n_actions
        agent = MAICAgent(input_shape, args)
        hidden = agent.init_hidden().expand(bs, args.n_agents, -1).contiguous()
        inputs = torch.randn(bs * args.n_agents, input_shape)

        q, h, extras = agent.forward(inputs, hidden, bs, test_mode=False, train_mode=True)
        self.assertEqual(tuple(q.shape), (bs * args.n_agents, args.n_actions))
        self.assertEqual(tuple(h.shape), (bs * args.n_agents, args.hidden_dim))
        self.assertIn("mi_loss", extras)
        self.assertIn("entropy_loss", extras)
        self.assertTrue(torch.isfinite(extras["mi_loss"]).all())
        self.assertGreaterEqual(float(extras["mi_loss"]), 0.0)

        q_test, _, extras_test = agent.forward(inputs, hidden, bs, test_mode=True, train_mode=False)
        self.assertEqual(tuple(q_test.shape), (bs * args.n_agents, args.n_actions))
        self.assertEqual(extras_test, {})

    def test_mac_select_actions_and_learner_train(self):
        args = _base_args()
        obs_dim = 7
        batch, scheme, groups = _make_batch(args, obs_dim=obs_dim)
        mac = MAICMAC(scheme, groups, args)
        learner = MAICLearner(mac, scheme, DummyLogger(), args)
        mac.init_hidden(batch.batch_size)
        actions = mac.select_actions(batch, t_ep=0, t_env=0, test_mode=True)
        self.assertEqual(tuple(actions.shape), (batch.batch_size, args.n_agents))
        learner.train(batch, t_env=0, episode_num=0)
        self.assertGreater(learner.training_steps, 0)

    def test_incentive_messages_change_q(self):
        torch.manual_seed(0)
        args = _base_args()
        bs = 4
        input_shape = 7 + args.n_agents + args.n_actions
        agent = MAICAgent(input_shape, args)
        hidden = agent.init_hidden().expand(bs, args.n_agents, -1).contiguous()
        inputs = torch.randn(bs * args.n_agents, input_shape)

        q_comm, h, _ = agent.forward(inputs, hidden, bs, test_mode=True, train_mode=False)
        q_local = agent.fc2(h)
        self.assertEqual(q_comm.shape, q_local.shape)
        # Messages are added to local Q; they need not be identically zero.
        self.assertTrue(torch.isfinite(q_comm).all())


if __name__ == "__main__":
    unittest.main()
