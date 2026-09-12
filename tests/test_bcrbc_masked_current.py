"""Decision-path checks for fixed masked history and current-only completion."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from modules.bcrbc.bcrbc_model import BCRBCModel
from controllers.bcrbc_mac import BCRBCMAC


def make_args():
    config = yaml.safe_load((ROOT / "src/config/default.yaml").read_text())
    config.update(yaml.safe_load((ROOT / "src/config/algs/bcrbc_qmix.yaml").read_text()))
    config.update(n_agents=2, n_actions=3, state_shape=9, device="cpu",
                  use_cuda=False, batch_size=2, batch_size_run=2,
                  env_info={"episode_limit": 20}, bcrbc_d_model=16,
                  bcrbc_agent_output_dim=16, bcrbc_q_hidden_dim=16,
                  bcrbc_z_dim=4, bcrbc_num_z_tokens=2, bcrbc_depth=4,
                  bcrbc_heads=2, bcrbc_context_window=3)
    config["env_args"]["max_delay"] = 2
    return SimpleNamespace(**config)


class MaskedCurrentTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.args = make_args()
        self.model = BCRBCModel(7, 3, 2, self.args)
        self.obs = torch.randn(2, 6, 2, 7)
        self.actions = torch.nn.functional.one_hot(
            torch.randint(3, (2, 6, 2)), 3).float()
        self.missing = torch.ones(2, 6, 2, 1, 1, dtype=torch.bool)
        self.missing[:, 1::2, 0] = False
        self.noise = torch.randn(2, 6, 2, 2, 4)

    def run_model(self, obs=None, noise=None, aux=False):
        return self.model.forward_training(
            self.obs if obs is None else obs, self.actions, self.missing,
            completion_noise=self.noise if noise is None else noise,
            compute_aux=aux,
        )

    def test_generated_history_does_not_feed_later_decisions(self):
        with torch.no_grad():
            original = self.run_model()
            noise = self.noise.clone()
            noise[:, 0] += 5
            changed = self.run_model(noise=noise)
        torch.testing.assert_close(original["z"][:, 1:], changed["z"][:, 1:])
        torch.testing.assert_close(original["q_values"][:, 1:], changed["q_values"][:, 1:])

    def test_zero_solver_steps_use_masked_latents_without_noise(self):
        self.model.flow_steps = 0
        with torch.no_grad():
            original = self.run_model()
            changed = self.run_model(noise=self.noise + 100)
        torch.testing.assert_close(original["z"], original["history_z"])
        torch.testing.assert_close(original["q_values"], changed["q_values"])

    def test_missing_values_and_future_observations_cannot_leak(self):
        with torch.no_grad():
            original = self.run_model()
            hidden = torch.where(self.missing[..., 0], self.obs + 100, self.obs)
            changed = self.run_model(obs=hidden)
            torch.testing.assert_close(original["q_values"], changed["q_values"])
            future = self.obs.clone()
            future[:, 4:] += 100
            changed = self.run_model(obs=future)
            torch.testing.assert_close(original["q_values"][:, :4], changed["q_values"][:, :4])

    def test_batched_matches_independent_current_queries(self):
        with torch.no_grad():
            batched = self.run_model()
            history = batched["history_z"]
            condition = self.model.prepare_condition(history, self.actions)
            for step in range(6):
                current = slice(step, step + 1)
                single = self.model.complete_current(
                    history[:, current], self.actions[:, current],
                    self.missing[:, current], self.noise[:, current],
                    condition, start_t=step,
                )
                torch.testing.assert_close(single["z"], batched["z"][:, current])
                torch.testing.assert_close(single["q_values"], batched["q_values"][:, current])

    def test_history_is_prepared_once_and_target_skips_auxiliary_work(self):
        with patch.object(self.model, "prepare_condition", wraps=self.model.prepare_condition) as prepare:
            with patch.object(self.model.observation_decoder, "forward", side_effect=AssertionError("target decoded")):
                self.run_model()
            self.assertEqual(prepare.call_count, 1)

    def test_completion_and_auxiliary_losses_backpropagate(self):
        output = self.run_model(aux=True)
        loss = output["q_values"].square().mean()
        for name in ("z", "predicted_z"):
            loss = loss + (output[name] - output["target_z"]).square().mean()
        for name in ("reconstructed_observations", "masked_reconstructed_observations",
                     "generated_reconstructed_observations"):
            loss = loss + (output[name] - self.obs).square().mean()
        loss.backward()
        for module in (self.model.observation_encoder, self.model.transformer,
                       self.model.z_predictor, self.model.q_head):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertTrue(any(g.abs().sum() > 0 for g in gradients))

    def test_late_arrival_cache_matches_dense_current_generation(self):
        mac = BCRBCMAC({"obs": {"vshape": 7}}, {"agents": 2}, self.args)
        times = torch.tensor([
            [-1, -1], [0, -1], [0, 0], [3, 1], [2, 4],
            [3, 3], [6, 4], [5, 7], [6, 6],
        ])[None, :, :, None].expand(2, -1, -1, -1)
        truth = torch.randn(2, 9, 2, 7)
        batch = {
            "obs": truth.gather(1, times.clamp_min(0).expand_as(truth)),
            "obs_gen_t": times,
            "actions": torch.randint(3, (2, 9, 2, 1)),
            "avail_actions": torch.ones(2, 9, 2, 3),
        }
        cached_state = dense_state = None
        for end in range(1, 10):
            torch.manual_seed(end)
            mac.use_kv_cache = True
            with patch.object(mac.agent, "complete_current", wraps=mac.agent.complete_current) as complete:
                cached, cached_state = mac.history_forward(
                    batch, slice(end - 1, end), cached_state, generate=True,
                )
                self.assertEqual(complete.call_count, 1)
                self.assertEqual(complete.call_args.args[0].shape[1], 1)
            torch.manual_seed(end)
            mac.use_kv_cache = False
            dense, dense_state = mac.history_forward(
                batch, slice(end - 1, end), dense_state, generate=True,
            )
            torch.testing.assert_close(cached["z"], dense["z"], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(cached["q_values"], dense["q_values"], atol=1e-6, rtol=1e-5)
            obs, actions = mac._build_inputs(batch, slice(0, end))
            aligned, missing = mac._align_observations(obs, times[:, :end])
            with torch.no_grad():
                train_output = mac.agent.forward_training(
                    aligned, actions, missing,
                    completion_noise=cached_state["noise"], compute_aux=False,
                )
            torch.testing.assert_close(cached["q_values"], train_output["q_values"][:, -1:],
                                       atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
