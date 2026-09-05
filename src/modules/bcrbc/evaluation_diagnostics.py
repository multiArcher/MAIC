"""Score actual online decisions without feeding oracle data to the controller."""

import json

import numpy as np
import torch


class EvaluationDiagnostics:
    def __init__(self, output_path, batch_size):
        self.output_path = output_path
        self.batch_size = batch_size
        self.episode_index = 0
        self.sums = [dict() for _ in range(batch_size)]

    @torch.no_grad()
    def record(self, runner, active):
        if not active:
            return
        for index in active:
            runner.parent_conns[index].send(("get_fresh_obs", None))
        fresh = [runner.parent_conns[index].recv() for index in active]
        mac = runner.mac
        observations = torch.as_tensor(
            np.asarray(fresh), device=mac.device, dtype=torch.float32
        )[:, None]
        z_true = mac.agent.encode_observations(observations)
        z_used = mac.decision_z[active]
        decoded = mac.agent.decode_observations(z_used)
        autoencoded = mac.agent.decode_observations(z_true)
        delivered = runner.batch["obs"][active, runner.t][:, None]
        generation_time = runner.batch["obs_gen_t"][active, runner.t, :, 0]
        masks = {
            "missing": generation_time < runner.t,
            "never_arrived": generation_time < 0,
            "stale_arrived": (generation_time >= 0)
            & (generation_time < runner.t),
        }
        errors = {
            "used_z_mse": (z_used - z_true).square().mean(dim=(-2, -1))[:, 0],
            "used_obs_mse": (decoded - observations).square().mean(-1)[:, 0],
            "tokenizer_mse": (autoencoded - observations).square().mean(-1)[:, 0],
            "stale_obs_mse": (delivered - observations).square().mean(-1)[:, 0],
        }
        errors["completion_gain"] = errors["stale_obs_mse"] - errors["used_obs_mse"]
        for row, index in enumerate(active):
            totals = self.sums[index]
            totals["agent_steps"] = totals.get("agent_steps", 0) + mac.n_agents
            for group, mask in masks.items():
                count_key = f"{group}_count"
                totals[count_key] = totals.get(count_key, 0) + mask[row].sum().item()
                for name, values in errors.items():
                    key = f"{group}_{name}_sum"
                    value = (values[row] * mask[row]).sum().item()
                    totals[key] = totals.get(key, 0.0) + value

    def finish(self, returns, lengths, wins):
        # Keep raw sums/counts: aggregation must not average per-episode MSEs.
        with self.output_path.open("a", encoding="utf-8") as stream:
            for index, totals in enumerate(self.sums):
                row = {
                    "episode": self.episode_index,
                    "environment": index,
                    "return": float(returns[index]),
                    "length": lengths[index],
                    "won": wins[index],
                    **totals,
                }
                stream.write(json.dumps(row) + "\n")
                self.episode_index += 1
        self.sums = [dict() for _ in range(self.batch_size)]
