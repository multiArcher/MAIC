"""Paired action diagnostics on the generated policy's actual trajectory."""

import json

import numpy as np
import torch


def action_scores(reference_q, candidate_q, available_actions):
    """Compare greedy actions and KL(reference || candidate), using softmax(Q)."""
    legal = available_actions.bool()
    reference_q = reference_q.float().masked_fill(~legal, -1e9)
    candidate_q = candidate_q.float().masked_fill(~legal, -1e9)
    reference_logp = reference_q.log_softmax(dim=-1)
    candidate_logp = candidate_q.log_softmax(dim=-1)
    reference_action = reference_q.argmax(dim=-1)
    candidate_action = candidate_q.argmax(dim=-1)
    kl = (reference_logp.exp() * (reference_logp - candidate_logp)).sum(dim=-1)
    kl = kl.clamp_min(0.0)  # Round-off can otherwise give a tiny negative KL.
    return {
        "agreement": candidate_action == reference_action,
        "kl": kl,
        "reference_action": reference_action,
        "candidate_action": candidate_action,
    }


class EvaluationDiagnostics:
    def __init__(self, output_path, batch_size, episode_offset=0):
        self.output_path = output_path
        self.batch_size = batch_size
        self.episode_index = episode_offset
        self.totals = {}
        self._reset_histories()

    def _reset_histories(self):
        self.sums = [dict() for _ in range(self.batch_size)]
        self.trajectories = [[] for _ in range(self.batch_size)]
        self.delay_histograms = [dict() for _ in range(self.batch_size)]
        self.reference_encoder_cache = None
        self.reference_dynamics_cache = None
        self.mask_state = None

    @torch.no_grad()
    def record(self, runner, active):
        if not active:
            return
        mac = runner.mac
        step = runner.t
        data = runner.get_diagnostic_data(active)
        # Full local observations are confined to this shadow branch.
        observations = torch.zeros_like(runner.batch["obs"][:, step:step + 1])
        observations[active, 0] = torch.as_tensor(
            np.asarray([item["observations"] for item in data]),
            device=mac.device, dtype=observations.dtype,
        )
        _, previous_actions = mac._build_inputs(
            runner.batch, slice(step, step + 1),
        )
        reference_z, encoder_cache = mac.agent.encode_observations(
            observations, kv_cache=self.reference_encoder_cache,
            use_kv_cache=True, rope_offset=step,
        )
        reference = mac.agent.estimate_clean_z(
            reference_z, previous_actions,
            torch.ones_like(reference_z[..., :1, :1]), start_t=step,
            kv_cache=self.reference_dynamics_cache,
            use_kv_cache=True, rope_offset=step,
        )
        self.reference_encoder_cache = mac._trim_cache(encoder_cache)
        self.reference_dynamics_cache = mac._trim_cache(reference["kv_cache"])
        mask_output, self.mask_state = mac.history_forward(
            runner.batch, slice(step, step + 1), self.mask_state, generate=False,
        )
        reference_q = reference["q_values"][:, 0, :, 0]
        mask_q = mask_output["q_values"][:, 0, :, 0]
        available = runner.batch["avail_actions"][:, step]
        generation_time = runner.batch["obs_gen_t"][:, step, :, 0]
        missing = generation_time < step
        eligible = missing & (available.sum(dim=-1) > 1)
        scores = {
            "generated": action_scores(reference_q, mac.decision_q_values, available),
            "mask": action_scores(reference_q, mask_q, available),
        }
        for row, index in enumerate(active):
            totals = self.sums[index]
            totals["decision_count"] = (
                totals.get("decision_count", 0) + eligible[index].sum().item()
            )
            for path, values in scores.items():
                for metric in ("agreement", "kl"):
                    key = f"{path}_{metric}_sum"
                    total = (values[metric][index] * eligible[index]).sum().item()
                    totals[key] = totals.get(key, 0.0) + total
            histogram = self.delay_histograms[index]
            for delay in data[row]["sampled_delays"]:
                key = str(delay)
                histogram[key] = histogram.get(key, 0) + 1
            if self.output_path is not None and self.episode_index + index < 4:
                self.trajectories[index].append({
                    "step": step,
                    "reference": scores["generated"]["reference_action"][index].tolist(),
                    "generated": scores["generated"]["candidate_action"][index].tolist(),
                    "mask": scores["mask"]["candidate_action"][index].tolist(),
                    "actual": runner.batch["actions"][index, step, :, 0].tolist(),
                    "missing": missing[index].tolist(),
                    "eligible": eligible[index].tolist(),
                })

    def finish(self, returns, lengths, wins):
        for totals in self.sums:
            for key, value in totals.items():
                self.totals[key] = self.totals.get(key, 0) + value
        if self.output_path is not None:
            with self.output_path.open("a", encoding="utf-8") as stream:
                for index, totals in enumerate(self.sums):
                    row = {
                        "episode": self.episode_index + index,
                        "return": float(returns[index]),
                        "length": int(lengths[index]),
                        "won": bool(wins[index]),
                        "sampled_delay_histogram": self.delay_histograms[index],
                        **totals,
                    }
                    stream.write(json.dumps(row) + "\n")
            trajectory_path = self.output_path.parent / "trajectories.jsonl"
            with trajectory_path.open("a", encoding="utf-8") as stream:
                for index, steps in enumerate(self.trajectories):
                    if steps:
                        stream.write(json.dumps({
                            "episode": self.episode_index + index, "steps": steps,
                        }) + "\n")
        self.episode_index += self.batch_size
        self._reset_histories()

    def log(self, logger, t_env):
        count = self.totals.get("decision_count", 0)
        logger.log_stat("test_action/decision_count", count, t_env)
        if count:
            for path in ("generated", "mask"):
                for metric in ("agreement", "kl"):
                    value = self.totals[f"{path}_{metric}_sum"] / count
                    logger.log_stat(f"test_action/{path}_{metric}", value, t_env)
        self.totals.clear()
