"""Evaluation-only packet distributions; arrival handling stays in DelayModel."""
import torch

from components.delay_model import DelayModel


class EvaluationDelay(DelayModel):
    def __init__(self, condition, seed):
        super().__init__(max_delay=condition["cap"])
        self.condition = condition
        self.generator = torch.Generator().manual_seed(seed)
        self.reset()

    def reset(self):
        super().reset()
        self.step = 0
        self.regime = 0
        self.regime_age = 0
        self.clipped_count = 0

    def _draw(self, shape, device):
        c = self.condition
        kind = c["kind"]
        if kind == "fixed":
            raw = torch.full(shape, float(c["value"]), device=device)
        elif kind == "uniform":
            raw = torch.randint(c["low"], c["high"] + 1, shape,
                                generator=self.generator, device=device).float()
        else:
            mean, std = c.get("mean", 0), c.get("std", 1)
            if kind in ("mixture", "periodic", "markov"):
                previous = self.regime
                # Regime is shared by agents in an environment; packet noise is independent.
                if kind == "mixture":
                    self.regime = int(torch.rand((), generator=self.generator) < c["high_probability"])
                elif kind == "periodic":
                    self.regime = (self.step // c["period"]) % 2
                elif self.step == 0:
                    self.regime = int(torch.rand((), generator=self.generator) < 0.5)
                elif torch.rand((), generator=self.generator) > c["stay_probability"]:
                    self.regime = 1 - self.regime
                self.regime_age = self.regime_age + 1 if self.step and previous == self.regime else 0
                mean = c["means"][self.regime]
                std = c["stds"][self.regime]
            raw = mean + std * torch.randn(shape, generator=self.generator, device=device)
        rounded = raw.clamp_min(0).ceil()
        self.clipped_count = int((rounded > self.max_delay).sum())
        self.step += 1
        return rounded.clamp_max(self.max_delay).long()
