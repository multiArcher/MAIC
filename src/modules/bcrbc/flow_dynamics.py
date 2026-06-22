"""Flow-matching world model head for BC-RBC.

A lightweight, amortized flow-matching ("x-prediction") generator over the
bottleneck latent ``z``. Adapted from DreamerV4's generative dynamics
(``previous_works/dreamerv4_dynamics.py``) but **without** the shortcut/bootstrap
machinery: the latent is low-dimensional, so a handful of Euler steps is cheap
and plain flow matching suffices.

Amortized design: the block-causal transformer already summarizes history into a
per-(agent, step) context vector (the belief readout). This head conditions on
that context plus a flow-time embedding ``tau in [0, 1]`` and predicts the clean
next latent ``z_{t+1}`` from a noised input ``x_tau = (1 - tau) * noise + tau * z``.

Notation (kept distinct on purpose):
- env time ``t``: the MDP step. Delay lives here.
- flow time ``tau in [0, 1]``: internal to generation only. ``tau = 0`` is pure
  noise, ``tau = 1`` is a finished latent. Walking ``tau`` from 0 to 1 produces
  one ``z``. It has nothing to do with the environment or with delay.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowDynamics(nn.Module):
    """Amortized flow-matching head: context (+ tau) -> clean next latent z.

    The model predicts the clean target ``z_1`` directly (x-prediction). The
    flow velocity used for sampling is recovered as ``v = (z_1_pred - x_tau) /
    (1 - tau)``, which is the constant-velocity straight-line interpolant of
    rectified flow / flow matching.
    """

    def __init__(self, context_dim: int, latent_dim: int, hidden_dim: int,
                 flow_time_embed_dim: int, num_flow_time_buckets: int, num_latent_tokens: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.num_latent_tokens = num_latent_tokens
        self.num_flow_time_buckets = num_flow_time_buckets
        # Discrete flow-time embedding (lookup), matching DreamerV4's discrete tau.
        self.tau_embed = nn.Embedding(num_flow_time_buckets + 1, flow_time_embed_dim)
        # Per-token query embedding so the num_latent_tokens latents generated from the
        # same per-agent context can differ (homogeneous-capacity tokens). Added
        # to the context along an explicit token axis.
        self.token_query = nn.Parameter(torch.zeros(num_latent_tokens, context_dim))
        # With a single latent token the query is a pure additive zero, so leave it at
        # zero; only break the symmetry between tokens when there is more than one.
        if num_latent_tokens > 1:
            nn.init.normal_(self.token_query, std=0.02)
        self.net = nn.Sequential(
            nn.Linear(context_dim + latent_dim + flow_time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def _with_token_axis(self, context: torch.Tensor) -> torch.Tensor:
        """Broadcast the per-agent size-1 slot axis up to num_latent_tokens, add per-token query.

        The belief context arrives with the explicit size-1 vector axis
        ``[..., 1, context_dim]`` (the per-agent slot). Adding the learned
        ``token_query`` (shape ``[num_latent_tokens, context_dim]``) broadcasts that axis
        to ``[..., num_latent_tokens, context_dim]`` so each generated token differs. Pure
        broadcast; no reshape of the batch prefix. num_latent_tokens=1 is an additive zero.
        """
        return context + self.token_query  # [...,1,ctx] + [num_latent_tokens,ctx] -> [...,num_latent_tokens,ctx]

    def _tau_ids(self, tau: torch.Tensor) -> torch.Tensor:
        return (tau.clamp(0.0, 1.0) * self.num_flow_time_buckets).round().long().clamp(0, self.num_flow_time_buckets)

    def predict_z1(self, context: torch.Tensor, x_tau: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Predict the clean latent z_1 from a noised latent x_tau.

        Args:
            context: [..., context_dim] history summary (already carries the token
                axis if one is present).
            x_tau: [..., latent_dim] noised latent at flow-time tau.
            tau: [..., 1] flow-time in [0, 1].
        Returns:
            z_1 prediction [..., latent_dim].
        """
        tau_emb = self.tau_embed(self._tau_ids(tau).squeeze(-1))
        net_in = torch.cat([context, x_tau, tau_emb], dim=-1)
        return self.net(net_in)

    def flow_loss(self, context: torch.Tensor, z_target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """X-prediction flow-matching loss (no bootstrap).

        Draws a random flow-time tau and noise, forms the straight-line
        interpolant x_tau, and trains the network to recover the clean target.
        A ramp weight (0.9*tau + 0.1) emphasizes later (cleaner) flow-times,
        matching DreamerV4.

        Args:
            context: [..., 1, context_dim] per-agent belief summary carrying the
                size-1 slot axis; broadcast to num_latent_tokens internally.
            z_target: [..., num_latent_tokens, latent_dim] clean target latents (stop-grad here).
            mask: optional [..., 1, 1] valid-step mask (broadcasts over num_latent_tokens).
        """
        z1 = z_target.detach()
        ctx = self._with_token_axis(context)  # [..., num_latent_tokens, context_dim]
        tau = torch.rand(*z1.shape[:-1], 1, device=z1.device, dtype=z1.dtype)
        noise = torch.randn_like(z1)
        x_tau = (1.0 - tau) * noise + tau * z1
        z1_pred = self.predict_z1(ctx, x_tau, tau)
        per = F.mse_loss(z1_pred, z1, reduction="none")
        ramp = 0.9 * tau + 0.1
        per = per * ramp
        if mask is not None:
            per = per.mean(dim=-1, keepdim=True) * mask
            return per.sum() / mask.sum().clamp_min(1.0)
        return per.mean()

    @torch.no_grad()
    def sample(self, context: torch.Tensor, steps: int = 8) -> torch.Tensor:
        """Generate clean latent(s) z by integrating the flow from noise.

        Euler integration of the recovered velocity over flow-time tau: 0 -> 1.

        Args:
            context: [..., 1, context_dim] per-agent belief carrying the size-1
                slot axis; broadcast to num_latent_tokens internally.
            steps: number of Euler steps K (4-16 is plenty for low-dim z).
        Returns:
            generated latent z [..., num_latent_tokens, latent_dim].
        """
        context = self._with_token_axis(context)  # [..., num_latent_tokens, context_dim]
        shape = (*context.shape[:-1], self.latent_dim)
        x = torch.randn(shape, device=context.device, dtype=context.dtype)
        dt = 1.0 / steps
        for k in range(steps):
            tau_val = torch.full((*context.shape[:-1], 1), k * dt, device=context.device, dtype=context.dtype)
            z1_pred = self.predict_z1(context, x, tau_val)
            # straight-line velocity toward the predicted clean target
            v = (z1_pred - x) / (1.0 - tau_val).clamp_min(1e-6)
            x = x + v * dt
        return x
