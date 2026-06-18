import torch
import torch.nn.functional as F


def reconstruction_loss(recon_obs: torch.Tensor, target_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_agent = F.mse_loss(recon_obs, target_obs, reduction="none").mean(dim=-1, keepdim=True)
    return (per_agent * mask).sum() / mask.sum().clamp_min(1.0)


def dynamics_loss(pred_latents: torch.Tensor, latents: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred_latents.size(1) <= 1:
        return pred_latents.sum() * 0.0
    pred = pred_latents[:, :-1]
    target = latents[:, 1:].detach()
    dyn_mask = mask[:, : pred.size(1)].unsqueeze(-1)
    loss = F.mse_loss(pred, target, reduction="none").mean(dim=-1, keepdim=True)
    return (loss * dyn_mask).sum() / dyn_mask.sum().clamp_min(1.0)


def retro_consistency_loss(
    corrected_beliefs: torch.Tensor,
    target_beliefs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    per_agent = F.mse_loss(corrected_beliefs, target_beliefs.detach(), reduction="none").mean(dim=-1, keepdim=True)
    return (per_agent * mask).sum() / mask.sum().clamp_min(1.0)


def message_reconstruction_loss(
    recon_msg: torch.Tensor,
    messages: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct each agent's incoming messages from its belief.

    Args:
        recon_msg: [B, T, n, 1, d_msg] decoded message content from the belief.
        messages: [B, T, n, n-1, d_msg] the (teacher) per-sender messages. The
            target is the mean over senders (keepdim so it stays [B,T,n,1,d_msg]),
            detached. With a single agent there are no senders (n-1 == 0) and the
            loss is zero.
        mask: [B, T, n, 1, 1] valid-step mask (broadcasts over the feature dim).
    """
    if messages.size(3) == 0:
        return recon_msg.sum() * 0.0
    target = messages.mean(dim=3, keepdim=True).detach()  # [B,T,n,1,d_msg]
    per_agent = F.mse_loss(recon_msg, target, reduction="none").mean(dim=-1, keepdim=True)
    return (per_agent * mask).sum() / mask.sum().clamp_min(1.0)


def arrival_loss(
    delay_logits: torch.Tensor,
    delay_target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Classify the (bucketized) delay of each agent's observation.

    Rank-agnostic: works whether the per-agent vector axis is present or not, as
    long as ``delay_logits`` is [..., buckets], ``delay_target`` is [..., 1], and
    ``mask`` broadcasts to [..., 1].

    Args:
        delay_logits: [B, T, n, 1, max_delay + 1] per-agent bucket logits.
        delay_target: [B, T, n, 1, 1] true delay bucket (long).
        mask: [B, T, n, 1, 1] valid-step mask.
    """
    buckets = delay_logits.size(-1)
    target = delay_target.long().clamp(min=0, max=buckets - 1)  # [..., 1]
    logits_flat = delay_logits.reshape(-1, buckets)
    target_flat = target.reshape(-1)
    ce = F.cross_entropy(logits_flat, target_flat, reduction="none")
    ce = ce.reshape(target.shape)  # [..., 1], matches the target/mask layout
    return (ce * mask).sum() / mask.sum().clamp_min(1.0)


def flow_matching_loss(
    flow_dynamics,
    context: torch.Tensor,
    z_target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Next-step flow-matching world-model loss.

    Conditions on the per-agent belief summary at step t (``context``) to predict
    the clean bottleneck latent at step t+1 (``z_target``). The time shift is
    applied by the caller; this wrapper just forwards to the flow head's
    x-prediction loss. Replaces the weak MLP ``dynamics_loss``.

    Args:
        flow_dynamics: FlowDynamics module.
        context: [B, T-1, n, context_dim] belief at steps 0..T-2.
        z_target: [B, T-1, n, z_dim] bottleneck latent at steps 1..T-1.
        mask: [B, T-1, n, 1] valid-step mask.
    """
    return flow_dynamics.flow_loss(context, z_target, mask)



