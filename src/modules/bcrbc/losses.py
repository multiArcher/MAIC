import torch
import torch.nn.functional as F


def reconstruction_loss(recon_obs: torch.Tensor, target_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Masked per-agent MSE between the obs decoded from z and the raw obs target.
    per_agent = F.mse_loss(recon_obs, target_obs, reduction="none").mean(dim=-1, keepdim=True)
    return (per_agent * mask).sum() / mask.sum().clamp_min(1.0)


def dynamics_loss(pred_latents: torch.Tensor, latents: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Legacy MLP next-latent predictor (superseded by flow_matching_loss). Aligns the
    # prediction at step t to the (stop-grad) latent at t+1. A single-step window has
    # no next-step target, so the loss is a structural zero.
    if pred_latents.size(1) <= 1:
        return pred_latents.sum() * 0.0
    pred = pred_latents[:, :-1]
    target = latents[:, 1:].detach()
    dynamics_mask = mask[:, : pred.size(1)].unsqueeze(-1)
    loss = F.mse_loss(pred, target, reduction="none").mean(dim=-1, keepdim=True)
    return (loss * dynamics_mask).sum() / dynamics_mask.sum().clamp_min(1.0)


def retro_consistency_loss(
    corrected_beliefs: torch.Tensor,
    target_beliefs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    # Align the online (delayed) beliefs to the zero-delay teacher beliefs (stop-grad).
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



