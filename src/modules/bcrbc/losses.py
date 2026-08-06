import torch
import torch.nn.functional as F


def reconstruction_loss(
    reconstructed_observations: torch.Tensor,
    target_observations: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    element_loss = F.mse_loss(
        reconstructed_observations,
        target_observations,
        reduction="none",
    ).mean(dim=-1, keepdim=True)
    return (element_loss * mask).sum() / mask.sum().clamp_min(1.0)


def retro_consistency_loss(
    corrected_agent_outputs: torch.Tensor,
    target_agent_outputs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    element_loss = F.mse_loss(
        corrected_agent_outputs,
        target_agent_outputs.detach(),
        reduction="none",
    ).mean(dim=-1, keepdim=True)
    return (element_loss * mask).sum() / mask.sum().clamp_min(1.0)


def message_reconstruction_loss(
    reconstructed_messages: torch.Tensor,
    messages: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct each agent's incoming messages from its agent output.

    Args:
        reconstructed_messages: [B, T, n, 1, d_msg].
        messages: [B, T, n, n-1, d_msg] the (teacher) per-sender messages. The
            target is the mean over senders (keepdim so it stays [B,T,n,1,d_msg]),
            detached. With a single agent there are no senders (n-1 == 0) and the
            loss is zero.
        mask: [B, T, n, 1, 1] valid-step mask (broadcasts over the feature dim).
    """
    if messages.size(3) == 0:
        return reconstructed_messages.sum() * 0.0
    target = messages.mean(dim=3, keepdim=True).detach()  # [B,T,n,1,d_msg]
    element_loss = F.mse_loss(
        reconstructed_messages,
        target,
        reduction="none",
    ).mean(dim=-1, keepdim=True)
    return (element_loss * mask).sum() / mask.sum().clamp_min(1.0)


def flow_matching_loss(
    predicted_z: torch.Tensor,
    target_z: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    element_loss = F.mse_loss(
        predicted_z,
        target_z.detach(),
        reduction="none",
    ).mean(dim=(-2, -1), keepdim=True)
    return (element_loss * mask).sum() / mask.sum().clamp_min(1.0)
