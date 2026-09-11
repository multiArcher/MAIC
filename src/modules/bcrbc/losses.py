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
