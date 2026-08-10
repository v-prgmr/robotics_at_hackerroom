import torch


def masked_weighted_mean(
    elementwise_loss: torch.Tensor,
    valid_mask: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average over valid elements while preserving reward-weight scale."""
    if elementwise_loss.ndim < 3:
        raise ValueError("Expected elementwise loss with shape [B, T, ...]")
    if valid_mask.shape != elementwise_loss.shape[:2]:
        raise ValueError("Action validity mask must have shape [B, T]")
    if weight is not None and weight.shape != valid_mask.shape:
        raise ValueError("TOPReward weight must have shape [B, T]")

    valid = valid_mask.to(device=elementwise_loss.device, dtype=elementwise_loss.dtype)
    factors = valid if weight is None else valid * weight.to(elementwise_loss)
    while factors.ndim < elementwise_loss.ndim:
        factors = factors.unsqueeze(-1)
    feature_count = elementwise_loss[0, 0].numel()
    denominator = valid.sum() * feature_count
    if denominator.item() <= 0:
        raise ValueError("Cannot compute loss for a batch with no valid actions")
    return (elementwise_loss * factors).sum() / denominator
