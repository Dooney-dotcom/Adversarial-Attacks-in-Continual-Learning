import torch
import numpy as np

def get_device() -> torch.device:
    """Detects and returns the best available computing device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()

def set_reproducibility(seed: int):
    """Sets seeds for Torch and NumPy to ensure reproducible results."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def mask_logits(logits: torch.Tensor, allowed_classes: range) -> torch.Tensor:
    """
    Masks logits for classes not in the allowed range (setting them to -inf).
    Essential for Split-MNIST to prevent prediction of unseen classes.
    """
    mask = torch.full_like(logits, float('-inf'))
    mask[:, list(allowed_classes)] = 0
    return logits + mask