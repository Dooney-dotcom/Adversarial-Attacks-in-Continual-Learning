import os
import torch
from torchvision import datasets, transforms
from aacl.config import Config

def add_backdoor_fixed(img_tensor: torch.Tensor) -> torch.Tensor:
    """
    Injects a 4x4 pixel white square trigger in the bottom-right corner.
    Handles both single image (3D) and batch (4D) tensors.
    """
    if img_tensor.dim() == 4: 
        _, _, h, w = img_tensor.shape
        img_tensor[:, :, h-5:h-1, w-5:w-1] = 1.0
    else: 
        _, h, w = img_tensor.shape
        img_tensor[:, h-5:h-1, w-5:w-1] = 1.0
    return img_tensor

def get_base_transform():
    """Standard transform for evaluation and clean training."""
    return transforms.Compose([transforms.ToTensor()])

def get_defense_transform():
    """
    AACL Defense Transform:
    RandomResizedCrop forces the model to learn global features, often
    cropping out the localized trigger in the corner.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(size=28, scale=(0.75, 1.0)), 
        transforms.RandomErasing(p=0.2),
    ])

def get_dataset():
    """Loads MNIST. Downloads if necessary."""
    cfg = Config()
    tr_t = get_base_transform()
    
    if not os.path.exists(os.path.join(cfg.DATA_ROOT, "MNIST")):
        datasets.MNIST(cfg.DATA_ROOT, train=True, download=True)
        
    # We return the raw train dataset (without transform) so we can apply
    # custom augmentations/backdoors dynamically in the loop.
    train_ds_raw = datasets.MNIST(cfg.DATA_ROOT, train=True, download=False, transform=None) 
    test_ds = datasets.MNIST(cfg.DATA_ROOT, train=False, download=False, transform=tr_t)
    
    return train_ds_raw, test_ds