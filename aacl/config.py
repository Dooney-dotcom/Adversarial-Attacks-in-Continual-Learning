import torch
from dataclasses import dataclass

@dataclass
class Config:
    # System
    SEED: int = 42
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_ROOT: str = './data'

    # Task Settings
    NUM_TASKS: int = 2
    CLASSES_PER_TASK: int = 5
    TOTAL_CLASSES: int = 10
    MEMORY_SIZE: int = 2000
    
    # Training Hyperparameters
    EPOCHS: int = 10
    BATCH_SIZE: int = 64
    LR: float = 0.01 # Learning Rate
    MOMENTUM: float = 0.9

    # Backdoor Attack Settings
    # Trigger is fixed in the bottom-right corner for 100% ASR on vulnerable models
    BACKDOOR_RATE: float = 0.20
    BACKDOOR_TARGET: int = 0

    # Defense (AACL) Hyperparameters
    AACL_LAMBDA: float = 2.0  # Weight for Contrastive Loss
    DIST_LAMBDA: float = 4.0  # Weight for Distillation Loss
    AACL_TEMP: float = 0.1    # Temperature for SimCLR