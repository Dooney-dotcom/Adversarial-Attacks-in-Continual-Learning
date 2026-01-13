from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    """Hyperparameters for the CL experiment."""
    seed: int = 42
    lr: float = 0.001 # Learning Rate
    batch_size: int = 64
    epochs_per_task: int = 10
    
    # EWC Hyperparameters
    ewc_lambda: float = 5_000
    
    # Adversarial / Attack Hyperparameters
    attack_eps: float = 0.3
    attack_steps: int = 5
    attack_lr: float = 0.1
    
    # TABA Hyperparameters
    memory_size: int = 500       # Size of buffer for Task A
    taba_mix_alpha: float = 0.5  # Center of mixing range [0.45, 0.55]