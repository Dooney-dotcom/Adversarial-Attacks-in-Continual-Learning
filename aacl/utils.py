import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from aacl.config import Config
from aacl.data import add_backdoor_fixed

def set_seed(seed: int = 42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"=== SEED FIXED AT {seed} ===")

def evaluate_full(learner, test_ds):
    """
    Evaluates the learner on:
    1. Task 1 Accuracy (Classes 0-4) - Stability
    2. Task 2 Accuracy (Classes 5-9) - Plasticity
    3. Backdoor Attack Success Rate (ASR) - Robustness
    """
    cfg = Config()
    learner.model.eval()
    
    # --- 1. Task 1 Accuracy ---
    idx_t1 = [i for i, t in enumerate(test_ds.targets) if t < cfg.CLASSES_PER_TASK]
    loader_t1 = DataLoader(Subset(test_ds, idx_t1), batch_size=128)
    corr_t1, tot_t1 = 0, 0
    
    # --- 2. Task 2 Accuracy ---
    # Only check if the model has actually seen Task 2 classes
    has_t2 = cfg.CLASSES_PER_TASK in learner.seen_classes
    idx_t2 = [i for i, t in enumerate(test_ds.targets) if t >= cfg.CLASSES_PER_TASK]
    loader_t2 = DataLoader(Subset(test_ds, idx_t2), batch_size=128) if has_t2 else None
    corr_t2, tot_t2 = 0, 0

    # --- 3. Backdoor ASR ---
    # Calculate success rate on non-target images injected with the trigger
    idx_bd = [i for i, t in enumerate(test_ds.targets) if t != cfg.BACKDOOR_TARGET]
    loader_bd = DataLoader(Subset(test_ds, idx_bd), batch_size=128)
    bd_suc, bd_tot = 0, 0

    with torch.no_grad():
        # Evaluate T1
        for x, y in loader_t1:
            x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
            corr_t1 += (learner.model(x).argmax(1) == y).sum().item()
            tot_t1 += y.size(0)
        
        # Evaluate T2
        if has_t2 and loader_t2:
            for x, y in loader_t2:
                x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
                corr_t2 += (learner.model(x).argmax(1) == y).sum().item()
                tot_t2 += y.size(0)

        # Evaluate Backdoor
        for x, y in loader_bd:
            x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
            # Inject trigger dynamically during test
            x_pois = add_backdoor_fixed(x.clone())
            # Success if model predicts the Backdoor Target
            bd_suc += (learner.model(x_pois).argmax(1) == cfg.BACKDOOR_TARGET).sum().item()
            bd_tot += x.size(0)

    acc_t1 = 100 * corr_t1 / tot_t1 if tot_t1 > 0 else 0.0
    acc_t2 = 100 * corr_t2 / tot_t2 if tot_t2 > 0 else 0.0
    asr = 100 * bd_suc / bd_tot if bd_tot > 0 else 0.0

    return acc_t1, acc_t2, asr