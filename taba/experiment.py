import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
from typing import Tuple
from torch.utils.data import DataLoader
import pandas as pd


from taba.config import ExperimentConfig
from taba.utils import DEVICE, set_reproducibility, mask_logits
from taba.data import DataHandler
from taba.model import MLP
from taba.ewc import EWC
from taba.trainer import TABATrainer, EWCTrainer
from taba.attacks import Attacker

def evaluate(model: torch.nn.Module, loader: DataLoader, desc: str) -> Tuple[float, float]:
    """Evaluates Standard Accuracy (SA) and Robust Accuracy (RA)."""
    model.eval()
    
    # SA
    correct_sa = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            pred = model(data).argmax(dim=1)
            correct_sa += pred.eq(target).sum().item()
            total += len(data)
    sa = 100. * correct_sa / total

    # RA (Fast check)
    correct_ra = 0
    total = 0
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        data_adv = Attacker.pgd_attack(model, data, target, eps=0.3, steps=7)
        with torch.no_grad():
            pred = model(data_adv).argmax(dim=1)
            correct_ra += pred.eq(target).sum().item()
            total += len(data)
    ra = 100. * correct_ra / total
    
    return sa, ra

def run_taba_experiment():
    cfg = ExperimentConfig()
    set_reproducibility(cfg.seed)
    print(f"Running Experiment on device: {DEVICE}")
    
    # Data Setup
    data_splits = DataHandler.get_split_mnist()
    dl_train_A = DataLoader(data_splits['train_A'], batch_size=cfg.batch_size, shuffle=True)
    dl_train_B = DataLoader(data_splits['train_B'], batch_size=cfg.batch_size, shuffle=True)
    dl_test_A = DataLoader(data_splits['test_A'], batch_size=1000)
    dl_test_B = DataLoader(data_splits['test_B'], batch_size=1000)
    
    # Initialize Model
    model = MLP().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
    
    # --- PHASE 1: Learn Task A ---
    print("\n=== PHASE 1: Training Task A (Clean) ===")
    for ep in range(cfg.epochs_per_task):
        model.train()
        for data, target in dl_train_A:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            out = mask_logits(model(data), range(0, 5))
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
    
    sa_A_init, ra_A_init = evaluate(model, dl_test_A, "Task A")
    print(f"Baseline Task A -> SA: {sa_A_init:.2f}% | RA: {ra_A_init:.2f}%")
    
    # Checkpoint & Memory
    model_state_after_A = copy.deepcopy(model.state_dict())
    ewc = EWC(model, dl_train_A) 
    memory_buffer_A = DataHandler.create_memory_buffer(data_splits['train_A'], cfg.memory_size)
    
    # =========================================================================
    # SCENARIO 1: EWC BASELINE (NO ATTACK)
    # =========================================================================
    print("\n=== SCENARIO 1: Task B (Standard EWC - No Attack) ===")
    model.load_state_dict(model_state_after_A) # Reset model
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
    
    for ep in range(cfg.epochs_per_task):
        # attack=False
        loss = EWCTrainer.train_epoch(
            model, dl_train_B, optimizer, ewc, memory_buffer_A, cfg, attack=False
        )
        if (ep+1) % 5 == 0: print(f"Ep {ep+1} Loss: {loss:.4f}")
        
    sa_A_clean, ra_A_clean = evaluate(model, dl_test_A, "Task A (Clean EWC)")
    sa_B_clean, ra_B_clean = evaluate(model, dl_test_B, "Task B (Clean EWC)")

    # =========================================================================
    # SCENARIO 2: ATTACK ONLY (Baseline Catastrophic Forgetting)
    # =========================================================================
    print("\n=== SCENARIO 2: Task B with ATTACK (No Defense) ===")
    model.load_state_dict(model_state_after_A) # Reset model
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
    
    for ep in range(cfg.epochs_per_task):
        # attack=True
        loss = EWCTrainer.train_epoch(
            model, dl_train_B, optimizer, ewc, memory_buffer_A, cfg, attack=True
        )
        if (ep+1) % 5 == 0: print(f"Ep {ep+1} Loss: {loss:.4f}")

    sa_A_pois, ra_A_pois = evaluate(model, dl_test_A, "Task A (Poisoned)")
    sa_B_pois, _ = evaluate(model, dl_test_B, "Task B (Poisoned)")

    # =========================================================================
    # SCENARIO 3: TABA DEFENSE
    # =========================================================================
    print("\n=== SCENARIO 3: Task B with ATTACK + TABA Defense ===")
    model.load_state_dict(model_state_after_A) # Reset model AGAIN
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
    
    for ep in range(cfg.epochs_per_task):
        loss = TABATrainer.train_epoch(
            model, dl_train_B, optimizer, ewc, memory_buffer_A, cfg
        )
        if (ep+1) % 5 == 0: print(f"Ep {ep+1} Loss: {loss:.4f}")
        
    sa_A_taba, ra_A_taba = evaluate(model, dl_test_A, "Task A (TABA)")
    sa_B_taba, ra_B_taba = evaluate(model, dl_test_B, "Task B (TABA)")

    # --- FINAL COMPARISON TABLE ---
    results = pd.DataFrame(
        columns = ['TASK', 'METRIC', 'EWC BASELINE (NO ATTACK)', 'EWC ATTACKED', 'TABA DEFENSE'],
        data = [
            ['Task A', 'SA (Clean)', f"{sa_A_clean:.2f}%", f"{sa_A_pois:.2f}%", f"{sa_A_taba:.2f}%"],
            ['Task A', 'RA (Robust)', f"{ra_A_clean:.2f}%", f"{ra_A_pois:.2f}%", f"{ra_A_taba:.2f}%"],
            ['Task B', 'SA (Clean)', f"{sa_B_clean:.2f}%", f"{sa_B_pois:.2f}%", f"{sa_B_taba:.2f}%"],
            ['Task B', 'RA (Robust)', f"{ra_B_clean:.2f}%", f"--", f"{ra_B_taba:.2f}%"],
        ]
    )

    print("\n=== FINAL RESULTS COMPARISON ===")
    print(results.to_string(index=False))
    print("="*100)
