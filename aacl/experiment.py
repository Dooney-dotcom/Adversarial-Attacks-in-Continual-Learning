import torch
import torch.optim as optim
import copy
import pandas as pd

from aacl.config import Config
from aacl.data import get_dataset
from aacl.model import SimpleCNN_AACL
from aacl.trainer import UnifiedLearner
from aacl.utils import set_seed, evaluate_full

def run_aacl_experiment():
    cfg = Config()
    set_seed(cfg.SEED)
    
    print(f"Device: {cfg.DEVICE}")
    train_ds, test_ds = get_dataset()
    
    print("\n=== STARTING COMPARATIVE WORKFLOW ===")
    
    # --- PHASE 1: BASELINE (TASK 1 CLEAN) ---
    model = SimpleCNN_AACL(num_classes=cfg.TOTAL_CLASSES).to(cfg.DEVICE)
    learner = UnifiedLearner(model)
    
    print("\n[Phase 1] Training Task 1 (Standard)...")
    learner.train_task(train_ds, 0, mode='standard')
    t1_acc, _, _ = evaluate_full(learner, test_ds)
    print(f"--> Baseline Task 1 Accuracy: {t1_acc:.2f}%")
    
    # Save State (Time Travel Checkpoint)
    checkpoint = {
        'model': copy.deepcopy(model.state_dict()),
        'memory': copy.deepcopy(learner.memory),
        'seen': copy.deepcopy(learner.seen_classes)
    }
    
    # --- PHASE 2: SCENARIO A (VULNERABLE) ---
    print("\n" + "!"*60)
    print("SCENARIO A: Task 2 VULNERABLE (No Defense)")
    print("!"*60)
    
    learner.train_task(train_ds, 1, mode='vulnerable')
    vuln_t1, vuln_t2, vuln_asr = evaluate_full(learner, test_ds)
    
    # --- PHASE 3: SCENARIO B (DEFENSE - AACL) ---
    print("\n" + "="*60)
    print("SCENARIO B: Task 2 DEFENSE (AACL + Distillation)")
    print("="*60)
    
    # Reload Checkpoint (Reset to end of Task 1)
    learner.model.load_state_dict(checkpoint['model'])
    learner.memory = copy.deepcopy(checkpoint['memory'])
    learner.seen_classes = copy.deepcopy(checkpoint['seen'])
    # Reset Optimizer
    learner.optimizer = optim.SGD(learner.model.parameters(), lr=cfg.LR, momentum=cfg.MOMENTUM)
    
    learner.train_task(train_ds, 1, mode='defense')
    def_t1, def_t2, def_asr = evaluate_full(learner, test_ds)

    # --- FINAL REPORT ---
    results = pd.DataFrame(
        columns = ['METRIC', 'BASELINE (T1)', 'VULNERABLE', 'DEFENSE (AACL)'],
        data = [
            ['Task 1 Accuracy (0-4)', f"{t1_acc:.2f}%", f"{vuln_t1:.2f}%", f"{def_t1:.2f}%"],
            ['Task 2 Accuracy (5-9)', 'N/A', f"{vuln_t2:.2f}%", f"{def_t2:.2f}%"],
            ['BACKDOOR ASR', '0.00%', f"{vuln_asr:.2f}%", f"{def_asr:.2f}%"]
        ]
    )

    print("\n=== FINAL RESULTS COMPARISON ===")
    print(results.to_string(index=False))
    print("="*100)