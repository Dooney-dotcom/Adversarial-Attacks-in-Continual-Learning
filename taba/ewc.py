import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from taba.utils import DEVICE, mask_logits

class EWC:
    """Elastic Weight Consolidation (EWC) implementation."""
    
    def __init__(self, model: nn.Module, dataloader: DataLoader):
        self.model = model
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.fisher = {}
        self.opt_params = {}
        self._compute_fisher(dataloader)
    
    def _compute_fisher(self, dataloader: DataLoader):
        print("\n[EWC] Computing Fisher Information Matrix...")
        self.fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}
        self.model.eval()
        
        # Save optimal parameters for the previous task
        self.opt_params = {n: p.clone().detach() for n, p in self.params.items()}
        
        for data, target in dataloader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            self.model.zero_grad()
            output = self.model(data)
            # Mask logits for Task A (0-4) during Fisher calc
            output = mask_logits(output, range(0, 5)) 
            loss = F.cross_entropy(output, target)
            loss.backward()
            
            for n, p in self.params.items():
                if p.grad is not None:
                    self.fisher[n] += p.grad.data ** 2
        
        # Normalize
        for n in self.fisher:
            self.fisher[n] /= len(dataloader)
            
        print(f"[EWC] Fisher computed. Max value: {max([f.max().item() for f in self.fisher.values()]):.4f}")

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Calculates the EWC loss penalty."""
        loss = 0
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.opt_params[n]) ** 2).sum()
        return loss