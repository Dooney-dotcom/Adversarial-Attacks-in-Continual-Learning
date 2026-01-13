import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple

from taba.config import ExperimentConfig
from taba.utils import DEVICE, mask_logits
from taba.ewc import EWC
from taba.attacks import Attacker

class EWCTrainer:
    """Handles standard EWC training, with OPTIONAL Clean-Label Poisoning."""
    
    @staticmethod
    def train_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, 
                    optimizer: optim.Optimizer, ewc: EWC, 
                    proxy_buffer: Tuple[torch.Tensor, torch.Tensor], 
                    config: ExperimentConfig,
                    attack: bool = False) -> float:
        
        model.train()
        total_loss = 0
        proxy_imgs, proxy_lbls = proxy_buffer

        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            # --- 1. ATTACK STEP ---
            if attack:
                slice_idx = min(32, len(proxy_imgs))
                curr_proxy = (proxy_imgs[:slice_idx], proxy_lbls[:slice_idx])
                
                data_poisoned = Attacker.generate_clean_label_poison(
                    model, data, target, curr_proxy, config
                )
                model.train() # Reset train mode
            else:
                # Clean scenario
                data_poisoned = data
            
            # --- 2. TRAINING STEP ---
            optimizer.zero_grad()
            
            # Forward
            output = model(data_poisoned)
            output = mask_logits(output, range(5, 10)) # Task B
            
            loss_main = F.cross_entropy(output, target)
            loss_ewc = config.ewc_lambda * ewc.penalty(model)
            
            loss = loss_main + loss_ewc
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()

        return total_loss / len(loader)

class TABATrainer:
    """Implements Task-Aware Boundary Augmentation (TABA)."""
    @staticmethod
    def taba_mix(x_o, y_o, x_n, y_n) -> Tuple[torch.Tensor, torch.Tensor]:
        lam = np.random.uniform(0.45, 0.55)
        min_len = min(len(x_o), len(x_n))
        x_o, y_o = x_o[:min_len], y_o[:min_len]
        x_n, y_n = x_n[:min_len], y_n[:min_len]
        x_aug = lam * x_o + (1 - lam) * x_n
        y_o_oh = F.one_hot(y_o, num_classes=10).float()
        y_n_oh = F.one_hot(y_n, num_classes=10).float()
        y_aug = lam * y_o_oh + (1 - lam) * y_n_oh
        return x_aug, y_aug

    @staticmethod
    def train_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, optimizer: optim.Optimizer, 
                    ewc: EWC, memory_buffer: Tuple[torch.Tensor, torch.Tensor], 
                    config: ExperimentConfig) -> float:
        model.train()
        total_loss = 0
        mem_imgs, mem_lbls = memory_buffer
        mem_size = len(mem_imgs)

        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            batch_size = len(data)

            # --- A. Boundary Mining (Old Task) ---
            idx = torch.randperm(mem_size)[:batch_size]
            B_o_img, B_o_lbl = mem_imgs[idx], mem_lbls[idx]
            adv_o = Attacker.pgd_attack(model, B_o_img, B_o_lbl)
            with torch.no_grad():
                mask_o = (model(adv_o).argmax(dim=1) != B_o_lbl)
            sel_o_img = B_o_img[mask_o] if mask_o.sum() > 0 else B_o_img
            sel_o_lbl = B_o_lbl[mask_o] if mask_o.sum() > 0 else B_o_lbl

            # --- B. Poisoning & Boundary Mining (New Task) ---
            proxy_slice = min(32, len(mem_imgs))
            proxy_batch = (mem_imgs[:proxy_slice], mem_lbls[:proxy_slice])
            data_poisoned = Attacker.generate_clean_label_poison(model, data, target, proxy_batch, config)
            model.train() 

            adv_n = Attacker.pgd_attack(model, data_poisoned, target)
            with torch.no_grad():
                mask_n = (model(adv_n).argmax(dim=1) != target)
            sel_n_img = data_poisoned[mask_n] if mask_n.sum() > 0 else data_poisoned
            sel_n_lbl = target[mask_n] if mask_n.sum() > 0 else target

            # --- C. TABA Mixing & Adv Training ---
            loss_taba = 0
            if len(sel_n_img) > 0 and len(sel_o_img) > 0:
                target_len = max(len(sel_n_img), len(sel_o_img))
                idx_o = torch.randint(0, len(sel_o_img), (target_len,))
                idx_n = torch.randint(0, len(sel_n_img), (target_len,))
                
                x_aug, y_aug_soft = TABATrainer.taba_mix(sel_o_img[idx_o], sel_o_lbl[idx_o], sel_n_img[idx_n], sel_n_lbl[idx_n])
                x_aug_adv = Attacker.pgd_attack(model, x_aug, y_aug_soft.argmax(dim=1))
                out_aug = model(x_aug_adv)
                log_probs = F.log_softmax(out_aug, dim=1)
                loss_taba = -(y_aug_soft * log_probs).sum(dim=1).mean()

            # --- D. Optimization Step ---
            optimizer.zero_grad()
            out_curr = model(data_poisoned)
            out_curr = mask_logits(out_curr, range(5, 10))
            loss_ce = F.cross_entropy(out_curr, target)
            loss_ewc = config.ewc_lambda * ewc.penalty(model)
            
            loss_total = loss_ce + loss_ewc + loss_taba
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss_total.item()

        return total_loss / len(loader)