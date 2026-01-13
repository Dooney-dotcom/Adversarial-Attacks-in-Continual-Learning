import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple
from taba.config import ExperimentConfig
from taba.utils import DEVICE

class Attacker:
    """Static methods for adversarial attacks (PGD and Poisoning)."""

    @staticmethod
    def generate_clean_label_poison(model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, 
                                    target_proxy_batch: Tuple[torch.Tensor, torch.Tensor], 
                                    config: ExperimentConfig) -> torch.Tensor:
        """
        Perturbs 'inputs' (Current Task) to look normal but have gradients that 
        interfere with the 'target_proxy_batch' (Previous Task).
        """
        model.eval()
        inputs_poison = inputs.clone().detach().to(DEVICE)
        inputs_poison.requires_grad = True
        
        data_A, target_A = target_proxy_batch
        
        poison_opt = optim.SGD([inputs_poison], lr=config.attack_lr)
        
        for _ in range(config.attack_steps):
            poison_opt.zero_grad()
            
            # Loss on current task
            out_B = model(inputs_poison)
            loss_B = F.cross_entropy(out_B, labels)
            
            # Loss on target task
            out_A = model(data_A)
            loss_A = F.cross_entropy(out_A, target_A)
            
            # Objective: Maximize Loss_A
            adv_loss = loss_B - 2.0 * loss_A 
            
            adv_loss.backward()
            poison_opt.step()
            
            # Projection
            delta = inputs_poison.data - inputs.data
            delta = torch.clamp(delta, -config.attack_eps, config.attack_eps)
            inputs_poison.data = torch.clamp(inputs.data + delta, 0, 1)
            
        inputs_poison.requires_grad = False
        return inputs_poison

    @staticmethod
    def pgd_attack(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, 
                   eps: float = 0.3, steps: int = 5, alpha: float = 0.05) -> torch.Tensor:
        """Standard PGD attack for robustness testing and TABA generation."""
        images = images.clone().detach().to(DEVICE)
        labels = labels.clone().detach().to(DEVICE)
        
        adv_images = images + torch.empty_like(images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, 0, 1).detach()
        
        for _ in range(steps):
            adv_images.requires_grad = True
            outputs = model(adv_images)
            loss = F.cross_entropy(outputs, labels)
            
            grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
            
            adv_images = adv_images.detach() + alpha * grad.sign()
            delta = torch.clamp(adv_images - images, -eps, eps)
            adv_images = torch.clamp(images + delta, 0, 1).detach()
            
        return adv_images