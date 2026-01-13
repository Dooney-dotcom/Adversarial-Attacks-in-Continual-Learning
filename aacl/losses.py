import torch
import torch.nn.functional as F
from aacl.config import Config

def simclr_loss(proj_1, proj_2, temp=None):
    """
    Calculates Contrastive Loss (SimCLR).
    Maximizes agreement between different augmentations of the same image.
    """
    if temp is None:
        temp = Config.AACL_TEMP
        
    batch_size = proj_1.shape[0]
    features = torch.cat([proj_1, proj_2], dim=0)
    
    # Compute Similarity Matrix
    sim_matrix = torch.matmul(features, features.T) / temp
    
    # Mask out self-similarity
    mask_self = torch.eye(2 * batch_size, device=features.device).bool()
    sim_matrix.masked_fill_(mask_self, -9e15)
    
    # Compute Log-Sum-Exp for denominator
    denominators = torch.logsumexp(sim_matrix, dim=1)
    
    # Extract positives (diagonals of the off-diagonal blocks)
    sim_xy = torch.diag(sim_matrix, batch_size)
    sim_yx = torch.diag(sim_matrix, -batch_size)
    positives = torch.cat([sim_xy, sim_yx], dim=0)
    
    return (-positives + denominators).mean()

def distillation_loss_stable(new_logits, old_logits, num_old_classes, T=2.0):
    """
    Knowledge Distillation to prevent Catastrophic Forgetting.
    Only applied to classes the old model knew.
    """
    old_l = old_logits[:, :num_old_classes]
    new_l = new_logits[:, :num_old_classes]
    
    log_probs = F.log_softmax(new_l / T, dim=1)
    probs_teacher = F.softmax(old_l.detach() / T, dim=1)
    
    return F.kl_div(log_probs, probs_teacher, reduction='batchmean') * (T * T)