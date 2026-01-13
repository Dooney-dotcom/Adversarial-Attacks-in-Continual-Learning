import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from aacl.config import Config
from aacl.data import get_base_transform, get_defense_transform, add_backdoor_fixed
from aacl.losses import simclr_loss, distillation_loss_stable

class UnifiedLearner:
    def __init__(self, model):
        self.cfg = Config()
        self.model = model
        self.old_model = None   
        self.seen_classes = []
        self.memory = {} # Replay Buffer
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.cfg.LR, momentum=self.cfg.MOMENTUM)
        self.base_trans = get_base_transform()
        self.defense_aug = get_defense_transform()

    def _update_memory(self, dataset, new_classes):
        """Updates the Replay Buffer (Memory) with examples from new classes."""
        self.model.eval()
        total = len(self.seen_classes) + len(new_classes)
        if total == 0: return
        
        # Calculate examples per class
        m = self.cfg.MEMORY_SIZE // total
        
        # Shrink existing memory
        for cls in self.memory: 
            self.memory[cls] = self.memory[cls][:m]
        
        # Add new data
        for c in new_classes:
            indices = [i for i, label in enumerate(dataset.targets) if label == c]
            if len(indices) > 0:
                sel = np.random.choice(indices, min(m, len(indices)), replace=False)
                self.memory[c] = [dataset[idx][0] for idx in sel]

    def _get_loader(self, dataset, new_classes):
        """Creates a DataLoader combining new data and Replay Memory."""
        # 1. Get new task data
        indices = [i for i, t in enumerate(dataset.targets) if t in new_classes]
        data = [(dataset[i][0], dataset[i][1]) for i in indices]
        
        # 2. Add memory data
        for cls in self.memory:
            for img in self.memory[cls]: 
                data.append((img, cls))
        
        def collate(batch):
            imgs = [b[0] for b in batch]
            lbls = [b[1] for b in batch]
            return imgs, torch.tensor(lbls)
            
        return DataLoader(data, batch_size=self.cfg.BATCH_SIZE, shuffle=True, collate_fn=collate)

    def train_task(self, train_ds_raw, task_id, mode='standard'):
        """
        Main training loop.
        Args:
          mode: 
            - 'standard': Clean training (Task 1)
            - 'vulnerable': Training with Attack, NO Defense (Naive Replay)
            - 'defense': Training with Attack + AACL (Augmentation + Contrastive + Distillation)
        """
        start = task_id * self.cfg.CLASSES_PER_TASK
        end = (task_id + 1) * self.cfg.CLASSES_PER_TASK
        new_classes = list(range(start, end))
        
        print(f"\n>>> TASK {task_id + 1} | Mode: {mode.upper()} | Classes: {new_classes}")
        
        num_old_classes = len(self.seen_classes)
        
        # Freeze old model for distillation (Only in Defense mode)
        if mode == 'defense' and task_id > 0:
            self.old_model = copy.deepcopy(self.model)
            self.old_model.eval()
            for p in self.old_model.parameters(): p.requires_grad = False
        else:
            self.old_model = None

        loader = self._get_loader(train_ds_raw, new_classes)
        self.model.train()

        for epoch in range(self.cfg.EPOCHS):
            loop = tqdm(loader, desc=f"Ep {epoch+1}/{self.cfg.EPOCHS}", leave=False)
            
            for img_pil_list, labels in loop:
                labels = labels.to(self.cfg.DEVICE)
                imgs = torch.stack([self.base_trans(img) for img in img_pil_list]).to(self.cfg.DEVICE)
                
                # --- A. POISON INJECTION ---
                # Only inject poison if we are in Task 2 and not in clean mode
                mask_bd = torch.zeros_like(labels, dtype=torch.bool)
                if task_id == 1 and mode in ['vulnerable', 'defense']:
                    mask_bd = (torch.rand(len(labels)) < self.cfg.BACKDOOR_RATE).to(self.cfg.DEVICE)
                    if mask_bd.sum() > 0:
                        imgs[mask_bd] = add_backdoor_fixed(imgs[mask_bd])
                        labels[mask_bd] = self.cfg.BACKDOOR_TARGET
                
                # --- B. FORWARD & LOSS ---
                logits, proj = self.model(imgs, return_features=True)
                loss = 0
                
                if mode == 'vulnerable':
                    # Scenario A: Standard CE. Model blindly learns the trigger.
                    loss = F.cross_entropy(logits, labels)
                
                elif mode == 'defense':
                    # Scenario B: AACL Defense
                    
                    # 1. Augmentation Branch (for Contrastive Learning)
                    imgs_aug = torch.stack([self.defense_aug(imgs[i]) for i in range(len(imgs))]).to(self.cfg.DEVICE)
                    _, proj_aug = self.model(imgs_aug, return_features=True)
                    
                    # 2. CE Sniper: Ignore poisoned labels in classification loss
                    ce_raw = F.cross_entropy(logits, labels, reduction='none')
                    if mask_bd.sum() > 0: 
                        ce_raw[mask_bd] = 0.0 # Zero out loss for poisoned samples
                    loss_ce = ce_raw.mean()
                    
                    # 3. Contrastive Loss (Aligns clean features, breaks trigger association)
                    loss_cl = simclr_loss(proj, proj_aug)
                    
                    # 4. Distillation (Preserves old knowledge)
                    loss_dist = torch.tensor(0.0).to(self.cfg.DEVICE)
                    if self.old_model:
                        with torch.no_grad(): 
                            old_logits = self.old_model(imgs)
                        loss_dist = distillation_loss_stable(logits, old_logits, num_old_classes)
                    
                    loss = loss_ce + (self.cfg.AACL_LAMBDA * loss_cl) + (self.cfg.DIST_LAMBDA * loss_dist)
                
                else: # 'standard' (Task 1)
                    loss = F.cross_entropy(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
        self._update_memory(train_ds_raw, new_classes)
        self.seen_classes.extend(new_classes)
        self.seen_classes = sorted(list(set(self.seen_classes)))