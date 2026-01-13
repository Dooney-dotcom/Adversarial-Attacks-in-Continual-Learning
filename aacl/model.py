import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN_AACL(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.classifier = nn.Linear(128, num_classes)
        
        # Projector used only during defense training (Contrastive Learning)
        # Maps features to a lower-dimensional latent space
        self.projector = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64)
        )

    def forward(self, x, return_features=False):
        feats = self.encoder(x)
        logits = self.classifier(feats)
        
        if return_features:
            # Normalize projection for cosine similarity in SimCLR
            proj = F.normalize(self.projector(feats), dim=1)
            return logits, proj
            
        return logits