import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from typing import Dict, Tuple
from taba.utils import DEVICE

class DataHandler:
    """Handles Split-MNIST data loading and subset creation."""
    
    @staticmethod
    def get_split_mnist(root: str = './data') -> Dict[str, Subset]:
        """
        Splits MNIST into Task A (digits 0-4) and Task B (digits 5-9).
        """
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_set = datasets.MNIST(root, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root, train=False, download=True, transform=transform)
        
        def filter_indices(dataset, labels):
            return [i for i, (_, y) in enumerate(dataset) if y in labels]

        task_a_labels = set(range(5))
        task_b_labels = set(range(5, 10))

        splits = {
            'train_A': Subset(train_set, filter_indices(train_set, task_a_labels)),
            'train_B': Subset(train_set, filter_indices(train_set, task_b_labels)),
            'test_A': Subset(test_set, filter_indices(test_set, task_a_labels)),
            'test_B': Subset(test_set, filter_indices(test_set, task_b_labels)),
        }
        return splits

    @staticmethod
    def create_memory_buffer(subset: Subset, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Creates a fixed memory buffer from a dataset subset for Replay/TABA."""
        loader = DataLoader(subset, batch_size=size, shuffle=True)
        images, targets = next(iter(loader))
        return images.to(DEVICE), targets.to(DEVICE)