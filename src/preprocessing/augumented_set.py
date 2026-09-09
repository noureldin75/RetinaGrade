import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.augumentation import apply_transform

class AugmentedDataset(Dataset):
    def __init__(self, images, labels, final_indices, transform):
        self.images = images
        self.labels = labels
        self.final_indices = final_indices
        self.transform = transform

    def __len__(self):
        return len(self.final_indices)

    def __getitem__(self,i):
        actual_index=self.final_indices[i]
        image=self.images[actual_index]
        label=self.labels[actual_index]
        
        image_tensor=apply_transform(self.transform, image)
        label_tensor=torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor

    