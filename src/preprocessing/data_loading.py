import numpy as np
import torch.utils.data as data
from augumentation import over_sample, train_transform, val_transform
from augumented_set import AugmentedDataset

def create_dataloaders(x_train, y_train, x_val, y_val, x_test, y_test, targets={1: 2, 2: 2, 3: 7, 4: 5}, batch_size=32, num_workers=4):
    """Creates and returns train, validation, and test dataloaders."""
    
    #Generate Indices
    train_indices = over_sample(y_train, targets=targets)
    val_indices = np.arange(len(y_val))
    test_indices = np.arange(len(y_test))

    # Instantiate Datasets
    train_dataset = AugmentedDataset(x_train, y_train, train_indices, transform=train_transform)
    val_dataset = AugmentedDataset(x_val, y_val, val_indices, transform=val_transform)
    test_dataset = AugmentedDataset(x_test, y_test, test_indices, transform=val_transform)

    #Create DataLoaders
    train_loader = data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader