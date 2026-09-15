import matplotlib.pyplot as plt
import numpy as np

def plot_augmented_batch(dataloader, num_images=16):
    """
    Fetches a single batch from a PyTorch DataLoader and plots the images.
    """
    # Grab one batch of data (Images and Labels)
    images, labels = next(iter(dataloader))
    
    # Ensure we don't try to plot more images than exist in the batch
    num_images = min(num_images, images.size(0))
    
    # Calculate grid size (e.g., 4x4 for 16 images)
    rows = int(np.ceil(np.sqrt(num_images)))
    cols = int(np.ceil(num_images / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()
    
    for i in range(num_images):
        # PyTorch stores images as (Channels, Height, Width).
        # Matplotlib requires (Height, Width, Channels), so we transpose it.
        img = images[i].numpy().transpose(1, 2, 0)
        
        # If you used standard ImageNet normalization in Albumentations, 
        # the tensor values will be negative. We must un-normalize them to view them.
        if img.min() < 0:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1) # Keep values strictly between 0 and 1 for plotting
        elif img.max() > 1.0:
            # If no normalization was applied and values are 0-255
            img = img.astype(np.uint8)
            
        axes[i].imshow(img)
        axes[i].set_title(f"Target Grade: {labels[i].item()}", fontsize=14)
        axes[i].axis('off')
        
    # Hide any extra empty subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.show()