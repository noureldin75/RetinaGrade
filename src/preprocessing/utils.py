import numpy as np
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
def load_data(df, image_dir):
    """
    Load images and labels from a DataFrame and an image directory.

    Parameters:
    df (pd.DataFrame): DataFrame containing image filenames and labels.
    image_dir (str): Directory where images are stored.

    Returns:
    tuple: A tuple containing two lists - images and labels.
    """
    x = []
    y = []

    for _, row in df.iterrows():
        image_path = os.path.join(image_dir, row["id_code"] + ".png")
        
        image = np.array(Image.open(image_path))
        
        x.append(image)
        y.append(row["diagnosis"])

    x = np.array(x)
    y = np.array(y)
    return x,y

def data_split(x, y, val_size=0.15, random_state=42):
    """
    Split the dataset into training and validation sets.

    Parameters:
    x (np.ndarray): Array of images.
    y (np.ndarray): Array of labels.
    val_size (float): Proportion of the dataset to include in the validation split.
    random_state (int): Random seed for reproducibility.

    Returns:
    tuple: A tuple containing training and validation sets - (x_train, x_val, y_train, y_val).
    """
    return train_test_split(x, y, test_size=val_size, random_state=random_state, shuffle=True)    