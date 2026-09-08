import numpy as np
import pandas as pd
import os
import cv2
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
        image = cv2.resize(np.array(Image.open(image_path)), (224, 224))  # Resize to 224x224
        # image = np.array(Image.open(image_path))
        
        x.append(image)
        y.append(row["diagnosis"])

    x = np.array(x)
    y = np.array(y)
    return x,y

def data_split(x, y, val_size=0.15,test_size=0.15, random_state=42):
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
    x_temp,x_test,y_temp,y_test=train_test_split(x,y,test_size=test_size,random_state=random_state,shuffle=True,stratify=y)

    relative_val_size = val_size / (1 - test_size)

    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp,test_size=relative_val_size,random_state=random_state,shuffle=True,stratify=y_temp
    )

    return x_train, x_val, x_test, y_train, y_val, y_test
