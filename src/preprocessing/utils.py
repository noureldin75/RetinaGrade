import numpy as np
import pandas as pd
import os
from PIL import Image

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
    