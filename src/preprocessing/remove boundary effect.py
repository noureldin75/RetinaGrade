"""
1_remove_boundary_effect.py

Preprocessing script for retina images (Diabetic Retinopathy pipeline).

This script:
    1. Detects the radius of the retina circle in each image.
    2. Rescales the image so the retina radius matches a fixed `scale`.
    3. Applies local color/illumination normalization (Gaussian blur subtraction).
    4. Removes the outer boundary (outer ~10%) that carries no useful signal.
    5. Saves the processed image to a new directory.

Usage:
    python 1_remove_boundary_effect.py
    (edit the paths and scale in the __main__ block below before running)
"""

import cv2
import numpy as np
import os
import time


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Cant delete {file_path}: {e}")


def scaleRadius(img, scale):
    """
    Estimates the radius of the retina circle in the image and resizes
    the image so that radius matches `scale`.

    INPUT
        img: Image array (as read by cv2.imread).
        scale: Target radius (in pixels) for the retina circle.
    OUTPUT
        Resized image (numpy array).
    """
    x = img[img.shape[0] // 2, :, :].sum(1)
    r = (x > x.mean() / 10).sum() // 2

    if r == 0:
        # Fallback to avoid division by zero on corrupt/blank images
        r = 1

    s = (scale * 1.0) / r
    return cv2.resize(img, (0, 0), fx=s, fy=s)


def remove_boundary_effect(path, new_path, scale, test_limit=None):
    """
    Crops, resizes, and normalizes all images from a directory and stores
    the results in a new directory.

    INPUT
        path: Path where the current, unscaled images are contained.
        new_path: Path to save the processed images.
        scale: Target radius (in pixels) for the retina circle.
        test_limit: Optional int. If set, only processes the first N images
                    found in `path` (useful for quick testing before running
                    on the full dataset).
    OUTPUT
        All images processed and saved from `path` to `new_path`.
    """
    start_time = time.time()

    create_directory(new_path)

    dire = [f for f in os.listdir(path) if f != '.DS_Store']  # skip macOS system file

    if test_limit is not None:
        dire = dire[:test_limit]
        print(f"[TEST MODE] Processing {len(dire)} image(s): {dire}")
    else:
        print(f"Processing {len(dire)} image(s) from: {path}")

    for item in dire:
        try:
            img = cv2.imread(os.path.join(path, item))
            if img is None:
                print(f"  [SKIP] Could not read: {item}")
                continue

            # 1. Scale so retina radius matches `scale`
            img = scaleRadius(img, scale)

            # 2. Subtract local average color (illumination normalization)
            img = cv2.addWeighted(
                img, 4,
                cv2.GaussianBlur(img, (0, 0), scale / 30),
                -4, 128
            )

            # 3. Remove outer 10% boundary (mask with a circle)
            mask = np.zeros(img.shape)
            cv2.circle(
                mask,
                (img.shape[1] // 2, img.shape[0] // 2),
                int(scale * 0.9),
                (1, 1, 1),
                -1, 8, 0
            )
            img = img * mask + 128 * (1 - mask)

            # 4. Save result
            cv2.imwrite(os.path.join(new_path, item), img)
            print(f"  [OK] {item}")

        except Exception as e:
            print(f"  [ERROR] {item}: {e}")

    elapsed = time.time() - start_time
    print(f"--- {elapsed:.2f} seconds ---")


if __name__ == "__main__":
    remove_boundary_effect(
        path='data/train_images/',
        new_path='data/processed_train_images/',
        scale=500,
        test_limit=5  # set to None to process the full dataset
    )