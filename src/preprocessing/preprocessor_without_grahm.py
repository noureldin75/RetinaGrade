import cv2
import numpy as np
from pathlib import Path
from typing import Union, Dict


class RetinaPreprocessor:

    def __init__(
            self,
            img_size: int = 512,
            clahe_clip: float = 2.0,
            clahe_grid: tuple = (8, 8),
    ):
        self.img_size = img_size
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid

    def crop_circle_and_resize(self, image: np.ndarray) -> np.ndarray:

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cropped = image[y:y + h, x:x + w]
        else:
            cropped = image

        resized = cv2.resize(cropped, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_AREA)
        return resized

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip,
                                tileGridSize=self.clahe_grid)
        l_clahe = clahe.apply(l)

        lab_clahe = cv2.merge([l_clahe, a, b])
        enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        return enhanced

    def _run_pipeline(self, img_bgr: np.ndarray, apply_clahe: bool = True) -> np.ndarray:
        img = self.crop_circle_and_resize(img_bgr)
        if apply_clahe:
            img = self.apply_clahe(img)
        return img

    def preprocess(
            self,
            image_path: Union[str, Path],
            apply_clahe: bool = True,
            return_tensor: bool = False,
    ) -> np.ndarray:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        img = self._run_pipeline(img, apply_clahe=apply_clahe)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        if return_tensor:
            img = np.transpose(img, (2, 0, 1))

        return img

    def preprocess_array(
            self,
            img: np.ndarray,
            apply_clahe: bool = True,
            return_tensor: bool = False,
            is_rgb: bool = True,
    ) -> np.ndarray:
        if img is None:
            raise ValueError("Input image is None")

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif is_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img = self._run_pipeline(img, apply_clahe=apply_clahe)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        if return_tensor:
            img = np.transpose(img, (2, 0, 1))

        return img

    def preprocess_for_visualization(self, image_path: Union[str, Path]) -> Dict[str, np.ndarray]:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        stages = {}
        stages['original'] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cropped = self.crop_circle_and_resize(img)
        stages['cropped_resized'] = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        clahe_applied = self.apply_clahe(cropped)
        stages['clahe_applied'] = cv2.cvtColor(clahe_applied, cv2.COLOR_BGR2RGB)

        return stages