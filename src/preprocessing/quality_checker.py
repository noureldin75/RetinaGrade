
import cv2
import numpy as np
from pathlib import Path
from typing import Union, Dict, Tuple, List, Any


class ImageQualityAssessor:
    """
    Lightweight image quality assessment for fundus images.
    Uses simple, fast metrics that work well on CPU.
    """

    def __init__(
            self,
            laplacian_threshold: float = 100.0,
            brightness_low: float = 30.0,
            brightness_high: float = 220.0,
            contrast_threshold: float = 20.0
    ):
        self.laplacian_threshold = laplacian_threshold
        self.brightness_low = brightness_low
        self.brightness_high = brightness_high
        self.contrast_threshold = contrast_threshold

    def _ensure_rgb(self, img: np.ndarray) -> np.ndarray:
        if img is None:
            raise ValueError("Input image is None")
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3:
            if img.shape[2] == 1:
                return cv2.cvtColor(img.squeeze(axis=2), cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            elif img.shape[2] == 3:
                return img
        raise ValueError(f"Unsupported image shape: {img.shape}")

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2:
            return img
        elif len(img.shape) == 3:
            if img.shape[2] == 1:
                return img.squeeze(axis=2)
            else:
                return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    def compute_laplacian_variance(self, img: np.ndarray) -> float:
        gray = self._to_grayscale(img)
        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    def compute_brightness(self, img: np.ndarray) -> float:
        gray = self._to_grayscale(img)
        if gray.max() <= 1.0 and gray.dtype in [np.float32, np.float64]:
            gray = gray * 255
        return float(np.mean(gray))

    def compute_contrast(self, img: np.ndarray) -> float:
        gray = self._to_grayscale(img)
        if gray.max() <= 1.0 and gray.dtype in [np.float32, np.float64]:
            gray = gray * 255
        return float(np.std(gray))

    def compute_fundus_coverage(self, img: np.ndarray) -> float:
        gray = self._to_grayscale(img)
        if gray.max() <= 1.0 and gray.dtype in [np.float32, np.float64]:
            gray = (gray * 255).astype(np.uint8)
        elif gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)
        non_black = np.sum(gray > 10)
        return float(non_black / gray.size)

    def assess_quality(self, img: np.ndarray) -> Tuple[float, List[str]]:
        img_rgb = self._ensure_rgb(img)

        laplacian_var = self.compute_laplacian_variance(img_rgb)
        brightness = self.compute_brightness(img_rgb)
        contrast = self.compute_contrast(img_rgb)
        coverage = self.compute_fundus_coverage(img_rgb)

        issues = []
        scores = []

        if laplacian_var < self.laplacian_threshold:
            issues.append("blurry")
            scores.append(laplacian_var / self.laplacian_threshold)
        else:
            scores.append(min(1.0, laplacian_var / (self.laplacian_threshold * 3)))

        if brightness < self.brightness_low:
            issues.append("underexposed")
            scores.append(brightness / self.brightness_low)
        elif brightness > self.brightness_high:
            issues.append("overexposed")
            scores.append((255 - brightness) / (255 - self.brightness_high))
        else:
            optimal = 125
            scores.append(max(0, 1.0 - abs(brightness - optimal) / optimal))

        if contrast < self.contrast_threshold:
            issues.append("low_contrast")
            scores.append(contrast / self.contrast_threshold)
        else:
            scores.append(min(1.0, contrast / (self.contrast_threshold * 3)))

        if coverage < 0.3:
            issues.append("insufficient_fundus_area")
            scores.append(coverage / 0.3)
        else:
            scores.append(min(1.0, coverage))

        weights = [0.35, 0.20, 0.25, 0.20]
        quality_score = max(0.0, min(1.0, sum(s * w for s, w in zip(scores, weights))))

        return quality_score, issues

