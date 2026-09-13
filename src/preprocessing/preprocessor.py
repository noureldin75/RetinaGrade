import cv2
import numpy as np
from pathlib import Path
from typing import Union, Dict


class RetinaPreprocessor:
    """
    Preprocessing pipeline for fundus images using Ben Graham's method + CLAHE.
    يتضمن Dynamic Masking لمنع الـ ringing artifacts عند حواف الدائرة.
    """

    def __init__(
            self,
            img_size: int = 300,
            ben_graham_sigma: float = 10,
            clahe_clip: float = 2.0,
            clahe_grid: tuple = (8, 8),
    ):
        self.img_size = img_size
        self.ben_graham_sigma = ben_graham_sigma
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)

    def crop_image_from_gray(self, img: np.ndarray, tol: int = 7) -> np.ndarray:
        if img.ndim == 2:
            mask = img > tol
            return img[np.ix_(mask.any(1), mask.any(0))]
        elif img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.shape[2] == 3 else img[:, :, 0]
            mask = gray > tol

            if not mask.any():
                return img

            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            return img[rmin:rmax + 1, cmin:cmax + 1]
        return img

    def get_dynamic_mask(self, img: np.ndarray, tol: int = 15) -> np.ndarray:
        """
        بيحدد فين الشبكية الحقيقية وفين الخلفية (سواء سودة أصلية أو ناتجة من padding).
        هنستخدمه عشان نمنع أي معالجة من "تلويث" منطقة الخلفية.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, tol, 255, cv2.THRESH_BINARY)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_3d = np.repeat((mask > 0)[:, :, np.newaxis], 3, axis=2)
        return mask_3d

    def ben_graham_preprocessing(self, img: np.ndarray, fundus_mask: np.ndarray) -> np.ndarray:
        """
        نسخة مظبوطة من Ben Graham method.

        المشكلة في النسخة الأصلية: الـ Gaussian blur بيدمج لون الشبكية
        مع الخلفية السودة عند الحواف، والـ +128 بتحول الخلفية لرمادي.

        الحل: قبل الـ blur، بنملأ منطقة الخلفية بمتوسط لون الشبكية نفسها
        (مش أسود)، عشان الـ blur ميعملش خلط بين الشبكية ولون مختلف تماماً.
        وبعد المعالجة، بنرجّع الخلفية لأسود نضيف بالماسك.
        """
        if fundus_mask[:, :, 0].any():
            mean_color = img[fundus_mask[:, :, 0]].mean(axis=0).astype(np.uint8)
            img_for_blur = img.copy()
            img_for_blur[~fundus_mask[:, :, 0]] = mean_color
        else:
            img_for_blur = img

        sigma = self.ben_graham_sigma * (img.shape[0] / float(self.img_size))
        blurred = cv2.GaussianBlur(img_for_blur, (0, 0), sigma)
        result = cv2.addWeighted(img, 4, blurred, -4, 128)

        # نرجّع الخلفية لأسود نضيف فوراً، قبل ما تتبعت لأي خطوة تانية
        result = np.where(fundus_mask, result, 0).astype(np.uint8)
        return result

    def apply_clahe(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def resize_with_aspect_ratio(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        scale = self.img_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pad_h = (self.img_size - new_h) // 2
        pad_w = (self.img_size - new_w) // 2
        result = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        result[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        return result

    def _run_pipeline(self, img: np.ndarray, apply_ben_graham: bool, apply_clahe: bool) -> np.ndarray:
        img = self.crop_image_from_gray(img)
        img = self.resize_with_aspect_ratio(img)

        fundus_mask = self.get_dynamic_mask(img)

        if apply_ben_graham:
            img = self.ben_graham_preprocessing(img, fundus_mask=fundus_mask)

        if apply_clahe:
            img = self.apply_clahe(img)
            # الماسك بيتطبق تاني بعد CLAHE، لأن CLAHE ممكن يأثر على حواف
            # الماسك نفسه شوية حتى لو كانت الخلفية أسود نضيف قبلها
            img = np.where(fundus_mask, img, 0).astype(np.uint8)

        return img

    def preprocess(
            self,
            image_path: Union[str, Path],
            apply_ben_graham: bool = True,
            apply_clahe: bool = True,
            return_tensor: bool = False,
    ) -> np.ndarray:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self._run_pipeline(img, apply_ben_graham, apply_clahe)

        img = img.astype(np.float32) / 255.0

        if return_tensor:
            img = np.transpose(img, (2, 0, 1))

        return img

    def preprocess_array(
            self,
            img: np.ndarray,
            apply_ben_graham: bool = True,
            apply_clahe: bool = True,
            return_tensor: bool = False,
            is_rgb: bool = True,
    ) -> np.ndarray:
        if img is None:
            raise ValueError("Input image is None")

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif not is_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self._run_pipeline(img, apply_ben_graham, apply_clahe)

        img = img.astype(np.float32) / 255.0

        if return_tensor:
            img = np.transpose(img, (2, 0, 1))

        return img

    def preprocess_for_visualization(self, image_path: Union[str, Path]) -> Dict[str, np.ndarray]:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        stages = {}
        stages['original'] = img.copy()

        img_cropped = self.crop_image_from_gray(img)
        img_resized = self.resize_with_aspect_ratio(img_cropped)
        stages['resized'] = img_resized.copy()

        fundus_mask = self.get_dynamic_mask(img_resized)

        img_graham = self.ben_graham_preprocessing(img_resized, fundus_mask=fundus_mask)
        stages['ben_graham'] = img_graham.copy()

        img_clahe = self.apply_clahe(img_graham)
        img_clahe = np.where(fundus_mask, img_clahe, 0).astype(np.uint8)
        stages['ben_graham_clahe'] = img_clahe.copy()

        return stages