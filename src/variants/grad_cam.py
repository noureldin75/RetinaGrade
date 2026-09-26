"""
grad_cam.py - Grad-CAM for the RetinaGrade EfficientNetB3 models.

Works with both the baseline classifier (model) and the CORN ordinal
model (model_corn) built via model_builder.build_model(), since both
share the same torchvision EfficientNet `features` backbone.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM for a torchvision-style EfficientNet backbone.

    Usage:
        cam = GradCAM(model, target_layer=model.features[8][0])
        heatmap, pred_class = cam.generate(input_tensor)
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        Args:
            input_tensor: preprocessed image, shape (1, 3, H, W), already on the
                same device as the model.
            class_idx: class to explain. If None, uses the model's own
                top prediction (argmax of the raw logits — for the CORN
                model, pass the class index explicitly since raw logits
                there are per-threshold, not per-class).

        Returns:
            heatmap: (H, W) float array in [0, 1], resized to input_tensor's H, W
            class_idx: the class index that was explained
        """
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Global-average-pool the gradients -> per-channel importance weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam, class_idx


def overlay_heatmap(original_rgb_uint8: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original image.

    Args:
        original_rgb_uint8: (H, W, 3) uint8 RGB image, same H/W as the heatmap
        heatmap: (H, W) float array in [0, 1] from GradCAM.generate()
        alpha: heatmap opacity

    Returns:
        (H, W, 3) uint8 RGB overlay image
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    overlay = (colored * alpha + original_rgb_uint8 * (1 - alpha)).astype(np.uint8)
    return overlay