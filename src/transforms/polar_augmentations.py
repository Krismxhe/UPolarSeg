"""
Polar-domain augmentations.
"""

import numpy as np
import cv2
import albumentations as A


class CircularAngularShift(A.DualTransform):

    def __init__(self, max_shift_deg: float = 15.0, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.max_shift_deg = max_shift_deg

    def _max_shift_px(self, H: int) -> int:
        return max(1, int(self.max_shift_deg * H / 360.0))

    def apply(self, img, shift: int = 0, **params):
        return np.roll(img, shift, axis=0)

    def apply_to_mask(self, mask, shift: int = 0, **params):
        return np.roll(mask, shift, axis=0)

    def get_params_dependent_on_targets(self, params):
        H = params["image"].shape[0]
        m = self._max_shift_px(H)
        return {"shift": int(np.random.randint(-m, m + 1))}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ("max_shift_deg",)


class RadialScaling(A.DualTransform):
    """
    沿径向轴做缩放, 模拟眼球大小差异 / 图像分辨率差异
    """

    def __init__(self, scale_range=(0.9, 1.1), always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.scale_range = scale_range

    def apply(self, img, scale: float = 1.0, **params):
        h, w = img.shape[:2]
        new_w = int(w * scale)
        resized = cv2.resize(img, (new_w, h), interpolation=cv2.INTER_LINEAR)
        if new_w >= w:
            return resized[:, :w]
        pad = np.zeros((h, w - new_w, img.shape[2]), dtype=img.dtype)
        return np.concatenate([resized, pad], axis=1)

    def apply_to_mask(self, mask, scale: float = 1.0, **params):
        h, w = mask.shape[:2]
        new_w = int(w * scale)
        resized = cv2.resize(mask, (new_w, h), interpolation=cv2.INTER_NEAREST)
        if new_w >= w:
            return resized[:, :w]
        pad = np.zeros((h, w - new_w), dtype=mask.dtype)
        return np.concatenate([resized, pad], axis=1)

    def get_params(self):
        return {"scale": float(np.random.uniform(*self.scale_range))}

    def get_transform_init_args_names(self):
        return ("scale_range",)
    