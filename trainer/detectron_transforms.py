import random
import cv2
import numpy as np
from PIL import Image

from detectron2.data.transforms import Augmentation, Transform, NoOpTransform


class RandomScaleTransform(Transform):
    """Scale the image and masks by a factor."""
    def __init__(self, scale_factor: float):
        super().__init__()
        self.scale_factor = scale_factor

    def apply_image(self, img):
        h, w = img.shape[:2]
        new_h = max(1, int(h * self.scale_factor))
        new_w = max(1, int(w * self.scale_factor))
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
        return np.array(pil_img)

    def apply_coords(self, coords):
        return coords * self.scale_factor

class RandomScale(Augmentation):
    """Randomly scale an image by ±scale_limit with probability prob."""
    def __init__(self, scale_limit=0.2, prob=0.5):
        super().__init__()
        self.scale_limit = scale_limit
        self.prob = prob

    def get_transform(self, image):
        if random.random() > self.prob:
            return NoOpTransform()
        factor = 1.0 + random.uniform(-self.scale_limit, self.scale_limit)
        return RandomScaleTransform(factor)

# ============================================================
# 1. Hue / Saturation / Value
# (Análogo a A.HueSaturationValue)
# ============================================================

class HSVTransform(Transform):
    def __init__(self, hue_shift, sat_shift, val_shift):
        super().__init__()
        self.hue_shift = hue_shift
        self.sat_shift = sat_shift
        self.val_shift = val_shift

    def apply_image(self, img):
        if img.ndim != 3 or img.shape[2] != 3:
            return img
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

        hsv[..., 0] = (hsv[..., 0] + self.hue_shift) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] + self.sat_shift, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] + self.val_shift, 0, 255)

        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def apply_coords(self, coords):
        return coords


class RandomHSV(Augmentation):
    def __init__(
        self,
        hue_shift_limit=0,
        sat_shift_limit=20,
        val_shift_limit=0,
        prob=0.5,
    ):
        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.val_shift_limit = val_shift_limit
        self.prob = prob

    def get_transform(self, image):
        if random.random() > self.prob:
            return NoOpTransform()

        h = random.uniform(-self.hue_shift_limit, self.hue_shift_limit)
        s = random.uniform(-self.sat_shift_limit, self.sat_shift_limit)
        v = random.uniform(-self.val_shift_limit, self.val_shift_limit)

        return HSVTransform(h, s, v)


# ============================================================
# 2. Gaussian Blur
# (Análogo a A.GaussianBlur)
# ============================================================

class GaussianBlurTransform(Transform):
    def __init__(self, ksize):
        super().__init__()
        self.ksize = ksize

    def apply_image(self, img):
        if img.ndim != 3 or img.shape[2] != 3:
            return img
        return cv2.GaussianBlur(img, (self.ksize, self.ksize), 0)

    def apply_coords(self, coords):
        return coords


class RandomGaussianBlur(Augmentation):
    def __init__(self, blur_limit=(3, 7), prob=0.5):
        self.blur_limit = blur_limit
        self.prob = prob

    def get_transform(self, image):
        if random.random() > self.prob:
            return NoOpTransform()

        k = random.randrange(self.blur_limit[0], self.blur_limit[1] + 1, 2)
        return GaussianBlurTransform(k)


# ============================================================
# 3. Motion Blur
# (Análogo a A.MotionBlur)
# ============================================================

class MotionBlurTransform(Transform):
    def __init__(self, ksize):
        super().__init__()
        self.ksize = ksize

    def apply_image(self, img):
        if img.ndim != 3 or img.shape[2] != 3:
            return img
        kernel = np.zeros((self.ksize, self.ksize))
        kernel[self.ksize // 2, :] = np.ones(self.ksize)
        kernel /= self.ksize
        return cv2.filter2D(img, -1, kernel)

    def apply_coords(self, coords):
        return coords


class RandomMotionBlur(Augmentation):
    def __init__(self, blur_limit=7, prob=0.5):
        self.blur_limit = blur_limit
        self.prob = prob

    def get_transform(self, image):
        if random.random() > self.prob:
            return NoOpTransform()

        k = random.randrange(3, self.blur_limit + 1, 2)
        return MotionBlurTransform(k)


# ============================================================
# 4. Gaussian Noise
# (Análogo a A.GaussNoise)
# ============================================================

class GaussianNoiseTransform(Transform):
    def __init__(self, std):
        super().__init__()
        self.std = std

    def apply_image(self, img):
        if img.ndim != 3 or img.shape[2] != 3:
            return img
        noise = np.random.normal(0, self.std * 255, img.shape)
        noisy = img.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def apply_coords(self, coords):
        return coords


class RandomGaussianNoise(Augmentation):
    def __init__(self, std_range=(0.05, 0.2), prob=0.5):
        self.std_range = std_range
        self.prob = prob

    def get_transform(self, image):
        if random.random() > self.prob:
            return NoOpTransform()

        std = random.uniform(*self.std_range)
        return GaussianNoiseTransform(std)


# ============================================================
# 5. ISO Noise
# (Análogo a A.ISONoise)
# ============================================================
class ISONoiseTransform(Transform):
    """
    Apply ISON-like noise to an RGB image.
    - Color shift in HSV hue channel.
    - Luminance noise added independently to all channels.
    """
    def __init__(self, color_shift: float, intensity: float):
        """
        Args:
            color_shift (float): Max fraction to shift hue channel (-color_shift, +color_shift)
            intensity (float): Stddev of luminance noise (0-1 scale)
        """
        super().__init__()
        self.color_shift = color_shift
        self.intensity = intensity

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        if img.ndim != 3 or img.shape[2] != 3:
            return img

        # Convert RGB -> HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Shift hue randomly
        hue_shift = np.random.uniform(-self.color_shift, self.color_shift) * 180  # HSV hue 0-179 in OpenCV
        hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180

        # Convert back to RGB
        img_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

        # Add luminance noise independently to each channel
        noise = img_rgb * np.random.normal(0, self.intensity, img_rgb.shape).astype(np.float32)
        img_rgb = np.clip(img_rgb + noise, 0.0, 1.0)

        # Return in 0-255 range
        return (img_rgb * 255).astype(np.uint8)

    def apply_coords(self, coords):
        # This transform does not change coordinates
        return coords


class RandomISONoise(Augmentation):
    """
    Randomly apply ISONoiseTransform with given probability.
    """
    def __init__(self, color_shift=(0.01, 0.05), intensity=(0.1, 0.5), prob=0.5):
        super().__init__()
        self.color_shift = color_shift
        self.intensity = intensity
        self.prob = prob

    def get_transform(self, image):
        if np.random.rand() > self.prob:
            # No transform
            return  NoOpTransform()
        
        # Randomly pick values in the provided ranges
        color_shift_val = np.random.uniform(self.color_shift[0], self.color_shift[1])
        intensity_val = np.random.uniform(self.intensity[0], self.intensity[1])

        return ISONoiseTransform(color_shift=color_shift_val, intensity=intensity_val)