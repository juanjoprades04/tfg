"""
Utilities for running orange segmentation inference with the final Mask2Former model.

Expected model directory:
models/orange_maskformer_final/
  - config.json
  - model.safetensors
  - preprocessor_config.json
  - training_args.bin  (optional for inference)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import math
import numpy as np
import torch
from PIL import Image
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation


DEFAULT_MODEL_DIR = "models/orange_maskformer_final"
ORANGE_CLASS_ID = 1


@lru_cache(maxsize=1)
def load_inference_model(model_dir: str = DEFAULT_MODEL_DIR):
    """Load the image processor and Mask2Former model once and cache them."""
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No existe la carpeta del modelo: {model_path}. "
            "Crea models/orange_maskformer_final/ y copia dentro config.json, "
            "model.safetensors y preprocessor_config.json."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load explicitly to avoid AutoImageProcessor issues with older Transformers versions
    # and configs saved as Mask2FormerImageProcessorFast.
    processor = Mask2FormerImageProcessor.from_pretrained(model_path)

    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path)
    model.to(device)
    model.eval()

    return processor, model, device


def _to_pil_rgb(image: Any) -> Image.Image:
    if image is None:
        raise ValueError("Debes subir una imagen para ejecutar inferencia.")

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, np.ndarray):
        return Image.fromarray(image.astype(np.uint8)).convert("RGB")

    raise TypeError(f"Tipo de imagen no soportado: {type(image)}")


def predict_mask(image: Any, model_dir: str = DEFAULT_MODEL_DIR) -> np.ndarray:
    """Predict semantic mask. Returns HxW mask with 0=background, 1=orange."""
    pil_image = _to_pil_rgb(image)
    processor, model, device = load_inference_model(model_dir)

    width, height = pil_image.size
    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    pred = processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=[(height, width)],
    )[0]

    mask = pred.detach().cpu().numpy().astype(np.uint8)
    mask = np.where(mask == ORANGE_CLASS_ID, ORANGE_CLASS_ID, 0).astype(np.uint8)
    return mask


def _sphere_diameter_from_component(component_mask: np.ndarray, bbox_w: int, bbox_h: int, area_px: int) -> float:
    """
    Estimate orange sphere diameter in pixels from the visible 2D component.

    Instead of using only the equivalent-area circle diameter, this uses the largest
    visible axis of the component. For approximately spherical fruits, the projected
    diameter is better represented by the largest apparent axis than by the area when
    the fruit is partially hidden by leaves/branches.
    """
    equivalent_diameter = _equivalent_diameter_px(float(area_px))
    bbox_diameter = float(max(bbox_w, bbox_h))

    ellipse_diameter = 0.0
    contours, _ = cv2.findContours(
        component_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if contours:
        contour = max(contours, key=cv2.contourArea)
        if len(contour) >= 5:
            _, axes, _ = cv2.fitEllipse(contour)
            ellipse_diameter = float(max(axes))

    # Conservative apparent sphere diameter: use the largest reliable visible axis.
    return max(equivalent_diameter, bbox_diameter, ellipse_diameter)


def count_oranges(mask: np.ndarray, min_area: int = 25) -> Tuple[int, list[Dict[str, Any]]]:
    """Count connected orange components and return component stats."""
    binary = (mask == ORANGE_CLASS_ID).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    components: list[Dict[str, Any]] = []
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[idx]

        component_mask = (labels[y:y + h, x:x + w] == idx).astype(np.uint8)
        sphere_diameter_px = _sphere_diameter_from_component(component_mask, w, h, area)
        visible_equivalent_diameter_px = _equivalent_diameter_px(float(area))

        components.append(
            {
                "area_px": area,
                "bbox": [x, y, w, h],
                "centroid": [float(cx), float(cy)],
                "visible_equivalent_diameter_px": float(visible_equivalent_diameter_px),
                "sphere_diameter_px_estimated": float(sphere_diameter_px),
            }
        )

    return len(components), components


def create_mask_image(mask: np.ndarray) -> Image.Image:
    """Create a visible black/white mask image."""
    return Image.fromarray((mask == ORANGE_CLASS_ID).astype(np.uint8) * 255)


def create_overlay(image: Any, mask: np.ndarray, alpha: float = 0.55) -> Image.Image:
    """Overlay predicted orange mask on the original image with visible contours."""
    pil_image = _to_pil_rgb(image)
    image_np = np.array(pil_image).astype(np.float32)

    overlay = image_np.copy()
    orange_color = np.array([255, 120, 0], dtype=np.float32)
    contour_color = np.array([255, 0, 0], dtype=np.float32)
    orange_pixels = mask == ORANGE_CLASS_ID

    overlay[orange_pixels] = (
        (1.0 - alpha) * overlay[orange_pixels]
        + alpha * orange_color
    )

    binary = orange_pixels.astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_uint8 = np.clip(overlay, 0, 255).astype(np.uint8)
    cv2.drawContours(overlay_uint8, contours, -1, contour_color.tolist(), thickness=3)

    return Image.fromarray(overlay_uint8)


def estimate_cm_per_pixel(
    image_width_px: int,
    distance_m: float = 2.0,
    horizontal_fov_deg: float = 70.0,
) -> float:
    """Approximate cm/px using camera distance and horizontal field of view."""
    if image_width_px <= 0 or distance_m <= 0 or horizontal_fov_deg <= 0:
        return 0.0
    scene_width_m = 2.0 * distance_m * math.tan(math.radians(horizontal_fov_deg / 2.0))
    return (scene_width_m * 100.0) / image_width_px


def _equivalent_diameter_px(area_px: float) -> float:
    """Diameter of a circle with the same area as the predicted component."""
    if area_px <= 0:
        return 0.0
    return 2.0 * math.sqrt(area_px / math.pi)


def extract_mask_stats(
    mask: np.ndarray,
    components: list[Dict[str, Any]],
    distance_m: float = 2.0,
    horizontal_fov_deg: float = 70.0,
) -> Dict[str, Any]:
    total_pixels = int(mask.size)
    orange_pixels = int((mask == ORANGE_CLASS_ID).sum())
    coverage_pct = (orange_pixels / total_pixels * 100.0) if total_pixels else 0.0

    image_width_px = int(mask.shape[1]) if mask.ndim == 2 else 0
    cm_per_px = estimate_cm_per_pixel(image_width_px, distance_m, horizontal_fov_deg)

    sphere_diameters_px = [float(c.get("sphere_diameter_px_estimated", 0.0)) for c in components]
    visible_diameters_px = [float(c.get("visible_equivalent_diameter_px", 0.0)) for c in components]
    sphere_diameters_cm = [d * cm_per_px for d in sphere_diameters_px]

    if sphere_diameters_px:
        mean_diameter_px = float(np.mean(sphere_diameters_px))
        min_diameter_px = float(np.min(sphere_diameters_px))
        max_diameter_px = float(np.max(sphere_diameters_px))
        mean_diameter_cm = float(np.mean(sphere_diameters_cm))
        min_diameter_cm = float(np.min(sphere_diameters_cm))
        max_diameter_cm = float(np.max(sphere_diameters_cm))
        mean_visible_diameter_px = float(np.mean(visible_diameters_px))
    else:
        mean_diameter_px = min_diameter_px = max_diameter_px = 0.0
        mean_diameter_cm = min_diameter_cm = max_diameter_cm = 0.0
        mean_visible_diameter_px = 0.0

    return {
        "orange_pixels": orange_pixels,
        "coverage_pct": coverage_pct,
        "cm_per_px_estimated": cm_per_px,
        "diameter_method": "estimated_sphere_projected_major_axis",
        "mean_visible_equivalent_diameter_px": mean_visible_diameter_px,
        "mean_diameter_px": mean_diameter_px,
        "min_diameter_px": min_diameter_px,
        "max_diameter_px": max_diameter_px,
        "mean_diameter_cm_estimated": mean_diameter_cm,
        "min_diameter_cm_estimated": min_diameter_cm,
        "max_diameter_cm_estimated": max_diameter_cm,
    }


def run_orange_inference(
    image: Any,
    model_dir: str = DEFAULT_MODEL_DIR,
    min_area: int = 25,
    distance_m: float = 2.0,
    horizontal_fov_deg: float = 70.0,
) -> Dict[str, Any]:
    """Full inference pipeline used by the Gradio app."""
    pil_image = _to_pil_rgb(image)
    mask = predict_mask(pil_image, model_dir=model_dir)
    count, components = count_oranges(mask, min_area=min_area)
    stats = extract_mask_stats(mask, components, distance_m, horizontal_fov_deg)

    return {
        "count": count,
        "mask": mask,
        "mask_image": create_mask_image(mask),
        "overlay": create_overlay(pil_image, mask),
        "components": components,
        "stats": stats,
    }
