"""Deterministic computer-vision metrics calculated from the uploaded image."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class DominantColor:
    hex: str
    rgb: tuple[int, int, int]
    percentage: float


@dataclass(frozen=True, slots=True)
class ImageMetrics:
    brightness: float
    contrast: float
    saturation: float
    sharpness_variance: float
    underexposed_percent: float
    overexposed_percent: float
    dimensions: str
    aspect_ratio: float
    aspect_ratio_label: str
    dominant_colors: tuple[DominantColor, ...]
    quality_notes: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _dominant_colors(rgb: np.ndarray, k: int = 5) -> tuple[DominantColor, ...]:
    # Downsample for speed and deterministic k-means initialization.
    sample = cv2.resize(rgb, (min(120, rgb.shape[1]), min(120, rgb.shape[0])), interpolation=cv2.INTER_AREA)
    pixels = np.float32(sample.reshape(-1, 3))
    unique_count = len(np.unique(pixels.astype(np.uint8), axis=0))
    clusters = max(1, min(k, unique_count, len(pixels)))
    if clusters == 1:
        centers = np.array([pixels.mean(axis=0)], dtype=np.float32)
        labels = np.zeros((len(pixels), 1), dtype=np.int32)
    else:
        cv2.setRNGSeed(17)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
        _, labels, centers = cv2.kmeans(
            pixels, clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
    counts = np.bincount(labels.flatten(), minlength=len(centers))
    order = np.argsort(counts)[::-1]
    colors: list[DominantColor] = []
    total = len(pixels)
    rounded_percentages: list[float] = []
    for position, index in enumerate(order):
        red, green, blue = np.clip(np.rint(centers[index]), 0, 255).astype(int)
        if position < len(order) - 1:
            percentage = round(float(counts[index] / total * 100), 1)
            rounded_percentages.append(percentage)
        else:
            # Keep the displayed distribution at exactly 100.0 after rounding.
            percentage = round(100.0 - sum(rounded_percentages), 1)
        colors.append(
            DominantColor(
                hex=f"#{red:02X}{green:02X}{blue:02X}",
                rgb=(int(red), int(green), int(blue)),
                percentage=percentage,
            )
        )
    return tuple(colors)


def calculate_metrics(image: Image.Image, aspect_ratio_label: str | None = None) -> ImageMetrics:
    """Calculate luminance, contrast, saturation, sharpness and color distribution.

    Brightness is mean Rec. 709 luma from gamma-encoded sRGB values on a 0–100 scale.
    Contrast is the standard deviation of that luma signal on a 0–100 scale.
    Saturation is mean HSV
    saturation on a 0–100 scale. Sharpness is Laplacian variance (unitless and
    resolution/content dependent), not a universal quality score.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_float = gray.astype(np.float32) / 255.0

    # Rec. 709 luma weighting applied to normalized, gamma-encoded sRGB channels.
    rgb_float = rgb.astype(np.float32) / 255.0
    luminance = (
        0.2126 * rgb_float[:, :, 0]
        + 0.7152 * rgb_float[:, :, 1]
        + 0.0722 * rgb_float[:, :, 2]
    )
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    underexposed = float(np.mean(luminance < 0.05) * 100)
    overexposed = float(np.mean(luminance > 0.95) * 100)

    ratio = width / height
    if aspect_ratio_label is None:
        aspect_ratio_label = "16:9 · YouTube standard" if abs(ratio - (16 / 9)) <= 0.025 else f"Custom · {ratio:.2f}:1"

    notes: list[str] = []
    if width < 1280 or height < 720:
        notes.append("Below 1280 × 720; consider exporting at YouTube's recommended 16:9 size.")
    if abs(ratio - (16 / 9)) > 0.025:
        notes.append("The image is not close to 16:9; check the intended placement and crop on YouTube.")
    if underexposed > 8:
        notes.append("A noticeable share of pixels is near-black; check whether shadow detail is intentional.")
    if overexposed > 8:
        notes.append("A noticeable share of pixels is near-white; check highlight detail.")
    if not notes:
        notes.append("No basic resolution, aspect-ratio, or extreme-clipping flag was triggered.")

    return ImageMetrics(
        brightness=round(float(luminance.mean() * 100), 1),
        contrast=round(float(luminance.std() * 100), 1),
        saturation=round(float(saturation.mean() * 100), 1),
        sharpness_variance=round(laplacian_variance, 2),
        underexposed_percent=round(underexposed, 2),
        overexposed_percent=round(overexposed, 2),
        dimensions=f"{width} × {height}",
        aspect_ratio=round(ratio, 3),
        aspect_ratio_label=aspect_ratio_label,
        dominant_colors=_dominant_colors(rgb),
        quality_notes=tuple(notes),
    )
