"""Orchestration for the local CV pipeline and optional remote vision review."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from .config import Settings
from .image_analyzer import ProcessedImage, load_image, prepare_image_for_ai
from .metrics import ImageMetrics, calculate_metrics
from .schemas import ThumbnailAnalysis
from .vision_client import VisionClient


@dataclass(frozen=True, slots=True)
class LocalAnalysis:
    processed_image: ProcessedImage
    metrics: ImageMetrics


def analyze_locally(raw_bytes: bytes, filename: str, settings: Settings) -> LocalAnalysis:
    processed = load_image(
        raw_bytes,
        filename,
        max_size_mb=settings.max_upload_mb,
        max_pixels=settings.max_image_pixels,
    )
    metrics = calculate_metrics(processed.image, processed.metadata.aspect_ratio_label)
    return LocalAnalysis(processed_image=processed, metrics=metrics)


def analyze_with_vision(
    image: Image.Image,
    settings: Settings,
    *,
    title: str,
    category: str,
    desired_style: str,
    language: str,
) -> ThumbnailAnalysis:
    image_bytes = prepare_image_for_ai(image, settings.max_ai_image_side)
    client = VisionClient(settings)
    return client.analyze(
        image_bytes,
        title=title,
        category=category,
        desired_style=desired_style,
        language=language,
    )
