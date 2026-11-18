"""Safe image loading, normalization, metadata extraction, and AI preparation."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePath

from PIL import Image, ImageOps, UnidentifiedImageError


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


class ImageValidationError(ValueError):
    """Raised when an uploaded image is invalid or outside configured limits."""


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    width: int
    height: int
    aspect_ratio: float
    aspect_ratio_label: str
    format: str
    file_size_bytes: int
    pixel_count: int


@dataclass(frozen=True, slots=True)
class ProcessedImage:
    image: Image.Image
    metadata: ImageMetadata


def _ratio_label(width: int, height: int) -> str:
    ratio = width / height
    if abs(ratio - (16 / 9)) <= 0.025:
        return "16:9 · YouTube standard"
    if abs(ratio - 1.0) <= 0.025:
        return "1:1 · Square"
    if abs(ratio - (4 / 5)) <= 0.025:
        return "4:5 · Portrait"
    if abs(ratio - (9 / 16)) <= 0.025:
        return "9:16 · Vertical"
    return f"Custom · {ratio:.2f}:1"


def load_image(
    raw_bytes: bytes,
    filename: str,
    *,
    max_size_mb: int = 12,
    max_pixels: int = 40_000_000,
) -> ProcessedImage:
    """Validate and decode image bytes, applying EXIF orientation without distortion."""
    if not raw_bytes:
        raise ImageValidationError("The uploaded file is empty.")

    extension = PurePath(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ImageValidationError("Unsupported file extension. Upload JPG, JPEG, PNG, or WEBP.")

    max_bytes = max_size_mb * 1024 * 1024
    if len(raw_bytes) > max_bytes:
        raise ImageValidationError(f"File is too large. Maximum upload size is {max_size_mb} MB.")

    try:
        with Image.open(BytesIO(raw_bytes)) as source:
            image_format = (source.format or "").upper()
            if image_format not in ALLOWED_FORMATS:
                raise ImageValidationError("The file content is not a supported image format.")
            if getattr(source, "is_animated", False):
                source.seek(0)
            width, height = source.size
            if width <= 0 or height <= 0:
                raise ImageValidationError("The image has invalid dimensions.")
            if width * height > max_pixels:
                raise ImageValidationError(
                    f"Image dimensions are too large ({width:,} × {height:,}). "
                    f"The configured limit is {max_pixels:,} pixels."
                )
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("Could not read this image. Try exporting it as JPG, PNG, or WEBP.") from exc

    # Verify extension/content consistency rather than trusting the browser filename.
    expected = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP"}[extension]
    if image_format != expected:
        raise ImageValidationError(
            f"The file extension says {extension}, but the image content is {image_format or 'unknown'}."
        )

    final_width, final_height = image.size
    metadata = ImageMetadata(
        width=final_width,
        height=final_height,
        aspect_ratio=final_width / final_height,
        aspect_ratio_label=_ratio_label(final_width, final_height),
        format=image_format,
        file_size_bytes=len(raw_bytes),
        pixel_count=final_width * final_height,
    )
    return ProcessedImage(image=image, metadata=metadata)


def prepare_image_for_ai(image: Image.Image, max_side: int = 1600) -> bytes:
    """Encode an analysis copy as JPEG, fitting within max_side without stretching."""
    if max_side < 256:
        raise ValueError("max_side must be at least 256 pixels.")
    prepared = image.copy().convert("RGB")
    prepared.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    output = BytesIO()
    prepared.save(output, format="JPEG", quality=90, optimize=True)
    return output.getvalue()
