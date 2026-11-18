from io import BytesIO

import pytest
from PIL import Image

from laqta_ai.image_analyzer import ImageValidationError, load_image, prepare_image_for_ai


def make_image(fmt: str = "PNG", size=(1280, 720), color=(20, 120, 180)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format=fmt)
    return buffer.getvalue()


def test_load_png_extracts_dimensions_and_ratio():
    result = load_image(make_image(), "thumbnail.png")
    assert result.metadata.width == 1280
    assert result.metadata.height == 720
    assert result.metadata.aspect_ratio == pytest.approx(16 / 9)
    assert "16:9" in result.metadata.aspect_ratio_label
    assert result.image.mode == "RGB"


def test_rejects_unsupported_extension():
    with pytest.raises(ImageValidationError, match="Unsupported file extension"):
        load_image(make_image(), "thumbnail.gif")


def test_rejects_extension_content_mismatch():
    with pytest.raises(ImageValidationError, match="extension says .jpg"):
        load_image(make_image("PNG"), "thumbnail.jpg")


def test_rejects_oversized_file():
    data = make_image()
    with pytest.raises(ImageValidationError, match="too large"):
        load_image(data, "thumbnail.png", max_size_mb=0)


def test_rejects_excessive_pixel_count():
    with pytest.raises(ImageValidationError, match="dimensions are too large"):
        load_image(make_image(size=(100, 100)), "thumbnail.png", max_pixels=999)


def test_prepare_ai_image_preserves_aspect_ratio():
    source = Image.new("RGB", (3000, 1000), (80, 100, 120))
    prepared = Image.open(BytesIO(prepare_image_for_ai(source, max_side=1200)))
    assert prepared.size == (1200, 400)
    assert prepared.format == "JPEG"


def test_prepare_ai_image_does_not_upscale_small_image():
    source = Image.new("RGB", (640, 360), (80, 100, 120))
    prepared = Image.open(BytesIO(prepare_image_for_ai(source, max_side=1600)))
    assert prepared.size == (640, 360)
