import numpy as np
from PIL import Image

from laqta_ai.metrics import calculate_metrics


def test_uniform_gray_has_expected_brightness_and_low_contrast():
    image = Image.fromarray(np.full((100, 160, 3), 128, dtype=np.uint8), mode="RGB")
    result = calculate_metrics(image)
    assert result.brightness == 50.2
    assert result.contrast == 0.0
    assert result.saturation == 0.0
    assert result.dimensions == "160 × 100"
    assert result.aspect_ratio == 1.6
    assert len(result.dominant_colors) == 1
    assert result.dominant_colors[0].percentage == 100.0


def test_colorful_image_reports_saturation_and_dominant_color():
    pixels = np.zeros((100, 100, 3), dtype=np.uint8)
    pixels[:, :70] = (255, 0, 0)
    pixels[:, 70:] = (0, 0, 255)
    result = calculate_metrics(Image.fromarray(pixels, mode="RGB"))
    assert result.saturation > 90
    assert result.dominant_colors[0].percentage == 70.0
    assert result.dominant_colors[0].hex in {"#FF0000", "#0000FF"}


def test_black_and_white_clipping_are_measured():
    pixels = np.zeros((10, 10, 3), dtype=np.uint8)
    pixels[5:, :, :] = 255
    result = calculate_metrics(Image.fromarray(pixels, mode="RGB"))
    assert result.underexposed_percent == 50.0
    assert result.overexposed_percent == 50.0
    assert result.contrast > 40
    assert len(result.quality_notes) >= 1


def test_metric_values_are_finite():
    rng = np.random.default_rng(42)
    image = Image.fromarray(rng.integers(0, 256, (80, 120, 3), dtype=np.uint8), mode="RGB")
    result = calculate_metrics(image)
    for value in (result.brightness, result.contrast, result.saturation, result.sharpness_variance):
        assert np.isfinite(value)
        assert value >= 0
    assert sum(c.percentage for c in result.dominant_colors) == 100.0
