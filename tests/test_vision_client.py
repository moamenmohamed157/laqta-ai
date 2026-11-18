import base64
from types import SimpleNamespace

import pytest

from laqta_ai.config import Settings
from laqta_ai.vision_client import VisionAPIError, VisionClient


def make_valid_response():
    dimensions = {
        key: {"score": 7, "rationale": "The visible design element is handled clearly.", "evidence": "A visible object is positioned near the center."}
        for key in (
            "visual_hierarchy", "subject_visibility", "composition_balance", "color_harmony",
            "text_readability", "background_clutter", "visual_focus", "title_relevance",
        )
    }
    points = lambda prefix: [
        {"element": f"{prefix} {i}", "observation": "A specific visible design detail is present in this image.", "why_it_matters": "This affects how quickly the viewer reads the thumbnail."}
        for i in range(1, 4)
    ]
    import json
    return json.dumps({
        "overall_summary": "The thumbnail has a clear central subject and a readable color structure.",
        "visual_analysis": dimensions,
        "strengths": points("Strength"),
        "improvement_opportunities": points("Opportunity"),
        "prioritized_recommendations": [{
            "priority": "High",
            "issue": "Subject separation",
            "action": "Increase local separation around the central subject while keeping edges natural.",
            "expected_design_effect": "The subject should be easier to distinguish at a small display size.",
        }],
        "image_editing_prompt": "Improve subject separation around the central subject while preserving the current identity, pose, framing, and all important objects.",
    })


def configured_settings():
    return Settings(
        vision_api_key="test-secret",
        vision_base_url="https://vision.example/v1",
        vision_model="vision-test",
        vision_timeout_seconds=5,
        max_upload_mb=12,
        max_image_pixels=40_000_000,
        max_ai_image_side=1600,
    )


def test_client_sends_image_and_validates_json(monkeypatch):
    captured = {}

    def fake_post(url, *, headers, json, timeout):
        captured.update(url=url, headers=headers, payload=json, timeout=timeout)
        return SimpleNamespace(status_code=200, json=lambda: {"choices": [{"message": {"content": make_valid_response()}}]})

    monkeypatch.setattr("laqta_ai.vision_client.requests.post", fake_post)
    client = VisionClient(configured_settings())
    result = client.analyze(
        b"fake-jpeg-bytes",
        title="A day in Cairo",
        category="Vlog / Lifestyle",
        desired_style="Cinematic daylight",
        language="English",
    )
    assert result.overall_summary.startswith("The thumbnail")
    assert captured["url"] == "https://vision.example/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-secret"
    image_url = captured["payload"]["messages"][1]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(image_url.split(",", 1)[1]) == b"fake-jpeg-bytes"


def test_client_rejects_invalid_model_json(monkeypatch):
    monkeypatch.setattr(
        "laqta_ai.vision_client.requests.post",
        lambda *args, **kwargs: SimpleNamespace(status_code=200, json=lambda: {"choices": [{"message": {"content": "not json"}}]}),
    )
    with pytest.raises(VisionAPIError, match="not valid Laqta AI analysis JSON"):
        VisionClient(configured_settings()).analyze(
            b"img", title="", category="Other", desired_style="", language="English"
        )


def test_client_reports_http_error_without_fake_result(monkeypatch):
    monkeypatch.setattr(
        "laqta_ai.vision_client.requests.post",
        lambda *args, **kwargs: SimpleNamespace(status_code=401, json=lambda: {"error": {"message": "Unauthorized"}}),
    )
    with pytest.raises(VisionAPIError, match="HTTP 401"):
        VisionClient(configured_settings()).analyze(
            b"img", title="", category="Other", desired_style="", language="English"
        )


def test_client_requires_credentials_when_provider_is_configured(monkeypatch):
    settings = Settings(
        vision_api_key=None,
        vision_base_url="https://vision.example/v1",
        vision_model="vision-test",
        vision_timeout_seconds=5,
        max_upload_mb=12,
        max_image_pixels=40_000_000,
        max_ai_image_side=1600,
    )
    with pytest.raises(VisionAPIError, match="not configured"):
        VisionClient(settings).analyze(
            b"img", title="", category="Other", desired_style="", language="English"
        )
