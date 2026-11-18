from laqta_ai.config import Settings


def test_settings_use_defaults_without_secrets(monkeypatch):
    for name in (
        "LAQTA_VISION_API_KEY", "LAQTA_VISION_BASE_URL", "LAQTA_VISION_MODEL",
        "LAQTA_VISION_TIMEOUT_SECONDS", "LAQTA_MAX_UPLOAD_MB", "LAQTA_MAX_IMAGE_PIXELS",
        "LAQTA_MAX_AI_IMAGE_SIDE",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = Settings.from_env()
    assert settings.vision_api_key is None
    assert settings.vision_base_url == "https://api.openai.com/v1"
    assert settings.max_upload_mb == 12
    assert settings.max_ai_image_side == 1600


def test_settings_ignore_invalid_integer_values(monkeypatch):
    monkeypatch.setenv("LAQTA_MAX_UPLOAD_MB", "not-a-number")
    monkeypatch.setenv("LAQTA_MAX_AI_IMAGE_SIDE", "100")
    settings = Settings.from_env()
    assert settings.max_upload_mb == 12
    assert settings.max_ai_image_side == 1600


def test_settings_strip_blank_or_whitespace_api_key(monkeypatch):
    monkeypatch.setenv("LAQTA_VISION_API_KEY", "   ")
    assert Settings.from_env().vision_api_key is None
