"""Environment-backed application settings."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load a local .env for developer workflows; real deployment secrets should be
# injected by the hosting environment or Streamlit secrets manager.
load_dotenv()


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings; secrets are read at runtime and never rendered in the UI."""

    vision_api_key: str | None
    vision_base_url: str
    vision_model: str
    vision_timeout_seconds: int
    max_upload_mb: int
    max_image_pixels: int
    max_ai_image_side: int

    @classmethod
    def from_env(cls, secrets: object | None = None) -> "Settings":
        """Read environment variables, optionally falling back to Streamlit secrets."""

        def read(name: str, default: str | None = None) -> str | None:
            value = os.getenv(name)
            if value is not None:
                return value
            if secrets is not None:
                try:
                    candidate = secrets.get(name)  # type: ignore[attr-defined]
                    if candidate is not None:
                        return str(candidate)
                except (AttributeError, KeyError, TypeError):
                    pass
            return default

        def read_int(name: str, default: int, minimum: int = 1) -> int:
            try:
                value = int(read(name, str(default)) or str(default))
            except (TypeError, ValueError):
                return default
            return value if value >= minimum else default

        api_key = read("LAQTA_VISION_API_KEY")
        return cls(
            vision_api_key=api_key.strip() if api_key and api_key.strip() else None,
            vision_base_url=(read("LAQTA_VISION_BASE_URL", "https://api.openai.com/v1") or "").rstrip("/"),
            vision_model=read("LAQTA_VISION_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini",
            vision_timeout_seconds=read_int("LAQTA_VISION_TIMEOUT_SECONDS", 90),
            max_upload_mb=read_int("LAQTA_MAX_UPLOAD_MB", 12),
            max_image_pixels=read_int("LAQTA_MAX_IMAGE_PIXELS", 40_000_000),
            max_ai_image_side=read_int("LAQTA_MAX_AI_IMAGE_SIDE", 1600, minimum=256),
        )

    @property
    def vision_is_configured(self) -> bool:
        return bool(self.vision_api_key and self.vision_base_url and self.vision_model)
