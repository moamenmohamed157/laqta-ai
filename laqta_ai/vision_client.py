"""OpenAI-compatible multimodal Vision API client with strict JSON validation."""

from __future__ import annotations

import base64
import json
import re
from typing import Any

import requests
from pydantic import ValidationError

from .config import Settings
from .prompts import build_analysis_messages
from .schemas import ThumbnailAnalysis


class VisionAPIError(RuntimeError):
    """A safe, user-displayable error from the optional vision provider."""


class VisionClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def analyze(
        self,
        image_bytes: bytes,
        *,
        title: str,
        category: str,
        desired_style: str,
        language: str,
    ) -> ThumbnailAnalysis:
        if not self.settings.vision_is_configured:
            raise VisionAPIError(
                "Vision AI is not configured. Add LAQTA_VISION_API_KEY and check the provider settings."
            )
        system_prompt, user_prompt = build_analysis_messages(
            title=title,
            category=category,
            desired_style=desired_style,
            language=language,
        )
        encoded = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "model": self.settings.vision_model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded}", "detail": "high"},
                        },
                    ],
                },
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.settings.vision_api_key:
            headers["Authorization"] = f"Bearer {self.settings.vision_api_key}"
        url = f"{self.settings.vision_base_url}/chat/completions"

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.settings.vision_timeout_seconds,
            )
            if response.status_code >= 400:
                detail = self._provider_error(response)
                raise VisionAPIError(f"Vision provider returned HTTP {response.status_code}: {detail}")
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except VisionAPIError:
            raise
        except requests.Timeout as exc:
            raise VisionAPIError("Vision analysis timed out. Try again or increase LAQTA_VISION_TIMEOUT_SECONDS.") from exc
        except requests.RequestException as exc:
            raise VisionAPIError("Could not connect to the configured Vision API. Check the endpoint and network.") from exc
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise VisionAPIError("The Vision API returned an unexpected response structure.") from exc

        if isinstance(content, list):
            # Some compatible providers return content blocks instead of one text string.
            content = "".join(
                item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"
            )
        if not isinstance(content, str) or not content.strip():
            raise VisionAPIError("The Vision API response did not contain analysis text.")

        try:
            cleaned = self._extract_json(content)
            data: Any = json.loads(cleaned)
            return ThumbnailAnalysis.model_validate(data)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise VisionAPIError(
                "The model response was not valid Laqta AI analysis JSON. No AI recommendations were displayed. "
                "Try again or select a model that supports image input and JSON mode."
            ) from exc

    @staticmethod
    def _extract_json(content: str) -> str:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("No JSON object found in model response.")
        return text[start : end + 1]

    def _provider_error(self, response: requests.Response) -> str:
        try:
            payload = response.json()
            message = payload.get("error", {}).get("message") or payload.get("message")
            if message:
                detail = str(message)
                if self.settings.vision_api_key:
                    detail = detail.replace(self.settings.vision_api_key, "[redacted]")
                return detail[:500]
        except (ValueError, AttributeError):
            pass
        return "The provider did not return a readable error message."
