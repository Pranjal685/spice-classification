"""
src/fallback.py
---------------
Tier-2 Multimodal Fallback Engine for the Indian Spice Classification project.

Uses Google Gemini Vision (gemini-2.5-flash) as a zero-shot VLM arbitrator
when the local MobileNetV3 model's confidence falls below the 75% gate.

Provides:
  - SpiceFallbackEngine  -- production-grade class with graceful degradation

Usage:
    from fallback import SpiceFallbackEngine
    engine = SpiceFallbackEngine()
    result = engine.identify_unknown_image("path/to/image.jpg")
"""

import json
import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv

# ── load .env from project root -------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

logger = logging.getLogger("spice.fallback")

# ── The 14 classes the local CNN knows about -------------------------------
_LOCAL_CLASSES = [
    "black_cardamom", "black_pepper", "chilli_byadgi", "chilli_guntur",
    "chilli_lavangi", "chilli_sankeshwari", "cumin", "dagad_phool",
    "dalchini", "dhaniya", "dried_coconut", "elaichi_dana",
    "elaichi_pod", "halkund",
]

# ── System prompt for Gemini -----------------------------------------------
_SYSTEM_PROMPT = """\
You are a specialist culinary-spice identification assistant.

Given a photograph, perform these steps:
1. Determine if the image contains a culinary spice or condiment.
2. If YES: identify the spice as precisely as possible (common name, regional
   Indian names if applicable, and scientific binomial).
3. Determine whether the spice is among the following 14 classes that our
   local classifier covers: {classes}.
4. Return ONLY a valid JSON object (no markdown fences, no commentary)
   matching EXACTLY this schema:

{{
  "is_spice": <bool>,
  "identified_name": "<string: common name with regional name in parentheses>",
  "scientific_name": "<string: binomial or 'N/A'>",
  "is_in_dataset_scope": <bool: true if it matches one of the 14 local classes>,
  "confidence_assessment": "<string: brief rationale for your identification>",
  "flavor_profile": "<string: taste and aroma description, or 'N/A'>",
  "common_uses": "<string: culinary applications, or 'N/A'>",
  "action_recommendation": "<string: next-step guidance>"
}}

If the image does NOT contain a spice:
- Set is_spice to false, identified_name to "None", scientific_name to "N/A",
  is_in_dataset_scope to false, flavor_profile to "N/A", common_uses to "N/A",
  confidence_assessment to a brief explanation, and action_recommendation to
  "Please upload a clear photograph of a culinary spice."

CRITICAL: Return ONLY the JSON object. No markdown code fences. No extra text.
""".format(classes=", ".join(_LOCAL_CLASSES))


# ── Error response templates -----------------------------------------------
def _api_error_response(reason: str) -> dict:
    """Return a structured error dict when the API call fails."""
    return {
        "is_spice": None,
        "identified_name": "FALLBACK_ERROR",
        "scientific_name": "N/A",
        "is_in_dataset_scope": False,
        "confidence_assessment": f"Tier-2 API call failed: {reason}",
        "flavor_profile": "N/A",
        "common_uses": "N/A",
        "action_recommendation": (
            "The Gemini fallback could not be reached. "
            "Please check your API key and network connectivity."
        ),
        "error": True,
        "error_reason": reason,
    }


def _stub_response() -> dict:
    """Return a structured response when no API key is configured."""
    return {
        "is_spice": None,
        "identified_name": "STUB_MODE",
        "scientific_name": "N/A",
        "is_in_dataset_scope": False,
        "confidence_assessment": (
            "Tier-2 Gemini fallback is in STUB mode because "
            "GEMINI_API_KEY is not configured."
        ),
        "flavor_profile": "N/A",
        "common_uses": "N/A",
        "action_recommendation": (
            "Set GEMINI_API_KEY in your .env file to enable "
            "the multimodal fallback engine."
        ),
        "error": True,
        "error_reason": "NO_API_KEY",
    }


# ── Main engine class ------------------------------------------------------
class SpiceFallbackEngine:
    """Zero-shot VLM fallback using Google Gemini Vision.

    Gracefully degrades if the API key is missing or the call fails.
    """

    def __init__(self):
        keys_env = os.environ.get("GEMINI_API_KEYS", "") or os.environ.get("GEMINI_API_KEY", "")
        self.api_keys = [k.strip() for k in keys_env.split(",") if k.strip() and k.strip() != "your_gemini_api_key_here"]
        self.candidate_models = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-3.5-flash"]
        self.model_name = self.candidate_models[0]
        self._clients = []
        self._current_client_idx = 0
        self.stub_mode = False

        if not self.api_keys:
            logger.warning(
                "GEMINI_API_KEY(S) is not set or is the placeholder value. "
                "SpiceFallbackEngine will run in STUB mode."
            )
            self.stub_mode = True
        else:
            try:
                from google import genai
                for key in self.api_keys:
                    self._clients.append(genai.Client(api_key=key))
                logger.info("SpiceFallbackEngine initialised with %d Gemini API keys.", len(self._clients))
            except Exception as exc:
                logger.error("Failed to initialise Gemini clients: %s", exc)
                self.stub_mode = True

    # ── public API ---------------------------------------------------------
    def identify_unknown_image(self, image_path: str | Path) -> dict:
        """Send an image to Gemini Vision for zero-shot spice identification.

        Args:
            image_path: Path to the image file.

        Returns:
            A dict conforming to the Tier-2 JSON schema, or an error dict.
        """
        if self.stub_mode:
            return _stub_response()

        image_path = Path(image_path)
        if not image_path.exists():
            return _api_error_response(f"Image file not found: {image_path}")

        try:
            return self._call_gemini(image_path)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            logger.error("Gemini fallback failed: %s", reason)
            return _api_error_response(reason)

    # ── private Gemini call ------------------------------------------------
    def _call_gemini(self, image_path: Path) -> dict:
        """Execute the actual Gemini API call and parse the JSON response."""
        from google.genai import types
        import mimetypes

        # Determine MIME type
        mime, _ = mimetypes.guess_type(str(image_path))
        if mime is None:
            mime = "image/jpeg"

        # Read image bytes
        image_bytes = image_path.read_bytes()

        response = None
        last_error = None

        # Iterate through models (gemini-2.5-flash first, then failover)
        for model in self.candidate_models:
            attempts = 0
            while attempts < len(self._clients):
                client = self._clients[self._current_client_idx]
                try:
                    logger.info("Querying Gemini fallback using model %s with key idx %d...", model, self._current_client_idx)
                    response = client.models.generate_content(
                        model=model,
                        contents=[
                            types.Part.from_bytes(data=image_bytes, mime_type=mime),
                            "Identify the spice in this image. Return ONLY a JSON object as specified.",
                        ],
                        config=types.GenerateContentConfig(
                            system_instruction=_SYSTEM_PROMPT,
                            temperature=0.1,
                            max_output_tokens=2048,
                        ),
                    )
                    self.model_name = model
                    break
                except Exception as exc:
                    last_error = exc
                    exc_str = str(exc)
                    if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str or "404" in exc_str or "503" in exc_str or "UNAVAILABLE" in exc_str:
                        logger.warning("Model %s / key %d quota or availability issue: %s", model, self._current_client_idx, exc_str[:120])
                        self._current_client_idx = (self._current_client_idx + 1) % len(self._clients)
                        attempts += 1
                    else:
                        raise exc

            if response is not None:
                break

        if response is None:
            raise Exception(f"All Gemini models and keys exhausted or unavailable: {last_error}")

        raw_text = response.text.strip()

        # Strip markdown code fences if the model wraps them anyway
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

        # Attempt JSON parse with truncation recovery
        try:
            result = json.loads(raw_text)
        except json.JSONDecodeError:
            # Try to repair truncated JSON by closing open strings/braces
            repaired = raw_text.rstrip()
            # Close an unterminated string value
            if repaired.count('"') % 2 != 0:
                repaired += '"'
            # Close missing braces
            open_braces = repaired.count('{') - repaired.count('}')
            if open_braces > 0:
                repaired += '}' * open_braces
            result = json.loads(repaired)

        # Validate minimal schema keys
        required = {
            "is_spice", "identified_name", "scientific_name",
            "is_in_dataset_scope", "confidence_assessment",
            "flavor_profile", "common_uses", "action_recommendation",
        }
        missing = required - set(result.keys())
        if missing:
            logger.warning("Gemini response missing keys: %s", missing)
            for key in missing:
                result[key] = "N/A"

        result["error"] = False
        return result

    # ── string repr --------------------------------------------------------
    def __repr__(self) -> str:
        mode = "STUB" if self.stub_mode else "LIVE"
        return f"SpiceFallbackEngine(model={self.model_name!r}, mode={mode})"
