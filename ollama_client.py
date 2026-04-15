"""
ollama_client.py — Step 2: Ollama LLM integration.

Handles:
- HTTP calls to Ollama /api/chat endpoint
- Retry with exponential backoff
- Timeout management
- Confidence score extraction from response
- Safe JSON parsing of structured LLM output
"""

import json
import logging
import time
from typing import Optional

import httpx
import yaml

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

def _load_ollama_cfg(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)["ollama"]


# ─────────────────────────────────────────────
# Core LLM caller
# ─────────────────────────────────────────────

class OllamaClient:
    """
    Thin wrapper around the Ollama /api/chat endpoint.

    Usage:
        client = OllamaClient()
        result = client.correct_chunk(raw_text, known_names)
    """

    def __init__(self, config_path: str = "config.yaml"):
        cfg = _load_ollama_cfg(config_path)
        self.base_url = cfg["base_url"].rstrip("/")
        self.model = cfg["model"]
        self.timeout = cfg["timeout_seconds"]
        self.max_retries = cfg["max_retries"]

    # ── public interface ──────────────────────

    def correct_chunk(
        self,
        raw_chunk: str,
        known_names: Optional[dict] = None,
    ) -> dict:
        """
        Send a transcript chunk to the LLM for correction.

        Returns a dict with keys:
            cleaned_text      str
            detected_names    list[str]
            corrected_names   dict  {original: corrected}
            uncertain_flags   list[str]
            confidence_score  float  (0.0 – 1.0)

        On LLM failure → returns a safe fallback dict with the original text
        and a flag so the pipeline can route to human review.
        """
        system_prompt = self._build_system_prompt()
        user_message = self._build_user_message(raw_chunk, known_names or {})

        raw_response = self._call_with_retry(system_prompt, user_message)
        if raw_response is None:
            logger.warning("LLM call failed after all retries — routing to human review.")
            return self._fallback_response(raw_chunk)

        return self._parse_llm_response(raw_response, raw_chunk)

    # ── prompt builders ───────────────────────

    @staticmethod
    def _build_system_prompt() -> str:
        return (
            "You are a medical transcription correction assistant. "
            "Your ONLY job is to fix grammar, spelling, and name inconsistencies "
            "in transcribed clinical text — without changing ANY medical meaning.\n\n"
            "STRICT RULES:\n"
            "1. NEVER alter medication names, dosages, or clinical values.\n"
            "2. NEVER add, remove, or invent clinical information.\n"
            "3. If you are uncertain about a name or term, add it to uncertain_flags.\n"
            "4. Preserve original tense and sentence structure as much as possible.\n"
            "5. Return ONLY valid JSON. No explanation text outside the JSON block."
        )

    @staticmethod
    def _build_user_message(raw_chunk: str, known_names: dict) -> str:
        names_section = (
            json.dumps(known_names, ensure_ascii=False)
            if known_names
            else "{}"
        )
        return (
            f"KNOWN NAMES DICTIONARY:\n{names_section}\n\n"
            f"RAW TRANSCRIPT CHUNK:\n{raw_chunk}\n\n"
            "Return a JSON object with exactly these keys:\n"
            "{\n"
            '  "cleaned_text": "<corrected transcript>",\n'
            '  "detected_names": ["<name1>", "<name2>"],\n'
            '  "corrected_names": {"<original>": "<corrected>"},\n'
            '  "uncertain_flags": ["<flag1>"],\n'
            '  "confidence_score": <float between 0.0 and 1.0>\n'
            "}"
        )

    # ── HTTP + retry ──────────────────────────

    def _call_with_retry(
        self,
        system_prompt: str,
        user_message: str,
    ) -> Optional[str]:
        """
        Call Ollama with exponential backoff retry.
        Returns failure on total failure.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "format": "json",       # Ollama structured output mode
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"Ollama call attempt {attempt}/{self.max_retries}")
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(
                        f"{self.base_url}/api/chat",
                        json=payload,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    # Ollama wraps response in message.content
                    content = data.get("message", {}).get("content", "")
                    if content:
                        logger.debug(f"Ollama responded on attempt {attempt}")
                        return content

                    logger.warning(f"Empty content from Ollama on attempt {attempt}")

            except httpx.TimeoutException:
                logger.warning(f"Ollama timeout on attempt {attempt}")
            except httpx.HTTPStatusError as e:
                logger.error(f"Ollama HTTP error {e.response.status_code} on attempt {attempt}")
            except Exception as e:
                logger.error(f"Unexpected Ollama error on attempt {attempt}: {e}")

            if attempt < self.max_retries:
                wait = 2 ** attempt   # 2s, 4s, 8s …
                logger.info(f"Retrying in {wait}s…")
                time.sleep(wait)

        return None

    # ── response parsing ──────────────────────

    @staticmethod
    def _parse_llm_response(raw: str, original_chunk: str) -> dict:
        """
        Safely parse the LLM JSON response.
        Falls back gracefully if JSON is malformed.
        """
        # Strip markdown fences if model wrapped output anyway
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON object from mixed response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    parsed = json.loads(text[start:end])
                except json.JSONDecodeError:
                    logger.error("Could not parse LLM JSON — using fallback.")
                    return OllamaClient._fallback_response(original_chunk)
            else:
                logger.error("No JSON object found in LLM response — using fallback.")
                return OllamaClient._fallback_response(original_chunk)

        # Validate required keys; fill missing ones safely
        return {
            "cleaned_text": parsed.get("cleaned_text", original_chunk),
            "detected_names": parsed.get("detected_names", []),
            "corrected_names": parsed.get("corrected_names", {}),
            "uncertain_flags": parsed.get("uncertain_flags", []),
            "confidence_score": float(parsed.get("confidence_score", 0.5)),
        }

    @staticmethod
    def _fallback_response(original_chunk: str) -> dict:
        """
        Safe fallback when LLM is unavailable or response is unparseable.
        Routes transcript to human review.
        """
        return {
            "cleaned_text": original_chunk,   # Return original unchanged
            "detected_names": [],
            "corrected_names": {},
            "uncertain_flags": ["LLM_UNAVAILABLE — full chunk needs human review"],
            "confidence_score": 0.0,
        }
