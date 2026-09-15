import json
import ollama
from typing import Optional, Dict, Any

from .config import (
    OLLAMA_URL,
    OLLAMA_MODEL,
    UNCERTAINTY_THRESHOLD,
)


class Classifier:
    def __init__(
        self,
        ollama_url: str = OLLAMA_URL,
        model: str = OLLAMA_MODEL,
    ):
        self.client = ollama.Client(host=ollama_url)
        self.model = model

    def _call_llm(self, prompt: str) -> Optional[Dict[str, Any]]:
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0.1},
            )
            content = response["message"]["content"]
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"LLM returned invalid JSON: {e}")
            return None
        except Exception as e:
            print(f"LLM call failed: {e}")
            return None

    def _build_metadata_prompt(self, title: str, abstract: str, tags: list) -> str:
        tags_str = ", ".join(tags) if tags else "none"
        abstract_short = abstract[:2000] if abstract else "none"
        return f"""Classify this research paper as "computational" or "experimental" based on title, abstract, and category.

Title: {title}
Abstract: {abstract_short}
Category: {tags_str}

Respond with JSON only:
{{
  "classification": "computational|experimental",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}"""

    def _build_fulltext_prompt(self, title: str, text: str) -> str:
        text_short = text[:12000] if text else "none"
        return f"""Classify this research paper as "computational" or "experimental" based on full text content.

Title: {title}
Content (first 3 pages): {text_short}

Respond with JSON only:
{{
  "classification": "computational|experimental",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}"""

    def classify_metadata(self, title: str, abstract: str, tags: list) -> Optional[Dict[str, Any]]:
        prompt = self._build_metadata_prompt(title, abstract, tags)
        result = self._call_llm(prompt)

        if result and self._validate_result(result):
            return result

        print("Retrying classification with stricter prompt...")
        strict_prompt = prompt + "\n\nIMPORTANT: Respond with valid JSON only."
        result = self._call_llm(strict_prompt)

        if result and self._validate_result(result):
            return result

        return {
            "classification": "unknown",
            "confidence": 0.0,
            "reasoning": "Classification failed after retries",
        }

    def classify_fulltext(self, title: str, text: str) -> Optional[Dict[str, Any]]:
        prompt = self._build_fulltext_prompt(title, text)
        result = self._call_llm(prompt)

        if result and self._validate_result(result):
            return result

        print("Retrying fulltext classification with stricter prompt...")
        strict_prompt = prompt + "\n\nIMPORTANT: Respond with valid JSON only."
        result = self._call_llm(strict_prompt)

        if result and self._validate_result(result):
            return result

        return {
            "classification": "unknown",
            "confidence": 0.0,
            "reasoning": "Fulltext classification failed after retries",
        }

    def _validate_result(self, result: Dict[str, Any]) -> bool:
        required_keys = {"classification", "confidence", "reasoning"}
        if not all(k in result for k in required_keys):
            return False

        if result["classification"] not in ("computational", "experimental", "unknown"):
            return False

        try:
            conf = float(result["confidence"])
            if not 0.0 <= conf <= 1.0:
                return False
        except (ValueError, TypeError):
            return False

        return True

    def should_use_fulltext(self, confidence: float) -> bool:
        return confidence < UNCERTAINTY_THRESHOLD