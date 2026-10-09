import json
import ollama
import logging
from typing import Dict, Any, Optional

from .config import (
    OLLAMA_URL,
    OLLAMA_MODEL,
    AGENT_PROMPT,
    EVALUATION_SCHEMA,
    DEFAULT_RESEARCH_INTERESTS,
    ABSTRACT_MAX_CHARS,
)


logger = logging.getLogger(__name__)

FALLBACK_KEYWORDS = [
    "machine learning", "deep learning", "neural network", "artificial intelligence",
    "bioinformatics", "computational", "genomics", "transcriptomics", "single-cell",
    "protein structure", "protein design", "drug discovery", "systems biology",
    "network analysis", "graph neural", "transformer", "large language model",
    "variant calling", "long-read", "md simulation", "molecular dynamics",
]


def _clamp_score(value: Any) -> int:
    """Coerce an LLM-provided score to an int in [0, 10]."""
    try:
        return max(0, min(10, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0


class AgentSelector:
    def __init__(
        self,
        ollama_url: str = OLLAMA_URL,
        model: str = OLLAMA_MODEL,
        research_interests: Optional[str] = None,
    ):
        self.client = ollama.Client(host=ollama_url)
        self.model = model
        self.research_interests = research_interests or DEFAULT_RESEARCH_INTERESTS

    def check_connection(self) -> bool:
        try:
            self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                options={"num_predict": 1},
            )
            return True
        except Exception as e:
            logger.warning(f"Cannot connect to Ollama ({self.model}): {e}")
            return False

    def build_prompt(self, paper: Dict) -> str:
        """Build prompt for a single paper."""
        abstract = paper.get("abstract", "") or ""
        if len(abstract) > ABSTRACT_MAX_CHARS:
            abstract = abstract[:ABSTRACT_MAX_CHARS] + "..."
        paper_text = (
            f"Title: {paper.get('title', '')}\n"
            f"Authors: {paper.get('authors', '')}\n"
            f"Category: {paper.get('category', '')}\n"
            f"Date: {paper.get('date', '')}\n"
            f"Abstract: {abstract}\n"
        )
        return AGENT_PROMPT.format(research_interests=self.research_interests.strip()) + paper_text

    def evaluate_paper(self, paper: Dict) -> Dict[str, Any]:
        """Evaluate a single paper. Falls back to keyword matching if the LLM fails."""
        prompt = self.build_prompt(paper)
        logger.debug(f"LLM Prompt length: {len(prompt)} chars for {paper.get('doi', 'unknown')}")

        content = ""
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format=EVALUATION_SCHEMA,
                options={"temperature": 0.2},
            )
            content = response["message"]["content"]
            logger.debug(f"LLM Raw response: {content[:500]}")
            eval_data = json.loads(content)
            if isinstance(eval_data, list):
                eval_data = eval_data[0] if eval_data else {}
            if not isinstance(eval_data, dict):
                raise ValueError(f"expected a JSON object, got {type(eval_data).__name__}")
            return self.normalize_evaluation(paper, eval_data)

        except json.JSONDecodeError as e:
            logger.error(f"LLM returned invalid JSON for {paper.get('doi')}: {e}")
            logger.debug(f"Raw content: {content}")
        except Exception as e:
            logger.error(f"LLM evaluation failed for {paper.get('doi')}: {e}")
        return self.fallback_evaluation(paper)

    @staticmethod
    def normalize_evaluation(paper: Dict, eval_data: Dict) -> Dict[str, Any]:
        """Validate LLM output; the DOI always comes from the paper, never the LLM."""
        recommendation = str(eval_data.get("recommendation", "SKIP")).strip().upper()
        if recommendation not in ("DOWNLOAD", "SKIP"):
            recommendation = "SKIP"
        return {
            "doi": paper.get("doi", ""),
            "relevance_score": _clamp_score(eval_data.get("relevance_score")),
            "novelty_score": _clamp_score(eval_data.get("novelty_score")),
            "rigor_score": _clamp_score(eval_data.get("rigor_score")),
            "recommendation": recommendation,
            "reasoning": str(eval_data.get("reasoning") or "No reasoning provided"),
            "source": "llm",
        }

    @staticmethod
    def fallback_evaluation(paper: Dict) -> Dict[str, Any]:
        """Keyword-based evaluation used when the LLM is unavailable."""
        text = f"{paper.get('title', '')} {paper.get('abstract', '')} {paper.get('category', '')}".lower()
        score = sum(1 for kw in FALLBACK_KEYWORDS if kw in text)
        return {
            "doi": paper.get("doi", ""),
            "relevance_score": _clamp_score(score * 1.5),
            "novelty_score": 5,
            "rigor_score": 5,
            "recommendation": "DOWNLOAD" if score >= 2 else "SKIP",
            "reasoning": f"Keyword match score: {score}/{len(FALLBACK_KEYWORDS)} (fallback mode)",
            "source": "fallback",
        }
