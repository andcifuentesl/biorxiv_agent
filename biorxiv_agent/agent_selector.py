import json
import ollama
import logging
from typing import List, Dict, Any

from .config import (
    OLLAMA_URL,
    OLLAMA_MODEL,
    AGENT_PROMPT,
    DEFAULT_RESEARCH_INTERESTS,
)


logger = logging.getLogger(__name__)


class AgentSelector:
    def __init__(
        self,
        ollama_url: str = OLLAMA_URL,
        model: str = OLLAMA_MODEL,
        research_interests: str = None,
    ):
        self.client = ollama.Client(host=ollama_url)
        self.model = model
        self.research_interests = research_interests or DEFAULT_RESEARCH_INTERESTS

    def _build_single_prompt(self, paper: Dict) -> str:
        """Build prompt for a single paper."""
        papers_text = f"\n--- Paper ---\n"
        papers_text += f"DOI: {paper.get('doi', '')}\n"
        papers_text += f"Title: {paper.get('title', '')}\n"
        papers_text += f"Authors: {paper.get('authors', '')}\n"
        papers_text += f"Category: {paper.get('category', '')}\n"
        papers_text += f"Date: {paper.get('date', '')}\n"
        abstract = paper.get('abstract', '')
        papers_text += f"Abstract: {abstract[:2000]}...\n" if len(abstract) > 2000 else f"Abstract: {abstract}\n"

        return AGENT_PROMPT.format(research_interests=self.research_interests) + papers_text

    def _evaluate_single(self, paper: Dict) -> Dict[str, Any]:
        """Evaluate a single paper."""
        prompt = self._build_single_prompt(paper)
        
        logger.debug(f"LLM Prompt length: {len(prompt)} chars for {paper.get('doi', 'unknown')}")
        
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0.2},
            )
            content = response["message"]["content"]
            logger.debug(f"LLM Raw response: {content[:500]}...")
            
            eval_data = json.loads(content)
            
            # Handle if LLM returns a single object instead of array
            if isinstance(eval_data, list):
                eval_data = eval_data[0] if eval_data else {}
            
            # Validate and ensure all fields
            return {
                "doi": eval_data.get("doi", paper.get("doi", "")),
                "relevance_score": eval_data.get("relevance_score", 0),
                "novelty_score": eval_data.get("novelty_score", 0),
                "rigor_score": eval_data.get("rigor_score", 0),
                "recommendation": eval_data.get("recommendation", "SKIP"),
                "reasoning": eval_data.get("reasoning", "No reasoning provided"),
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"LLM returned invalid JSON for {paper.get('doi')}: {e}")
            logger.debug(f"Raw content: {content}")
            return self._fallback_single(paper)
        except Exception as e:
            logger.error(f"LLM evaluation failed for {paper.get('doi')}: {e}")
            return self._fallback_single(paper)

    def evaluate_papers(self, papers: List[Dict]) -> List[Dict[str, Any]]:
        """Evaluate papers SEQUENTIALLY (one at a time) for reliability."""
        if not papers:
            return []

        evaluations = []
        for i, paper in enumerate(papers):
            logger.info(f"Evaluating paper {i+1}/{len(papers)}: {paper.get('title', '')[:60]}...")
            eval_data = self._evaluate_single(paper)
            evaluations.append(eval_data)
            
            logger.info(f"  Result: {eval_data.get('recommendation')} "
                       f"(relevance: {eval_data.get('relevance_score')}/10, "
                       f"novelty: {eval_data.get('novelty_score')}/10, "
                       f"rigor: {eval_data.get('rigor_score')}/10) "
                       f"- {eval_data.get('reasoning', '')[:100]}")
        
        return evaluations

    def _fallback_single(self, paper: Dict) -> Dict[str, Any]:
        """Fallback for a single paper."""
        keywords = [
            "machine learning", "deep learning", "neural network", "AI", "artificial intelligence",
            "bioinformatics", "computational", "genomics", "transcriptomics", "single-cell",
            "protein structure", "protein design", "drug discovery", "systems biology",
            "network analysis", "graph neural", "transformer", "LLM", "large language model",
            "variant calling", "long-read", "MD simulation", "molecular dynamics"
        ]
        
        text = f"{paper.get('title', '')} {paper.get('abstract', '')} {paper.get('category', '')}".lower()
        score = sum(1 for kw in keywords if kw in text)
        recommendation = "DOWNLOAD" if score >= 2 else "SKIP"
        
        return {
            "doi": paper.get("doi", ""),
            "relevance_score": min(score * 1.5, 10),
            "novelty_score": 5,
            "rigor_score": 5,
            "recommendation": recommendation,
            "reasoning": f"Keyword match score: {score}/{len(keywords)} (fallback mode)"
        }