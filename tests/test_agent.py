import json
from unittest.mock import MagicMock

from biorxiv_agent.agent_selector import AgentSelector
from biorxiv_agent.biorxiv_client import dedupe_latest_versions, _parse_retry_after
from biorxiv_agent.database import DatabaseManager


PAPER = {
    "doi": "10.1101/2026.01.01.000001",
    "title": "A graph neural network for single-cell trajectory inference",
    "authors": "Doe, J.",
    "category": "bioinformatics",
    "date": "2026-01-01",
    "abstract": "We present a deep learning method for single-cell RNA-seq.",
    "version": "1",
    "server": "biorxiv",
}


def _selector_returning(content: str) -> AgentSelector:
    selector = AgentSelector(research_interests="- Bacterial chemotaxis modeling")
    selector.client = MagicMock()
    selector.client.chat.return_value = {"message": {"content": content}}
    return selector


def test_prompt_includes_custom_interests():
    selector = AgentSelector(research_interests="- Bacterial chemotaxis modeling")
    assert "Bacterial chemotaxis modeling" in selector.build_prompt(PAPER)


def test_evaluation_uses_paper_doi_and_clamps_scores():
    content = json.dumps({
        "doi": "10.1101/wrong",
        "relevance_score": "12",
        "novelty_score": -3,
        "rigor_score": 7.6,
        "recommendation": "download",
        "reasoning": "ok",
    })
    result = _selector_returning(content).evaluate_paper(PAPER)
    assert result["doi"] == PAPER["doi"]
    assert (result["relevance_score"], result["novelty_score"], result["rigor_score"]) == (10, 0, 8)
    assert result["recommendation"] == "DOWNLOAD"
    assert result["source"] == "llm"


def test_invalid_json_uses_fallback():
    result = _selector_returning("not json").evaluate_paper(PAPER)
    assert result["source"] == "fallback"
    assert result["recommendation"] == "DOWNLOAD"  # deep learning + single-cell + graph neural


def test_dedupe_keeps_latest_version():
    papers = [dict(PAPER, version="1"), dict(PAPER, version="3"), dict(PAPER, version="2")]
    assert [p["version"] for p in dedupe_latest_versions(papers)] == ["3"]


def test_parse_retry_after():
    assert _parse_retry_after("30") == 30
    assert _parse_retry_after(None) == 10
    assert _parse_retry_after("garbage") == 10
    assert _parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 10  # date in the past


def test_database_tracks_processing(tmp_path):
    db = DatabaseManager(str(tmp_path / "papers.db"))
    fallback = AgentSelector.fallback_evaluation(PAPER)
    db.save_evaluation(PAPER, fallback)
    assert not db.is_processed(PAPER["doi"])  # fallback results are retried

    db.save_result(PAPER["doi"], {"classification": "computational", "confidence": 0.9, "reasoning": "x"}, "/tmp/a.pdf")
    llm_eval = dict(fallback, source="llm", recommendation="SKIP")
    db.save_evaluation(PAPER, llm_eval)
    assert db.is_processed(PAPER["doi"])
    assert db.has_pdf(PAPER["doi"])  # re-evaluation keeps the download
    assert db.count() == 1
