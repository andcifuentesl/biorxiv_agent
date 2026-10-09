import os
from pathlib import Path

BIORXIV_API_URL = "https://api.biorxiv.org"
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("BIORXIV_AGENT_MODEL", "nemotron-3.5-lightning:latest")
# Output location is resolved once here; override with --output or BIORXIV_AGENT_OUTPUT
OUTPUT_FOLDER = str(Path(os.environ.get("BIORXIV_AGENT_OUTPUT", Path.cwd() / "output")).resolve())
UNCERTAINTY_THRESHOLD = 0.7
REQUEST_TIMEOUT = 120
MAX_RETRIES = 5
BACKOFF_FACTOR = 2
POLL_INTERVAL = 3600
PDF_DOWNLOAD_DELAY = 10
ABSTRACT_MAX_CHARS = 2000

AGENT_PROMPT = """
You are a research paper selection agent. Your job is to decide whether a single bioRxiv/medRxiv
paper matches the user's research interests and is worth downloading.

=== USER RESEARCH INTERESTS ===
{research_interests}
=== END OF INTERESTS ===

SCORING GUIDELINES (unless the interests above say otherwise):
- relevance_score 8-10: the paper's MAIN contribution directly matches a high-priority interest
- relevance_score 5-7: substantial overlap with the interests, but not the main focus
- relevance_score 3-4: only tangentially related
- relevance_score 0-2: unrelated or explicitly excluded by the interests
- novelty_score 0-10: how new the method/finding appears to be
- rigor_score 0-10: how sound the methodology appears from the abstract

Recommend "DOWNLOAD" when relevance_score >= 7, otherwise "SKIP".

Respond with a single JSON object only, with exactly these fields:
- "relevance_score": integer 0-10
- "novelty_score": integer 0-10
- "rigor_score": integer 0-10
- "recommendation": "DOWNLOAD" or "SKIP"
- "reasoning": brief explanation (1-2 sentences)

Example:
{{
  "relevance_score": 9,
  "novelty_score": 8,
  "rigor_score": 7,
  "recommendation": "DOWNLOAD",
  "reasoning": "Develops a new deep learning method for protein structure prediction, a core interest."
}}

Paper to evaluate:
"""

# JSON schema passed to Ollama's `format` so the model is constrained to these fields
EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "relevance_score": {"type": "integer", "minimum": 0, "maximum": 10},
        "novelty_score": {"type": "integer", "minimum": 0, "maximum": 10},
        "rigor_score": {"type": "integer", "minimum": 0, "maximum": 10},
        "recommendation": {"type": "string", "enum": ["DOWNLOAD", "SKIP"]},
        "reasoning": {"type": "string"},
    },
    "required": ["relevance_score", "novelty_score", "rigor_score", "recommendation", "reasoning"],
}

CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "classification": {"type": "string", "enum": ["computational", "experimental"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reasoning": {"type": "string"},
    },
    "required": ["classification", "confidence", "reasoning"],
}

DEFAULT_RESEARCH_INTERESTS = """
HIGH PRIORITY:
- Machine learning / deep learning applications in biology
- Computational biology and bioinformatics
- Genomics, transcriptomics, single-cell analysis
- Protein structure prediction and design
- Systems biology and network analysis
- AI/ML for drug discovery
- MD simulations

EXCLUDE:
- Pure wet-lab papers without a computational contribution
- Clinical trials, case studies, epidemiology
"""
