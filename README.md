# bioRxiv Intelligent Paper Agent

Autonomous agent that fetches bioRxiv/medRxiv papers, uses LLM (Ollama) to evaluate relevance to your research interests, downloads selected PDFs, and classifies them.

## Features
- **LLM-powered selection** - Agent reads abstracts and decides what's relevant
- **Auto PDF download** - Downloads full PDFs for selected papers
- **Classification** - Tags papers as computational/experimental
- **SQLite database** - Structured storage for papers, evaluations, and classifications
- **Daemon or cron** - Run continuously or scheduled

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install the package (required for `python -m biorxiv_agent.main`)
pip install .

# 3. Start Ollama (separate terminal)
# Default model: nemotron-3.5-lightning:latest (~27GB, requires ~32GB+ RAM)
ollama pull nemotron-3.5-lightning:latest
ollama serve

# Alternative lighter models:
# ollama pull nemotron-3-nano:4b      # ~4GB, requires ~8GB RAM
# ollama pull llama3.1:8b             # ~5GB, requires ~10GB RAM
# ollama pull mistral:7b              # ~4GB, requires ~8GB RAM

# 4. Customize your interests
# Edit research_interests.txt

# 5. Run once (for cron) - from project root:
# Option A: Direct from package directory (no install needed)
cd biorxiv_agent
python3 main.py --once --days 3 --max-papers 10 --interests ../research_interests.txt

# Option B: Module syntax with PYTHONPATH (from project root, no install needed)
cd ..
PYTHONPATH=. python3 -m biorxiv_agent.main --once --days 3 --max-papers 10 --interests research_interests.txt

# Option C: Module syntax (requires pip install .)
python3 -m biorxiv_agent.main --once --days 3 --max-papers 10 --interests research_interests.txt

# 6. Or run as daemon (continuous)
python3 -m biorxiv_agent.main --daemon --interval 3600 --days 1 --interests research_interests.txt
```

## RAM Requirements

| Model | Size | Minimum RAM | Recommended RAM |
|-------|------|-------------|-----------------|
| nemotron-3.5-lightning:latest | 27 GB | 32 GB | 48 GB+ |
| nemotron-3-nano:4b | 4 GB | 8 GB | 12 GB+ |
| llama3.1:8b | 5 GB | 10 GB | 16 GB+ |
| mistral:7b | 4 GB | 8 GB | 12 GB+ |
| llama3.1:70b | 40 GB | 48 GB | 64 GB+ |

**Rule of thumb:** RAM ≥ model size + 4-8 GB for OS/other apps. For nemotron-3.5-lightning (27GB), you need at least 32GB RAM, preferably 48GB+.

## Output
```
biorxiv_agent/output/
├── papers.db                # SQLite database with all papers, evaluations, classifications
├── agent.log                # Agent log file
├── .biorxiv_agent_state.json  # State tracking file
└── pdfs/                    # Downloaded PDFs
```

### Database Schema
The `papers.db` SQLite database contains a `papers` table with:
- `doi` (UNIQUE) - Paper DOI
- `title`, `authors`, `date`, `category`, `abstract` - Paper metadata
- `relevance_score`, `novelty_score`, `rigor_score` - Agent scores (0-10)
- `agent_recommendation` - "DOWNLOAD" or "SKIP"
- `agent_reasoning` - Agent's reasoning
- `classification` - "computational" / "experimental" / "unknown"
- `classification_confidence` - Confidence score (0.0-1.0)
- `classification_reasoning` - Classifier reasoning
- `pdf_downloaded` - 1 if PDF downloaded, 0 otherwise
- `pdf_path` - Local path to downloaded PDF
- `processed_at` - Timestamp when processed

### Querying the Database
```bash
# View all papers
sqlite3 output/papers.db "SELECT doi, title, agent_recommendation, classification FROM papers;"

# View only downloaded papers
sqlite3 output/papers.db "SELECT doi, title, pdf_path FROM papers WHERE pdf_downloaded=1;"

# View papers by recommendation
sqlite3 output/papers.db "SELECT doi, title, relevance_score FROM papers WHERE agent_recommendation='DOWNLOAD' ORDER BY relevance_score DESC;"

# Export to CSV
sqlite3 -header -csv output/papers.db "SELECT * FROM papers;" > paper_summary.csv
```

## Configuration
Edit `biorxiv_agent/config.py` for:
- `OLLAMA_MODEL` - Default: `nemotron-3.5-lightning:latest` (change to lighter model if RAM limited)
- `UNCERTAINTY_THRESHOLD` - Confidence threshold for PDF fallback (default 0.7)
- `POLL_INTERVAL` - Daemon poll interval in seconds
- `MAX_RETRIES` / `BACKOFF_FACTOR` - API retry behavior
- `REQUEST_TIMEOUT` - HTTP request timeout (default 120s)

## Requirements
- Python 3.8+
- Ollama running locally
- **RAM**: 32GB+ for default model, 8GB+ for lighter models
- Dependencies in `requirements.txt`:
  - `requests` - HTTP client for bioRxiv API
  - `pymupdf` - PDF text extraction
  - `ollama` - LLM client
  - `sqlite3` - Built into Python standard library (no install needed)