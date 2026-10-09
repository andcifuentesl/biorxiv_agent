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
# 1. Install the package (adds the `biorxiv-agent` command)
pip install .

# 2. Start Ollama (separate terminal)
# Default model: nemotron-3.5-lightning:latest (~27GB, requires ~32GB+ RAM)
ollama pull nemotron-3.5-lightning:latest
ollama serve

# Alternative lighter models (select with BIORXIV_AGENT_MODEL=<name>):
# ollama pull nemotron-3-nano:4b      # ~4GB, requires ~8GB RAM
# ollama pull llama3.1:8b             # ~5GB, requires ~10GB RAM
# ollama pull mistral:7b              # ~4GB, requires ~8GB RAM

# 3. Customize your interests
# Edit research_interests.txt (it is inserted into the LLM prompt)

# 4. Run once (for cron), from the project root
biorxiv-agent --once --days 3 --max-papers 10 --interests research_interests.txt
# or, without installing:
PYTHONPATH=. python3 -m biorxiv_agent.main --once --days 3 --max-papers 10 --interests research_interests.txt

# 5. Or run as daemon (continuous)
biorxiv-agent --daemon --interval 3600 --days 1 --interests research_interests.txt
```

## Upgrading from 1.0

Version 1.1 changes how the agent is run and what it stores. If you used 1.0:

- **`--interests` now takes effect.** In 1.0 the interests file was silently ignored and a hard-coded
  list was used. Expect different selections; review `research_interests.txt` before running.
- **New way to run.** `pip install .` now installs the package and a `biorxiv-agent` command.
  Running `python3 main.py` from inside `biorxiv_agent/` no longer works (it never did reliably);
  use `biorxiv-agent ...` or `PYTHONPATH=. python3 -m biorxiv_agent.main ...` from the project root.
- **Output location.** Output goes to `./output` relative to where you run the command, or to
  `--output <dir>` / `BIORXIV_AGENT_OUTPUT`. To keep using an old database, point `--output` at
  the folder that contains your existing `papers.db`.
- **No more state file.** `.biorxiv_agent_state.json` is no longer read or written; `papers.db` is
  the record of what has been processed. Existing databases get the new columns automatically.
- **Every evaluated paper is saved**, including SKIPs, so the database grows faster but each paper
  is sent to the LLM only once.
- **Fallback evaluations are retried.** When Ollama is unavailable, papers are scored by keyword
  matching and stored with `eval_source = 'fallback'`; the next run re-evaluates them with the LLM.
- **Classification** uses the abstract first and only reads the PDF when confidence is below
  `UNCERTAINTY_THRESHOLD` (in 1.0 the PDF was always used when available).
- **New options:** `--output`, `-v/--verbose`, and the `BIORXIV_AGENT_MODEL` / `OLLAMA_HOST`
  environment variables.
- **Requirements:** Python 3.9+ and `ollama>=0.4.0` (needed for JSON-schema output).

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
output/                      # ./output by default; change with --output or BIORXIV_AGENT_OUTPUT
├── papers.db                # SQLite database: every evaluated paper (DOWNLOAD and SKIP)
├── agent.log                # Agent log file
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
- `eval_source` - "llm" or "fallback" (keyword matching when Ollama is unavailable; these are re-evaluated next run)
- `server`, `version` - Source server and preprint version

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
- `OLLAMA_MODEL` - Default: `nemotron-3.5-lightning:latest` (or set `BIORXIV_AGENT_MODEL`)
- `OLLAMA_URL` - Default: `http://localhost:11434` (or set `OLLAMA_HOST`)
- `UNCERTAINTY_THRESHOLD` - Below this metadata-classification confidence, the PDF text is used (default 0.7)
- `POLL_INTERVAL` - Daemon poll interval in seconds
- `MAX_RETRIES` / `BACKOFF_FACTOR` - API retry behavior
- `REQUEST_TIMEOUT` - HTTP request timeout (default 120s)

## Requirements
- Python 3.9+
- Ollama running locally
- **RAM**: 32GB+ for default model, 8GB+ for lighter models
- Dependencies in `requirements.txt`:
  - `requests` - HTTP client for bioRxiv API
  - `pymupdf` - PDF text extraction
  - `ollama` - LLM client
  - `sqlite3` - Built into Python standard library (no install needed)