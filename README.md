# bioRxiv Intelligent Paper Agent

Autonomous agent that fetches bioRxiv/medRxiv papers, uses LLM (Ollama) to evaluate relevance to your research interests, downloads selected PDFs, and classifies them.

## Features
- **LLM-powered selection** - Agent reads abstracts and decides what's relevant
- **Auto PDF download** - Downloads full PDFs for selected papers
- **Classification** - Tags papers as computational/experimental
- **Summary sheet** - CSV + JSON output for easy tracking
- **Daemon or cron** - Run continuously or scheduled

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Ollama (separate terminal)
ollama pull nemotron-3-nano:4b  # or llama3.1:8b, etc. Depends on the RAM avaialble 
ollama serve

# 3. Customize your interests
# Edit research_interests.txt

# 4. Run once (for cron)
python -m biorxiv_agent.main --once --days 1 --interests research_interests.txt

# 5. Or run as daemon (continuous)
python -m biorxiv_agent.main --daemon --interval 3600 --days 1 --interests research_interests.txt
```

## Output
```
~/biorxiv_agent/
├── pdfs/                    # Downloaded PDFs
├── selected_papers.json     # Full details
└── paper_summary.csv        # Spreadsheet-ready summary
```

## Configuration
Edit `config.py` for:
- `OLLAMA_MODEL` - Change model (e.g., `llama3.1:70b`, `mistral:7b`)
- `UNCERTAINTY_THRESHOLD` - Confidence threshold for PDF fallback (default 0.7)
- `POLL_INTERVAL` - Daemon poll interval in seconds
- `MAX_RETRIES` / `BACKOFF_FACTOR` - API retry behavior
