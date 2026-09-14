import time
import json
import signal
import argparse
import csv
import logging
from datetime import datetime
from pathlib import Path

from config import (
    OUTPUT_FOLDER,
    RESULTS_FILE,
    PDFS_FOLDER,
    POLL_INTERVAL,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    BACKOFF_FACTOR,
)

from state import StateManager
from biorxiv_client import BioRxivClient
from agent_selector import AgentSelector
from pdf_extractor import extract_first_n_pages
from classifier import Classifier


def setup_logging():
    """Setup logging to both file and console."""
    log_path = Path(OUTPUT_FOLDER) / "agent.log"
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    
    # Reduce verbose HTTP/urllib3/httpx logs
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


class BioRxivAgent:
    def __init__(self, poll_interval: int = 3600, research_interests: str = None):
        self.state = StateManager()
        self.biorxiv = BioRxivClient()
        self.selector = AgentSelector(research_interests=research_interests)
        self.classifier = Classifier()
        self.running = True
        self.poll_interval = poll_interval
        
        Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
        Path(PDFS_FOLDER).mkdir(parents=True, exist_ok=True)
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logger.info("Shutdown signal received. Saving state...")
        self.running = False

    def _save_summary_sheet(self, papers: list, evaluations: list, results: list) -> None:
        csv_path = Path(OUTPUT_FOLDER) / "paper_summary.csv"
        
        rows = []
        for paper, eval_data, result in zip(papers, evaluations, results):
            rows.append({
                "doi": paper.get("doi"),
                "title": paper.get("title"),
                "authors": paper.get("authors"),
                "date": paper.get("date"),
                "category": paper.get("category"),
                "relevance_score": eval_data.get("relevance_score"),
                "novelty_score": eval_data.get("novelty_score"),
                "rigor_score": eval_data.get("rigor_score"),
                "agent_recommendation": eval_data.get("recommendation"),
                "agent_reasoning": eval_data.get("reasoning"),
                "classification": result.get("classification") if result else "N/A",
                "classification_confidence": result.get("confidence") if result else "N/A",
                "pdf_downloaded": "YES" if result else "NO",
                "pdf_path": result.get("pdf_path") if result else "",
                "processed_at": datetime.now().isoformat(),
            })
        
        fieldnames = list(rows[0].keys()) if rows else []
        file_exists = csv_path.exists()
        
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)
        
        print(f"  Summary sheet updated: {csv_path}")

    def _save_json_results(self, papers: list, evaluations: list, results: list) -> None:
        json_path = Path(OUTPUT_FOLDER) / RESULTS_FILE
        
        existing = []
        if json_path.exists():
            try:
                with open(json_path, "r") as f:
                    existing = json.load(f)
            except:
                pass
        
        for paper, eval_data, result in zip(papers, evaluations, results):
            existing.append({
                "doi": paper.get("doi"),
                "title": paper.get("title"),
                "authors": paper.get("authors"),
                "date": paper.get("date"),
                "category": paper.get("category"),
                "abstract": paper.get("abstract", "")[:500],
                "evaluation": eval_data,
                "classification": result.get("classification") if result else None,
                "classification_confidence": result.get("confidence") if result else None,
                "classification_reasoning": result.get("reasoning") if result else None,
                "pdf_path": result.get("pdf_path") if result else None,
                "processed_at": datetime.now().isoformat(),
            })
        
        with open(json_path, "w") as f:
            json.dump(existing, f, indent=2)

    def run_cycle(self, days_back: int = 1, server: str = "biorxiv", max_papers: int = 50) -> int:
        logger.info("=" * 60)
        logger.info(f"Fetching papers from last {days_back} day(s)...")
        
        papers = self.biorxiv.get_recent_papers(days_back, server)
        if not papers:
            logger.warning("No papers found")
            return 0
        
        new_papers = [p for p in papers if not self.state.is_processed(p.get("doi", ""))]
        
        logger.info(f"Found {len(new_papers)} new papers (of {len(papers)} total)")
        
        if not new_papers:
            return 0
        
        logger.info("Agent evaluating all papers...")
        evaluations = self.selector.evaluate_papers(new_papers)
        
        # Pair papers with evaluations and sort by relevance score (descending)
        paper_eval_pairs = list(zip(new_papers, evaluations))
        paper_eval_pairs.sort(key=lambda x: (
            x[1].get("relevance_score", 0),
            x[1].get("novelty_score", 0),
            x[1].get("rigor_score", 0)
        ), reverse=True)
        
        # Filter to only DOWNLOAD recommendations, then take top max_papers
        download_candidates = [(p, e) for p, e in paper_eval_pairs if e.get("recommendation") == "DOWNLOAD"]
        top_papers = download_candidates[:max_papers]
        
        logger.info(f"Agent recommended {len(download_candidates)} papers for DOWNLOAD, "
                   f"selecting top {len(top_papers)} for download")
        
        results = []
        downloaded = 0
        
        for paper, eval_data in top_papers:
            if not self.running:
                break
                
            doi = paper.get("doi", "")
            title = paper.get("title", "")[:70]
            
            logger.info(f"Paper: {title}... | DOI: {doi}")
            logger.info(f"  Agent: DOWNLOAD (relevance: {eval_data.get('relevance_score', 0)}/10, "
                       f"novelty: {eval_data.get('novelty_score', 0)}/10, "
                       f"rigor: {eval_data.get('rigor_score', 0)}/10)")
            logger.debug(f"  Reasoning: {eval_data.get('reasoning', '')}")
            
            self.state.mark_processed(doi)
            
            pdf_bytes = self.biorxiv.download_pdf(doi, paper.get("version", 1))
            pdf_path = None
            classification_result = None
            
            if pdf_bytes:
                pdf_path = self.biorxiv.save_pdf(doi, pdf_bytes)
                logger.info(f"  PDF saved: {pdf_path}")
                
                fulltext = extract_first_n_pages(pdf_bytes, n=3)
                if fulltext:
                    classification_result = self.classifier.classify_fulltext(
                        paper.get("title", ""), fulltext
                    )
                    logger.info(f"  Classification (fulltext): {classification_result.get('classification')} "
                          f"(confidence: {classification_result.get('confidence', 0):.2f})")
                    logger.debug(f"  Classification reasoning: {classification_result.get('reasoning', '')}")
                
                self.state.mark_downloaded(doi)
                downloaded += 1
                
                # Small delay between PDF downloads to avoid rate limiting
                time.sleep(10)
            else:
                classification_result = self.classifier.classify_metadata(
                    paper.get("title", ""),
                    paper.get("abstract", ""),
                    [paper.get("category", "")] if paper.get("category") else []
                )
                logger.info(f"  Classification (metadata): {classification_result.get('classification')} "
                      f"(confidence: {classification_result.get('confidence', 0):.2f})")
                logger.debug(f"  Classification reasoning: {classification_result.get('reasoning', '')}")
            
            self.state.mark_selected(doi)
            results.append({
                "classification": classification_result.get("classification") if classification_result else "unknown",
                "confidence": classification_result.get("confidence") if classification_result else 0.0,
                "reasoning": classification_result.get("reasoning") if classification_result else "",
                "pdf_path": pdf_path,
            })
        
        # Save all evaluations (not just downloaded ones) for record keeping
        selected_papers = [p for p, e in top_papers]
        selected_evals = [e for p, e in top_papers]
        selected_results = results
        
        if selected_papers:
            self._save_summary_sheet(selected_papers, selected_evals, selected_results)
            self._save_json_results(selected_papers, selected_evals, selected_results)
        
        logger.info(f"Cycle complete: {downloaded} PDFs downloaded from {len(top_papers)} selected")
        return downloaded

    def run_daemon(self, days_back: int = 1, server: str = "biorxiv", max_papers: int = 50) -> None:
        logger.info(f"Starting BioRxiv Agent Daemon - polling every {self.poll_interval}s")
        logger.info(f"Research interests loaded")
        logger.info("Press Ctrl+C to stop")
        
        if not self.biorxiv.check_connection():
            logger.error("Cannot connect to bioRxiv API")
            return
        logger.info("Connected to bioRxiv API")
        
        try:
            self.selector.client.chat(
                model=self.selector.model,
                messages=[{"role": "user", "content": "test"}],
                options={"num_predict": 1},
            )
            logger.info(f"Connected to Ollama ({self.selector.model})")
        except Exception as e:
            logger.warning(f"Cannot connect to Ollama: {e}")
        
        while self.running:
            try:
                self.run_cycle(days_back, server, max_papers)
            except Exception as e:
                logger.error(f"Cycle error: {e}")
            
            for _ in range(self.poll_interval):
                if not self.running:
                    break
                time.sleep(1)
        
        logger.info("Agent stopped")


def main():
    parser = argparse.ArgumentParser(description="bioRxiv Intelligent Paper Agent")
    parser.add_argument("--daemon", action="store_true", help="Run continuously as daemon")
    parser.add_argument("--interval", type=int, default=3600, help="Poll interval in seconds")
    parser.add_argument("--days", type=int, default=1, help="Days back to check")
    parser.add_argument("--server", choices=["biorxiv", "medrxiv"], default="biorxiv")
    parser.add_argument("--max-papers", type=int, default=50, help="Max papers per cycle")
    parser.add_argument("--interests", type=str, help="Path to research interests file")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()
    
    research_interests = None
    if args.interests:
        try:
            with open(args.interests, "r") as f:
                research_interests = f.read()
            logger.info(f"Loaded custom research interests from {args.interests}")
        except Exception as e:
            logger.error(f"Failed to load interests file: {e}")
    
    agent = BioRxivAgent(poll_interval=args.interval, research_interests=research_interests)
    
    logger.info("Starting bioRxiv Intelligent Paper Agent")
    logger.info(f"Output: {OUTPUT_FOLDER}")
    logger.info(f"PDFs: {PDFS_FOLDER}")
    
    if args.daemon:
        agent.run_daemon(args.days, args.server, args.max_papers)
    else:
        agent.run_cycle(args.days, args.server, args.max_papers)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
