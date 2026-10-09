import time
import signal
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

from .config import (
    OUTPUT_FOLDER,
    POLL_INTERVAL,
    PDF_DOWNLOAD_DELAY,
)

from .biorxiv_client import BioRxivClient
from .agent_selector import AgentSelector
from .pdf_extractor import extract_first_n_pages
from .classifier import Classifier
from .database import DatabaseManager


logger = logging.getLogger("biorxiv_agent")


def setup_logging(output_folder: str, verbose: bool = False) -> None:
    """Setup logging to both file and console."""
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    log_path = Path(output_folder) / "agent.log"

    # Reduce verbose HTTP/urllib3/httpx logs
    for noisy in ("urllib3", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


class BioRxivAgent:
    def __init__(
        self,
        poll_interval: int = POLL_INTERVAL,
        research_interests: Optional[str] = None,
        output_folder: str = OUTPUT_FOLDER,
    ):
        self.output_folder = Path(output_folder)
        self.biorxiv = BioRxivClient(pdfs_folder=str(self.output_folder / "pdfs"))
        self.selector = AgentSelector(research_interests=research_interests)
        self.classifier = Classifier()
        self.db = DatabaseManager(str(self.output_folder / "papers.db"))
        self.running = True
        self.poll_interval = poll_interval

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logger.info("Shutdown signal received. Finishing current paper...")
        self.running = False

    def _classify(self, paper: Dict, pdf_bytes: Optional[bytes]) -> Dict:
        """Classify from metadata first; only read the PDF when the model is unsure."""
        result = self.classifier.classify_metadata(
            paper.get("title", ""),
            paper.get("abstract", ""),
            [paper["category"]] if paper.get("category") else [],
        )
        source = "metadata"
        if pdf_bytes and self.classifier.should_use_fulltext(result.get("confidence", 0.0)):
            fulltext = extract_first_n_pages(pdf_bytes, n=3)
            if fulltext:
                result = self.classifier.classify_fulltext(paper.get("title", ""), fulltext)
                source = "fulltext"
        logger.info(f"  Classification ({source}): {result.get('classification')} "
                    f"(confidence: {result.get('confidence', 0):.2f})")
        logger.debug(f"  Classification reasoning: {result.get('reasoning', '')}")
        return result

    def run_cycle(self, days_back: int = 1, server: str = "biorxiv", max_papers: int = 50) -> int:
        logger.info("=" * 60)
        logger.info(f"Fetching {server} papers from last {days_back} day(s)...")

        papers = self.biorxiv.get_recent_papers(days_back, server)
        if not papers:
            logger.warning("No papers found")
            return 0

        new_papers = [p for p in papers if not self.db.is_processed(p.get("doi", ""))]
        logger.info(f"Found {len(new_papers)} new papers (of {len(papers)} total)")
        if not new_papers:
            return 0

        # 1. Evaluate every new paper and persist the decision immediately
        evaluated = []
        for i, paper in enumerate(new_papers, 1):
            if not self.running:
                break
            logger.info(f"Evaluating paper {i}/{len(new_papers)}: {paper.get('title', '')[:60]}...")
            eval_data = self.selector.evaluate_paper(paper)
            self.db.save_evaluation(paper, eval_data)
            evaluated.append((paper, eval_data))
            logger.info(f"  Result: {eval_data['recommendation']} "
                        f"(relevance: {eval_data['relevance_score']}/10, "
                        f"novelty: {eval_data['novelty_score']}/10, "
                        f"rigor: {eval_data['rigor_score']}/10) "
                        f"- {eval_data['reasoning'][:100]}")

        # 2. Pick the top DOWNLOAD recommendations
        candidates = [
            (p, e) for p, e in evaluated
            if e["recommendation"] == "DOWNLOAD" and not self.db.has_pdf(p.get("doi", ""))
        ]
        candidates.sort(
            key=lambda x: (x[1]["relevance_score"], x[1]["novelty_score"], x[1]["rigor_score"]),
            reverse=True,
        )
        top_papers = candidates[:max_papers]
        logger.info(f"Agent recommended {len(candidates)} papers for DOWNLOAD, "
                    f"selecting top {len(top_papers)} for download")

        # 3. Download, classify, and record each selected paper
        downloaded = 0
        for paper, eval_data in top_papers:
            if not self.running:
                break

            doi = paper.get("doi", "")
            logger.info(f"Paper: {paper.get('title', '')[:70]}... | DOI: {doi}")

            pdf_bytes = self.biorxiv.download_pdf(doi, paper.get("version", 1), paper.get("server", server))
            pdf_path = self.biorxiv.save_pdf(doi, pdf_bytes) if pdf_bytes else None
            if pdf_path:
                logger.info(f"  PDF saved: {pdf_path}")
                downloaded += 1

            classification = self._classify(paper, pdf_bytes)
            self.db.save_result(doi, classification, pdf_path)

            if pdf_bytes:
                # Small delay between PDF downloads to avoid rate limiting
                time.sleep(PDF_DOWNLOAD_DELAY)

        logger.info(f"Cycle complete: {downloaded} PDFs downloaded from {len(top_papers)} selected "
                    f"({self.db.count()} papers in database)")
        return downloaded

    def run_daemon(self, days_back: int = 1, server: str = "biorxiv", max_papers: int = 50) -> None:
        logger.info(f"Starting BioRxiv Agent Daemon - polling every {self.poll_interval}s")
        logger.info("Press Ctrl+C to stop")

        if not self.biorxiv.check_connection():
            logger.error("Cannot connect to bioRxiv API")
            return
        logger.info("Connected to bioRxiv API")

        while self.running:
            try:
                self.run_cycle(days_back, server, max_papers)
            except Exception:
                logger.exception("Cycle error")

            for _ in range(self.poll_interval):
                if not self.running:
                    break
                time.sleep(1)

        logger.info("Agent stopped")


def main():
    parser = argparse.ArgumentParser(description="bioRxiv Intelligent Paper Agent")
    parser.add_argument("--daemon", action="store_true", help="Run continuously as daemon")
    parser.add_argument("--interval", type=int, default=POLL_INTERVAL, help="Poll interval in seconds")
    parser.add_argument("--days", type=int, default=1, help="Days back to check")
    parser.add_argument("--server", choices=["biorxiv", "medrxiv"], default="biorxiv")
    parser.add_argument("--max-papers", type=int, default=50, help="Max papers to download per cycle")
    parser.add_argument("--interests", type=str, help="Path to research interests file")
    parser.add_argument("--output", type=str, default=OUTPUT_FOLDER,
                        help="Output folder for database, PDFs and log (default: ./output)")
    parser.add_argument("--once", action="store_true", help="Run once and exit (default)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    output_folder = str(Path(args.output).resolve())
    setup_logging(output_folder, args.verbose)

    research_interests = None
    if args.interests:
        try:
            research_interests = Path(args.interests).read_text(encoding="utf-8")
            logger.info(f"Loaded custom research interests from {args.interests}")
        except OSError as e:
            parser.error(f"Failed to load interests file: {e}")
    else:
        logger.info("No --interests file given; using built-in default interests")

    agent = BioRxivAgent(
        poll_interval=args.interval,
        research_interests=research_interests,
        output_folder=output_folder,
    )

    logger.info("Starting bioRxiv Intelligent Paper Agent")
    logger.info(f"Output: {output_folder}")

    if agent.selector.check_connection():
        logger.info(f"Connected to Ollama ({agent.selector.model})")
    else:
        logger.warning("Ollama unavailable: using keyword fallback; these papers will be re-evaluated next run")

    if args.daemon:
        agent.run_daemon(args.days, args.server, args.max_papers)
    else:
        agent.run_cycle(args.days, args.server, args.max_papers)

    logger.info("Done!")


if __name__ == "__main__":
    main()
