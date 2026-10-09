import time
import requests
import logging
from email.utils import parsedate_to_datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta, timezone
from requests.exceptions import JSONDecodeError

from .config import (
    BIORXIV_API_URL,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    BACKOFF_FACTOR,
    OUTPUT_FOLDER,
)


logger = logging.getLogger(__name__)

PDF_HOSTS = {
    "biorxiv": "https://www.biorxiv.org",
    "medrxiv": "https://www.medrxiv.org",
}


def _parse_retry_after(value: Optional[str], default: int = 10) -> int:
    """Retry-After may be a number of seconds or an HTTP date."""
    if not value:
        return default
    try:
        seconds = int(value)
    except ValueError:
        try:
            seconds = int((parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError):
            return default
    return seconds if seconds >= 1 else default


def _version_number(paper: Dict) -> int:
    try:
        return int(paper.get("version", 1))
    except (TypeError, ValueError):
        return 1


def dedupe_latest_versions(papers: List[Dict]) -> List[Dict]:
    """The API returns one entry per version; keep only the latest version of each DOI."""
    latest: Dict[str, Dict] = {}
    for paper in papers:
        doi = paper.get("doi")
        if not doi:
            continue
        if doi not in latest or _version_number(paper) > _version_number(latest[doi]):
            latest[doi] = paper
    return list(latest.values())


class BioRxivClient:
    def __init__(self, api_url: str = BIORXIV_API_URL, pdfs_folder: Optional[str] = None):
        self.api_url = api_url.rstrip("/")
        self.pdfs_folder = Path(pdfs_folder or Path(OUTPUT_FOLDER) / "pdfs")
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; biorxiv-agent/1.0)",
        })
        self.pdfs_folder.mkdir(parents=True, exist_ok=True)

    def _request_with_retry(self, endpoint: str, params: dict = None) -> Optional[requests.Response]:
        url = f"{self.api_url}{endpoint}"

        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.get(
                    url, params=params, timeout=REQUEST_TIMEOUT,
                    headers={"Accept": "application/json"},
                )
            except requests.RequestException as e:
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"BioRxiv API error after {MAX_RETRIES} attempts: {e}")
                    return None
                wait_time = BACKOFF_FACTOR ** attempt * 2
                logger.warning(f"Request failed: {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue

            status = response.status_code
            logger.debug(f"Response status: {status}")

            # Rate limiting
            if status == 429:
                retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                logger.warning(f"Rate limited (429), waiting {retry_after}s (attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(retry_after)
                continue

            # Server errors (5xx) - retry with longer backoff
            if 500 <= status < 600:
                logger.warning(f"Server error HTTP {status}: {response.text[:200]}")
                if attempt < MAX_RETRIES - 1:
                    wait = BACKOFF_FACTOR ** (attempt + 2) * 5  # Longer wait: 20s, 40s, 80s...
                    logger.info(f"Waiting {wait}s before retry...")
                    time.sleep(wait)
                    continue
                return None

            # Other client errors are not retryable
            if status >= 400:
                logger.error(f"HTTP {status} for {url}: {response.text[:200]}")
                return None

            # Empty response body - treat as retryable
            if not response.text or not response.text.strip():
                logger.warning(f"Empty response body (status {status}), attempt {attempt+1}/{MAX_RETRIES}")
                if attempt < MAX_RETRIES - 1:
                    wait = BACKOFF_FACTOR ** attempt * 2
                    logger.info(f"Waiting {wait}s before retry...")
                    time.sleep(wait)
                    continue
                return None

            return response
        return None

    def _get_date_range(self, days_back: int) -> tuple:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    def get_recent_papers(self, days: int = 1, server: str = "biorxiv") -> List[Dict[str, Any]]:
        start_date, end_date = self._get_date_range(days)
        interval = f"{start_date}/{end_date}"
        all_papers = []
        cursor = 0

        while True:
            endpoint = f"/details/{server}/{interval}/{cursor}/json"
            response = self._request_with_retry(endpoint)
            if not response:
                break

            content_type = response.headers.get("Content-Type", "")
            if "application/json" not in content_type:
                logger.warning(f"Unexpected Content-Type: {content_type}, response: {response.text[:200]}")
                break

            try:
                data = response.json()
            except JSONDecodeError as e:
                logger.error(f"API returned invalid JSON (status {response.status_code}): {e}")
                logger.debug(f"Response text: {response.text[:500]}")
                break

            messages = data.get("messages", [])
            if not messages or messages[0].get("status") != "ok":
                logger.warning(f"API status not ok: {messages[0] if messages else 'no messages'}")
                break

            papers = data.get("collection", [])
            if not papers:
                break

            all_papers.extend(papers)
            total = int(messages[0].get("total", 0))
            count = int(messages[0].get("count", 0))
            if count <= 0:
                break
            cursor += count
            if cursor >= total:
                break

        for paper in all_papers:
            paper.setdefault("server", server)
        return dedupe_latest_versions(all_papers)

    def download_pdf(self, doi: str, version: int = 1, server: str = "biorxiv") -> Optional[bytes]:
        host = PDF_HOSTS.get(server, PDF_HOSTS["biorxiv"])
        pdf_url = f"{host}/content/{doi}v{version}.full.pdf"
        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.get(pdf_url, timeout=REQUEST_TIMEOUT)
            except requests.RequestException as e:
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"PDF download error after {MAX_RETRIES} attempts: {e}")
                    return None
                wait_time = BACKOFF_FACTOR ** attempt * 2
                logger.warning(f"PDF download error, retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue

            if response.status_code == 429:
                retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                logger.warning(f"PDF rate limited, waiting {retry_after}s (attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(retry_after)
                continue
            if response.status_code != 200:
                logger.warning(f"PDF download failed: HTTP {response.status_code} ({pdf_url})")
                return None
            # bioRxiv sometimes serves an HTML bot-check page with status 200
            if not response.content.startswith(b"%PDF"):
                logger.warning(
                    f"PDF download returned non-PDF content "
                    f"({response.headers.get('Content-Type', 'unknown')}) for {pdf_url}"
                )
                return None
            return response.content
        return None

    def save_pdf(self, doi: str, pdf_bytes: bytes) -> Optional[str]:
        safe_doi = doi.replace("/", "_").replace(".", "_")
        filepath = self.pdfs_folder / f"{safe_doi}.pdf"
        try:
            filepath.write_bytes(pdf_bytes)
            return str(filepath)
        except OSError as e:
            logger.error(f"Failed to save PDF: {e}")
            return None

    def check_connection(self) -> bool:
        start_date, end_date = self._get_date_range(1)
        interval = f"{start_date}/{end_date}"
        response = self._request_with_retry(f"/details/biorxiv/{interval}/0/json")
        return response is not None
