import time
import requests
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta
from requests.exceptions import JSONDecodeError

from .config import (
    BIORXIV_API_URL,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    BACKOFF_FACTOR,
    PDFS_FOLDER,
)


logger = logging.getLogger(__name__)


class BioRxivClient:
    def __init__(self, api_url: str = BIORXIV_API_URL):
        self.api_url = api_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; biorxiv-agent/1.0)",
            "Accept": "application/json",
        })
        Path(PDFS_FOLDER).mkdir(parents=True, exist_ok=True)

    def _request_with_retry(self, endpoint: str, params: dict = None) -> Optional[requests.Response]:
        url = f"{self.api_url}{endpoint}"
        
        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
                
                # Rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 10))
                    if retry_after < 1:
                        retry_after = 10
                    logger.warning(f"Rate limited (429), waiting {retry_after}s (attempt {attempt+1}/{MAX_RETRIES})")
                    time.sleep(retry_after)
                    continue
                
                # Server errors (5xx) - retry with longer backoff
                status = response.status_code
                logger.debug(f"Response status: {status}")
                if 500 <= status < 600:
                    logger.warning(f"Server error HTTP {status}: {response.text[:200]}")
                    if attempt < MAX_RETRIES - 1:
                        wait = BACKOFF_FACTOR ** (attempt + 2) * 5  # Longer wait: 20s, 40s, 80s...
                        logger.info(f"Waiting {wait}s before retry...")
                        time.sleep(wait)
                        continue
                    return None
                
                # Other HTTP errors
                if response.status_code >= 400:
                    logger.warning(f"HTTP {response.status_code}: {response.text[:200]}")
                    response.raise_for_status()
                
                # Empty response body - treat as retryable
                if not response.text or not response.text.strip():
                    logger.warning(f"Empty response body (status {response.status_code}), attempt {attempt+1}/{MAX_RETRIES}")
                    if attempt < MAX_RETRIES - 1:
                        wait = BACKOFF_FACTOR ** attempt * 2  # Longer wait for empty responses
                        logger.info(f"Waiting {wait}s before retry...")
                        time.sleep(wait)
                        continue
                    return None
                
                return response
                
            except requests.RequestException as e:
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"BioRxiv API error after {MAX_RETRIES} attempts: {e}")
                    return None
                wait_time = BACKOFF_FACTOR ** attempt * 2
                logger.warning(f"Request failed: {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)
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
            
            # Check content type
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
            cursor += count
            if cursor >= total:
                break
                
        return all_papers

    def download_pdf(self, doi: str, version: int = 1) -> Optional[bytes]:
        pdf_url = f"https://www.biorxiv.org/content/{doi}v{version}.full.pdf"
        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.get(pdf_url, timeout=REQUEST_TIMEOUT)
                if response.status_code == 200:
                    return response.content
                elif response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 10))
                    if retry_after < 1:
                        retry_after = 10
                    logger.warning(f"PDF rate limited, waiting {retry_after}s (attempt {attempt+1}/{MAX_RETRIES})")
                    time.sleep(retry_after)
                    continue
                else:
                    logger.warning(f"PDF download failed: HTTP {response.status_code}")
                    return None
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"PDF download error after {MAX_RETRIES} attempts: {e}")
                    return None
                wait_time = BACKOFF_FACTOR ** attempt * 2
                logger.warning(f"PDF download error, retrying in {wait_time}s...")
                time.sleep(wait_time)
        return None

    def save_pdf(self, doi: str, pdf_bytes: bytes) -> Optional[str]:
        safe_doi = doi.replace("/", "_").replace(".", "_")
        filename = f"{safe_doi}.pdf"
        filepath = Path(PDFS_FOLDER) / filename
        try:
            with open(filepath, "wb") as f:
                f.write(pdf_bytes)
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save PDF: {e}")
            return None

    def check_connection(self) -> bool:
        start_date, end_date = self._get_date_range(1)
        interval = f"{start_date}/{end_date}"
        response = self._request_with_retry(f"/details/biorxiv/{interval}/0/json")
        return response is not None
