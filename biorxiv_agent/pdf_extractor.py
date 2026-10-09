import logging
import pymupdf as fitz
from typing import Optional


logger = logging.getLogger(__name__)

MAX_CHARS = 15000


def extract_text(pdf_bytes: bytes) -> Optional[str]:
    return extract_first_n_pages(pdf_bytes, n=None)


def extract_first_n_pages(pdf_bytes: bytes, n: Optional[int] = 3) -> Optional[str]:
    """Extract cleaned text from the first n pages (all pages if n is None)."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except fitz.FileDataError as e:
        logger.warning(f"PDF extraction failed (corrupted/encrypted): {e}")
        return None
    except Exception as e:
        logger.warning(f"PDF extraction failed: {e}")
        return None

    try:
        with doc:
            text_parts = []
            for i, page in enumerate(doc):
                if n is not None and i >= n:
                    break
                text = page.get_text()
                if text.strip():
                    text_parts.append(text.strip())
        return _clean_text("\n\n".join(text_parts))
    except Exception as e:
        logger.warning(f"Error extracting text from PDF: {e}")
        return None


def _clean_text(text: str) -> str:
    text = " ".join(text.split())
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "... [truncated]"
    return text
