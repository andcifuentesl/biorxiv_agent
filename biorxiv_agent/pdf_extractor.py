import pymupdf as fitz
from typing import Optional


def extract_text(pdf_bytes: bytes) -> Optional[str]:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except fitz.FileDataError as e:
        print(f"PDF extraction failed (corrupted/encrypted): {e}")
        return None
    except Exception as e:
        print(f"PDF extraction failed: {e}")
        return None

    try:
        text_parts = []
        for page in doc:
            text = page.get_text()
            if text.strip():
                text_parts.append(text.strip())
        doc.close()

        full_text = "\n\n".join(text_parts)
        return _clean_text(full_text)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None


def extract_first_n_pages(pdf_bytes: bytes, n: int = 3) -> Optional[str]:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except fitz.FileDataError as e:
        print(f"PDF extraction failed (corrupted/encrypted): {e}")
        return None
    except Exception as e:
        print(f"PDF extraction failed: {e}")
        return None

    try:
        text_parts = []
        for i, page in enumerate(doc):
            if i >= n:
                break
            text = page.get_text()
            if text.strip():
                text_parts.append(text.strip())
        doc.close()

        full_text = "\n\n".join(text_parts)
        return _clean_text(full_text)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None


def _clean_text(text: str) -> str:
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line:
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = " ".join(text.split())

    max_chars = 15000
    if len(text) > max_chars:
        text = text[:max_chars] + "... [truncated]"

    return text