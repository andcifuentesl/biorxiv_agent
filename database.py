import sqlite3
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from config import OUTPUT_FOLDER


class DatabaseManager:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(Path(OUTPUT_FOLDER) / "papers.db")
        self.db_path = db_path
        Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doi TEXT UNIQUE NOT NULL,
                title TEXT,
                authors TEXT,
                date TEXT,
                category TEXT,
                abstract TEXT,
                relevance_score INTEGER,
                novelty_score INTEGER,
                rigor_score INTEGER,
                agent_recommendation TEXT,
                agent_reasoning TEXT,
                classification TEXT,
                classification_confidence REAL,
                classification_reasoning TEXT,
                pdf_downloaded INTEGER DEFAULT 0,
                pdf_path TEXT,
                processed_at TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_papers_processed_at ON papers(processed_at)
        """)
        conn.commit()
        conn.close()

    def insert_paper(self, paper: Dict, eval_data: Dict, result: Optional[Dict]) -> None:
        doi = paper.get("doi")
        try:
            conn = self._get_conn()
            conn.execute("""
                INSERT OR REPLACE INTO papers (
                    doi, title, authors, date, category, abstract,
                    relevance_score, novelty_score, rigor_score,
                    agent_recommendation, agent_reasoning,
                    classification, classification_confidence, classification_reasoning,
                    pdf_downloaded, pdf_path, processed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doi,
                paper.get("title"),
                paper.get("authors"),
                paper.get("date"),
                paper.get("category"),
                paper.get("abstract", "")[:2000],
                eval_data.get("relevance_score"),
                eval_data.get("novelty_score"),
                eval_data.get("rigor_score"),
                eval_data.get("recommendation"),
                eval_data.get("reasoning"),
                result.get("classification") if result else None,
                result.get("confidence") if result else None,
                result.get("reasoning") if result else None,
                1 if result and result.get("pdf_path") else 0,
                result.get("pdf_path") if result else None,
                datetime.now().isoformat(),
            ))
            conn.commit()
            conn.close()
            import logging
            logging.getLogger(__name__).debug(f"Inserted paper {doi}")
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Failed to insert paper {doi}: {e}")
            raise

    def get_all_papers(self) -> List[Dict]:
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM papers ORDER BY processed_at DESC").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_papers_by_recommendation(self, recommendation: str) -> List[Dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM papers WHERE agent_recommendation = ? ORDER BY processed_at DESC",
            (recommendation,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_downloaded_papers(self) -> List[Dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM papers WHERE pdf_downloaded = 1 ORDER BY processed_at DESC"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def count(self) -> int:
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) as c FROM papers").fetchone()
        conn.close()
        return row["c"]