import sqlite3
import logging
from contextlib import closing
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from .config import OUTPUT_FOLDER, ABSTRACT_MAX_CHARS


logger = logging.getLogger(__name__)

# Columns added after the original schema; created on startup if missing
MIGRATIONS = {
    "eval_source": "TEXT",
    "server": "TEXT",
    "version": "INTEGER",
}


class DatabaseManager:
    """SQLite store; also the source of truth for which papers were already processed."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(Path(OUTPUT_FOLDER) / "papers.db")
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with closing(self._get_conn()) as conn, conn:
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
            existing = {row["name"] for row in conn.execute("PRAGMA table_info(papers)")}
            for column, col_type in MIGRATIONS.items():
                if column not in existing:
                    conn.execute(f"ALTER TABLE papers ADD COLUMN {column} {col_type}")
            # Rows from before eval_source existed were evaluated by the LLM unless marked as fallback
            conn.execute("""
                UPDATE papers SET eval_source = CASE
                    WHEN agent_reasoning LIKE '%(fallback mode)' THEN 'fallback' ELSE 'llm' END
                WHERE eval_source IS NULL
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_processed_at ON papers(processed_at)")

    def is_processed(self, doi: str) -> bool:
        """A paper is done once the LLM has evaluated it; fallback evaluations get retried."""
        with closing(self._get_conn()) as conn:
            row = conn.execute(
                "SELECT 1 FROM papers WHERE doi = ? AND eval_source = 'llm'", (doi,)
            ).fetchone()
        return row is not None

    def has_pdf(self, doi: str) -> bool:
        with closing(self._get_conn()) as conn:
            row = conn.execute(
                "SELECT 1 FROM papers WHERE doi = ? AND pdf_downloaded = 1", (doi,)
            ).fetchone()
        return row is not None

    def save_evaluation(self, paper: Dict, eval_data: Dict) -> None:
        """Insert or update a paper's metadata and agent evaluation, keeping any download/classification."""
        doi = paper.get("doi")
        with closing(self._get_conn()) as conn, conn:
            conn.execute("""
                INSERT INTO papers (
                    doi, title, authors, date, category, abstract, server, version,
                    relevance_score, novelty_score, rigor_score,
                    agent_recommendation, agent_reasoning, eval_source, processed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doi) DO UPDATE SET
                    title = excluded.title,
                    authors = excluded.authors,
                    date = excluded.date,
                    category = excluded.category,
                    abstract = excluded.abstract,
                    server = excluded.server,
                    version = excluded.version,
                    relevance_score = excluded.relevance_score,
                    novelty_score = excluded.novelty_score,
                    rigor_score = excluded.rigor_score,
                    agent_recommendation = excluded.agent_recommendation,
                    agent_reasoning = excluded.agent_reasoning,
                    eval_source = excluded.eval_source,
                    processed_at = excluded.processed_at
            """, (
                doi,
                paper.get("title"),
                paper.get("authors"),
                paper.get("date"),
                paper.get("category"),
                (paper.get("abstract") or "")[:ABSTRACT_MAX_CHARS],
                paper.get("server"),
                paper.get("version"),
                eval_data.get("relevance_score"),
                eval_data.get("novelty_score"),
                eval_data.get("rigor_score"),
                eval_data.get("recommendation"),
                eval_data.get("reasoning"),
                eval_data.get("source", "llm"),
                datetime.now().isoformat(),
            ))
        logger.debug(f"Saved evaluation for {doi}")

    def save_result(self, doi: str, classification: Dict[str, Any], pdf_path: Optional[str]) -> None:
        """Record classification and download outcome for an already-evaluated paper."""
        with closing(self._get_conn()) as conn, conn:
            conn.execute("""
                UPDATE papers SET
                    classification = ?,
                    classification_confidence = ?,
                    classification_reasoning = ?,
                    pdf_downloaded = ?,
                    pdf_path = ?
                WHERE doi = ?
            """, (
                classification.get("classification"),
                classification.get("confidence"),
                classification.get("reasoning"),
                1 if pdf_path else 0,
                pdf_path,
                doi,
            ))
        logger.debug(f"Saved result for {doi}")

    def _query(self, sql: str, params: tuple = ()) -> List[Dict]:
        with closing(self._get_conn()) as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def get_all_papers(self) -> List[Dict]:
        return self._query("SELECT * FROM papers ORDER BY processed_at DESC")

    def get_papers_by_recommendation(self, recommendation: str) -> List[Dict]:
        return self._query(
            "SELECT * FROM papers WHERE agent_recommendation = ? ORDER BY processed_at DESC",
            (recommendation,),
        )

    def get_downloaded_papers(self) -> List[Dict]:
        return self._query("SELECT * FROM papers WHERE pdf_downloaded = 1 ORDER BY processed_at DESC")

    def count(self) -> int:
        return self._query("SELECT COUNT(*) AS c FROM papers")[0]["c"]
