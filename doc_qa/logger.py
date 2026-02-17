"""
Interaction logger — SQLite-backed audit trail.

Stores every Q&A interaction for debugging and compliance (AC: auditability).
Schema: document ID, question, retrieved passage IDs, answer text, timestamp, found flag.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_DB_DIR = Path(".doc_qa_index")
_DB_PATH = _DB_DIR / "interactions.db"


def _connect() -> sqlite3.Connection:
    _DB_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS interactions (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp           TEXT    NOT NULL,
            doc_id              TEXT    NOT NULL,
            question            TEXT    NOT NULL,
            retrieved_chunk_ids TEXT    NOT NULL,
            answer              TEXT    NOT NULL,
            found               INTEGER NOT NULL,
            fallback            INTEGER NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def log_interaction(result: dict) -> None:
    """Persist a Q&A result to the audit log."""
    conn = _connect()
    conn.execute(
        """
        INSERT INTO interactions
          (timestamp, doc_id, question, retrieved_chunk_ids, answer, found, fallback)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            result["doc_id"],
            result["question"],
            json.dumps(result.get("retrieved_chunk_ids", [])),
            result["answer"],
            int(result.get("found", False)),
            int(result.get("fallback", True)),
        ),
    )
    conn.commit()
    conn.close()


def get_history(doc_id: str = None, limit: int = 20) -> list:
    """Return recent interactions, optionally filtered by doc_id."""
    conn = _connect()
    if doc_id:
        rows = conn.execute(
            "SELECT * FROM interactions WHERE doc_id = ? ORDER BY timestamp DESC LIMIT ?",
            (doc_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM interactions ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()

    cols = [
        "id", "timestamp", "doc_id", "question",
        "retrieved_chunk_ids", "answer", "found", "fallback",
    ]
    return [dict(zip(cols, row)) for row in rows]
