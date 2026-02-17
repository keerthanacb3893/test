"""
Chunk document segments and build a BM25 retrieval index.

The index is persisted to disk as JSON under .doc_qa_index/ so it
survives between CLI invocations without re-parsing the document.

Chunking strategy:
  - Sliding window of CHUNK_WORDS words with OVERLAP_WORDS overlap
  - Each chunk inherits its source segment's page_num and section label
  - Overlap ensures answers that span chunk boundaries are still retrievable
"""

import json
import uuid
import hashlib
from pathlib import Path

CHUNK_WORDS = 250
OVERLAP_WORDS = 50
INDEX_DIR = Path(".doc_qa_index")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def index_document(filepath: str, segments: list) -> str:
    """
    Chunk all segments, write the index to disk, and return the doc_id.

    doc_id is a deterministic hash of the filepath so re-indexing the same
    file overwrites the previous index cleanly.
    """
    doc_id = _doc_id(filepath)
    chunks = _build_chunks(doc_id, segments)

    INDEX_DIR.mkdir(exist_ok=True)
    index_path = INDEX_DIR / f"{doc_id}_chunks.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(
            {"doc_id": doc_id, "filepath": str(filepath), "chunks": chunks},
            f,
            indent=2,
        )

    return doc_id


def load_index(doc_id: str) -> tuple:
    """
    Load an existing index and return (chunks_list, BM25_instance).

    BM25 is rebuilt in-memory from the stored chunks — fast enough for
    typical document sizes (<1s) and avoids pickle serialization issues.
    """
    from rank_bm25 import BM25Okapi

    path = INDEX_DIR / f"{doc_id}_chunks.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No index found for doc_id '{doc_id}'. "
            f"Run: python main.py index <filepath>"
        )

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    chunks = data["chunks"]
    tokenized = [_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    return chunks, bm25


def list_documents() -> list:
    """Return metadata for all indexed documents."""
    if not INDEX_DIR.exists():
        return []

    docs = []
    for path in sorted(INDEX_DIR.glob("*_chunks.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        docs.append(
            {
                "doc_id": data["doc_id"],
                "filepath": data["filepath"],
                "chunk_count": len(data["chunks"]),
            }
        )
    return docs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _doc_id(filepath: str) -> str:
    return hashlib.md5(str(filepath).encode()).hexdigest()[:12]


def _build_chunks(doc_id: str, segments: list) -> list:
    chunks = []
    for text, page_num, section in segments:
        words = text.split()
        start = 0
        while start < len(words):
            end = min(start + CHUNK_WORDS, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(
                {
                    "chunk_id": uuid.uuid4().hex[:8],
                    "doc_id": doc_id,
                    "text": chunk_text,
                    "page_num": page_num,
                    "section": section,
                    "word_start": start,
                }
            )
            if end == len(words):
                break
            start += CHUNK_WORDS - OVERLAP_WORDS
    return chunks


def _tokenize(text: str) -> list:
    return text.lower().split()
