from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    page_num: Optional[int]
    section: Optional[str]
    word_start: int


@dataclass
class Citation:
    chunk_id: str
    page_num: Optional[int]
    section: Optional[str]
    snippet: str  # 1–3 line quote from the passage


@dataclass
class QAResult:
    question: str
    doc_id: str
    answer: str
    found: bool                    # False = document didn't contain the answer
    citations: list = field(default_factory=list)       # list[Citation]
    retrieved_chunk_ids: list = field(default_factory=list)
    fallback: bool = False         # mirrors (not found)
