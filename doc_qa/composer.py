"""
Answer composer — the GenAI layer.

Takes the user's question and the retrieved chunks, calls Claude with a
strict grounding prompt, and returns a structured answer with citations.

Key guarantee: the model is instructed never to answer from memory.
If the passages don't contain the answer, it returns a grounded refusal
with the checked passages shown (AC: fallback + show retrieved snippets).
"""

import json
import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

_SYSTEM = """You are a Document Q&A assistant. You answer questions ONLY using the passages provided.

Rules — follow these without exception:
1. Use ONLY the provided passages. Never use outside knowledge or training data.
2. If the passages do not contain enough information to answer the question, set "found": false
   and answer with exactly: "I can't find this in the document."
3. Every claim in your answer must be traceable to at least one passage via its chunk_id.
4. Citations must include a short verbatim snippet (1–3 lines) from the passage.
5. Do not speculate, extrapolate, or fill gaps with general knowledge.

Return ONLY valid JSON — no preamble, no explanation, no markdown fences:
{
  "found": true | false,
  "answer": "Your concise factual answer, or 'I can\\'t find this in the document.' if not found.",
  "citations": [
    {
      "chunk_id": "the chunk_id of the passage you used",
      "page_num": null or integer,
      "section": "section label or null",
      "snippet": "exact 1-3 line quote from the passage supporting this claim"
    }
  ]
}

When found is false, still populate citations with the top passages that were checked,
so the user can verify what the system looked at."""


def compose_answer(question: str, chunks: list, doc_id: str) -> dict:
    """
    Call Claude with the question and retrieved chunks.
    Returns a dict suitable for logging and display:
      question, doc_id, found, answer, citations, retrieved_chunk_ids, fallback
    """
    user_msg = _build_user_message(question, chunks)

    response = _client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = response.content[0].text.strip()
    data = _parse_json(raw)

    return {
        "question": question,
        "doc_id": doc_id,
        "found": data.get("found", False),
        "answer": data.get("answer", ""),
        "citations": data.get("citations", []),
        "retrieved_chunk_ids": [c["chunk_id"] for c in chunks],
        "fallback": not data.get("found", True),
    }


def _build_user_message(question: str, chunks: list) -> str:
    passages = []
    for i, chunk in enumerate(chunks, start=1):
        loc_parts = []
        if chunk.get("page_num"):
            loc_parts.append(f"Page {chunk['page_num']}")
        if chunk.get("section"):
            loc_parts.append(f"Section: {chunk['section']}")
        loc = " | ".join(loc_parts) or "Location unknown"

        passages.append(
            f"[Passage {i} | chunk_id: {chunk['chunk_id']} | {loc}]\n{chunk['text']}"
        )

    passages_block = "\n\n".join(passages)

    return (
        f"Question: {question}\n\n"
        f"---\n\n"
        f"Passages retrieved from the document:\n\n"
        f"{passages_block}\n\n"
        f"---\n\n"
        f"Answer the question using ONLY these passages. Return valid JSON."
    )


def _parse_json(text: str) -> dict:
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    return json.loads(text.strip())
