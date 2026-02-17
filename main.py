#!/usr/bin/env python3
"""
Document Q&A CLI

Commands:
  python main.py index <filepath>              Index a document (PDF, DOCX, TXT, MD)
  python main.py ask <doc_id> "<question>"     Ask a question grounded in that document
  python main.py list                          List all indexed documents
  python main.py history [doc_id]              Show recent interaction history
"""

import sys
import textwrap


def _divider(char="─", width=72):
    print(char * width)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_index(filepath: str) -> None:
    from doc_qa.extractor import extract
    from doc_qa.indexer import index_document

    import os
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    print(f"Extracting text from '{filepath}'...")
    try:
        segments = extract(filepath)
    except Exception as e:
        print(f"Error: Extraction failed — {e}")
        sys.exit(1)

    total_words = sum(len(s[0].split()) for s in segments)
    print(f"  Extracted {len(segments)} segment(s), ~{total_words:,} words.")

    print("Building index...")
    doc_id = index_document(filepath, segments)

    print(f"\n  Done.")
    print(f"  doc_id : {doc_id}")
    print(f"  To ask a question:")
    print(f'    python main.py ask {doc_id} "Your question here"')


def cmd_ask(doc_id: str, question: str) -> None:
    from doc_qa.retriever import retrieve
    from doc_qa.composer import compose_answer
    from doc_qa.logger import log_interaction

    print(f"Retrieving relevant passages for: \"{question}\"")
    try:
        chunks = retrieve(doc_id, question)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not chunks:
        print("No passages retrieved — is the doc_id correct?")
        sys.exit(1)

    print(f"  Retrieved {len(chunks)} passage(s). Generating grounded answer...")
    result = compose_answer(question, chunks, doc_id)
    log_interaction(result)

    _divider("═")
    print(f"Q: {question}")
    _divider()

    if result["fallback"]:
        print("A: I can't find this in the document.\n")
        print("Passages that were checked (top 3):")
        for chunk in chunks[:3]:
            loc = chunk.get("section") or (f"Page {chunk['page_num']}" if chunk.get("page_num") else "")
            print(f"\n  [{loc}]")
            print(textwrap.fill(chunk["text"][:300] + "...", width=70, initial_indent="  "))
    else:
        print(f"A: {result['answer']}\n")

        if result.get("citations"):
            print("Citations:")
            for i, cit in enumerate(result["citations"], 1):
                parts = []
                if cit.get("page_num"):
                    parts.append(f"Page {cit['page_num']}")
                if cit.get("section"):
                    parts.append(cit["section"])
                loc = " | ".join(parts) if parts else "Document"
                print(f"\n  [{i}] {loc}  (chunk: {cit.get('chunk_id', '?')})")
                if cit.get("snippet"):
                    print(textwrap.fill(f'  "{cit["snippet"]}"', width=70, initial_indent="  "))

    _divider("═")
    print("Interaction logged for audit.")


def cmd_list() -> None:
    from doc_qa.indexer import list_documents

    docs = list_documents()
    if not docs:
        print("No documents indexed yet.")
        print("Run: python main.py index <filepath>")
        return

    print(f"Indexed documents ({len(docs)}):")
    _divider()
    for d in docs:
        print(f"  doc_id : {d['doc_id']}")
        print(f"  file   : {d['filepath']}")
        print(f"  chunks : {d['chunk_count']}")
        print()


def cmd_history(doc_id: str = None) -> None:
    from doc_qa.logger import get_history

    rows = get_history(doc_id=doc_id, limit=10)
    if not rows:
        print("No interactions recorded yet.")
        return

    scope = f"for doc {doc_id}" if doc_id else "across all documents"
    print(f"Recent interactions ({scope}, latest {len(rows)}):")
    _divider()
    for r in rows:
        status = "found" if r["found"] else "not found"
        print(f"  [{r['timestamp'][:19]}] [{status}]")
        print(f"  doc : {r['doc_id']}")
        print(f"  Q   : {r['question'][:80]}")
        print(f"  A   : {r['answer'][:120]}")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:]

    if not args:
        print(__doc__)
        sys.exit(0)

    cmd = args[0]

    if cmd == "index" and len(args) == 2:
        cmd_index(args[1])
    elif cmd == "ask" and len(args) == 3:
        cmd_ask(args[1], args[2])
    elif cmd == "list":
        cmd_list()
    elif cmd == "history":
        cmd_history(args[1] if len(args) > 1 else None)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
