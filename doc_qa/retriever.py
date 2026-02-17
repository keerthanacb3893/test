"""
BM25 retrieval: maps a natural-language question to the top-k most
relevant chunks from the indexed document.

BM25 (Best Match 25) scores chunks based on term frequency and inverse
document frequency — a strong baseline for keyword-rich document retrieval
that requires no embeddings or external APIs.
"""

from doc_qa.indexer import load_index, _tokenize

TOP_K = 6  # chunks passed to the model; more = more context but higher cost


def retrieve(doc_id: str, question: str, top_k: int = TOP_K) -> list:
    """
    Return up to top_k chunks most relevant to the question, each as a dict:
      chunk_id, doc_id, text, page_num, section, word_start, score
    """
    chunks, bm25 = load_index(doc_id)
    scores = bm25.get_scores(_tokenize(question))

    scored = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

    results = []
    for chunk, score in scored[:top_k]:
        results.append({**chunk, "score": round(float(score), 4)})

    return results
