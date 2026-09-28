"""Second pass over the retrieved chunks: a cross-encoder puts the best ones first.

The embedding search compares a question vector with chunk vectors that were
made separately. A cross-encoder reads the question and a chunk together, so
it's slower (it runs once per chunk) but better at telling which of the
candidates actually answers the question.

Claude reads all 5 excerpts, so the order barely matters to it. The local
model only reads the top 3, and it's only asked at all when the reranker
thinks one of them fits (see app/local_llm.py).
"""

from dataclasses import replace
from functools import lru_cache

from app import config


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.RERANK_MODEL)


def rerank(question: str, chunks: list) -> list:
    """The same chunks, best first, each with its rerank score filled in."""
    if not chunks:
        return []
    pairs = [(question, f"{c.section}: {c.text}") for c in chunks]
    scores = _model().predict(pairs, show_progress_bar=False)
    scored = [replace(c, rerank_score=round(float(s), 2)) for c, s in zip(chunks, scores)]
    return sorted(scored, key=lambda c: c.rerank_score, reverse=True)
