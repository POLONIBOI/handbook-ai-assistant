"""Part 2 (retrieval): embed the question and pull the closest chunks from ChromaDB."""

from dataclasses import dataclass

from app import config


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    page: int
    section: str
    similarity: float  # cosine similarity, 1.0 = identical meaning
    rerank_score: float | None = None  # filled in by app/reranker.py


class Retriever:
    def __init__(self, collection=None, embed=None, top_k: int = config.TOP_K, rerank=None):
        if collection is None:
            from app.vector_store import get_collection

            collection = get_collection()
        if embed is None:
            from app.embeddings import embed_query as embed
        if rerank is None:
            # rerank=False keeps plain similarity order (used by the eval script)
            from app.reranker import rerank as default_rerank

            rerank = default_rerank if config.RERANK_MODEL else False

        self._collection = collection
        self._embed = embed
        self._rerank = rerank
        self.top_k = top_k

    def search(self, question: str) -> list[RetrievedChunk]:
        """Top-k chunks for the question, best match first."""
        result = self._collection.query(
            query_embeddings=[self._embed(question)],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )
        hits = [
            RetrievedChunk(
                text=doc,
                page=int(meta["page"]),
                section=meta["section"],
                # Collection uses cosine distance, which is 1 - similarity.
                similarity=round(1 - distance, 3),
            )
            for doc, meta, distance in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]
        return self._rerank(question, hits) if self._rerank else hits
