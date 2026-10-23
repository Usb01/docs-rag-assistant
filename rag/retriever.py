from dataclasses import dataclass

import numpy as np

from .bm25 import BM25
from .store import Store, StoredChunk


@dataclass
class Hit:
    chunk: StoredChunk
    score: float          # fused score used for ranking
    keyword: float        # raw BM25
    semantic: float | None
    coverage: float = 1.0  # idf-weighted share of query terms present in this chunk


class Retriever:
    """Hybrid search: BM25 always, plus embeddings when an embedder is configured.

    The two rankings are merged with reciprocal rank fusion, which avoids
    having to put BM25 scores and cosine similarities on the same scale.
    """

    def __init__(self, store: Store, embedder=None, rrf_k: int = 60):
        self.store = store
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.refresh()

    def refresh(self) -> None:
        self.chunks = self.store.all_chunks()
        self.bm25 = BM25([c.indexed_text for c in self.chunks]) if self.chunks else None
        have_vectors = self.chunks and all(c.embedding is not None for c in self.chunks)
        self.matrix = np.vstack([c.embedding for c in self.chunks]) if have_vectors else None

    def search(self, query: str, k: int = 4) -> list[Hit]:
        if not self.chunks:
            return []
        kw = np.asarray(self.bm25.scores(query))
        fused = np.zeros(len(self.chunks))
        for rank, idx in enumerate(np.argsort(-kw)):
            if kw[idx] > 0:
                fused[idx] += 1 / (self.rrf_k + rank)

        sem = None
        if self.matrix is not None and self.embedder is not None:
            q = self.embedder.embed([query])[0]
            sem = self.matrix @ q
            for rank, idx in enumerate(np.argsort(-sem)[:20]):
                fused[idx] += 1 / (self.rrf_k + rank)

        order = [i for i in np.argsort(-fused) if fused[i] > 0][:k]
        return [Hit(self.chunks[i], float(fused[i]), float(kw[i]), float(sem[i]) if sem is not None else None,
                    self.bm25.coverage(query, i))
                for i in order]
