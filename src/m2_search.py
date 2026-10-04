from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    if not text:
        return ""
    try:
        from underthesea import word_tokenize
        segmented = word_tokenize(text, format="text")
        return segmented.replace("_", " ")
    except Exception:
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi

        self.documents = chunks
        self.corpus_tokens = []
        for c in chunks:
            text = c["text"] if isinstance(c, dict) else c.text
            tokens = segment_vietnamese(text).lower().split()
            self.corpus_tokens.append(tokens)

        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or not self.documents:
            return []

        tokenized_query = segment_vietnamese(query).lower().split()
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        scored_indices = [
            (i, float(scores[i])) for i in range(len(scores)) if scores[i] > 0
        ]
        scored_indices.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i, score in scored_indices[:top_k]:
            doc = self.documents[i]
            text = doc["text"] if isinstance(doc, dict) else doc.text
            metadata = doc.get("metadata", {}) if isinstance(doc, dict) else getattr(doc, "metadata", {})
            results.append(SearchResult(text=text, score=score, metadata=metadata, method="bm25"))

        return results


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
            self.client.get_collections()
        except Exception:
            self.client = QdrantClient(":memory:")
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        if not chunks:
            return

        from qdrant_client.models import Distance, VectorParams, PointStruct

        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )

        texts = [c["text"] if isinstance(c, dict) else c.text for c in chunks]
        vectors = self._get_encoder().encode(texts, show_progress_bar=False)

        points = []
        for i, (c, v) in enumerate(zip(chunks, vectors)):
            text = c["text"] if isinstance(c, dict) else c.text
            meta = c.get("metadata", {}) if isinstance(c, dict) else getattr(c, "metadata", {})
            payload = {**meta, "text": text}
            vec = v.tolist() if hasattr(v, "tolist") else list(v)
            points.append(PointStruct(id=i, vector=vec, payload=payload))

        self.client.upsert(collection_name=collection, points=points)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if not query or not query.strip():
            return []

        try:
            if not self.client.collection_exists(collection):
                return []
        except Exception:
            pass

        encoder = self._get_encoder()
        query_vec = encoder.encode(query)
        query_vector = query_vec.tolist() if hasattr(query_vec, "tolist") else list(query_vec)

        response = self.client.query_points(collection_name=collection, query=query_vector, limit=top_k)
        results = []
        for pt in response.points:
            payload = pt.payload or {}
            text = payload.get("text", "")
            meta = {k: v for k, v in payload.items() if k != "text"}
            results.append(SearchResult(text=text, score=float(pt.score), metadata=meta, method="dense"))
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank + 1)."""
    if not results_list:
        return []

    rrf_scores: dict[str, dict] = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {
                    "score": 0.0,
                    "metadata": result.metadata,
                }
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)

    sorted_items = sorted(rrf_scores.items(), key=lambda item: item[1]["score"], reverse=True)
    fused_results = []
    for text, info in sorted_items[:top_k]:
        fused_results.append(SearchResult(
            text=text,
            score=info["score"],
            metadata=info["metadata"],
            method="hybrid"
        ))
    return fused_results


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
