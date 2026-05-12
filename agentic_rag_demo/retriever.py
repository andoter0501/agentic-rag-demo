from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from .loader import Chunk


_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_\u4e00-\u9fff]+")


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    retrieval_type: str = "hybrid"


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_PATTERN.findall(text)]


class SparseRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.doc_tf: list[Counter[str]] = [Counter(tokenize(c.content)) for c in chunks]
        self.idf = self._build_idf(self.doc_tf)

    @staticmethod
    def _build_idf(doc_tf: list[Counter[str]]) -> dict[str, float]:
        n = len(doc_tf)
        df: Counter[str] = Counter()
        for tf in doc_tf:
            df.update(set(tf.keys()))
        return {term: math.log((1 + n) / (1 + freq)) + 1.0 for term, freq in df.items()}

    def _vectorize(self, tf: Counter[str]) -> dict[str, float]:
        vec: dict[str, float] = {}
        for term, freq in tf.items():
            if term not in self.idf:
                continue
            vec[term] = (1 + math.log(freq)) * self.idf[term]
        return vec

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(v * b.get(k, 0.0) for k, v in a.items())
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        q_vec = self._vectorize(Counter(tokenize(query)))
        scored: list[SearchResult] = []

        for chunk, tf in zip(self.chunks, self.doc_tf):
            score = self._cosine(q_vec, self._vectorize(tf))
            if score > 0:
                scored.append(
                    SearchResult(chunk=chunk, score=score, retrieval_type="sparse")
                )

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]


class DenseRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        index: Any | None = None,
    ) -> None:
        self.chunks = chunks
        self.embeddings = embeddings
        self.index = index
        self.chunk_id_to_idx: dict[str, int] = {
            c.chunk_id: i for i, c in enumerate(chunks)
        }
        self.idx_to_chunk_id: dict[int, str] = {
            i: c.chunk_id for i, c in enumerate(chunks)
        }

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        if self.index is not None:
            return self._search_index(query_embedding, top_k)
        return self._search_naive(query_embedding, top_k)

    def _search_naive(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[SearchResult]:
        if not self.embeddings:
            return []

        q_vec = _normalize(query_embedding)
        scores: list[tuple[int, float]] = []

        for idx, emb in enumerate(self.embeddings):
            emb_norm = _normalize(emb)
            score = sum(q * e for q, e in zip(q_vec, emb_norm))
            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results: list[SearchResult] = []
        for idx, score in scores[:top_k]:
            chunk = self.chunks[idx]
            results.append(
                SearchResult(chunk=chunk, score=score, retrieval_type="dense")
            )

        return results

    def _search_index(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[SearchResult]:
        from .vector_index import VectorIndex

        if not isinstance(self.index, VectorIndex):
            return self._search_naive(query_embedding, top_k)

        raw_results = self.index.search(query_embedding, top_k)

        results: list[SearchResult] = []
        for chunk_id, score in raw_results:
            if chunk_id in self.chunk_id_to_idx:
                idx = self.chunk_id_to_idx[chunk_id]
                results.append(
                    SearchResult(
                        chunk=self.chunks[idx], score=score, retrieval_type="dense"
                    )
                )

        return results


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        sparse: SparseRetriever | None = None,
        dense: DenseRetriever | None = None,
        sparse_weight: float = 0.4,
        dense_weight: float = 0.6,
        fusion_method: Literal["rrf", "weighted", "convex"] = "rrf",
    ) -> None:
        self.chunks = chunks
        self.sparse = sparse or SparseRetriever(chunks)
        self.dense = dense
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        self.fusion_method = fusion_method

    def search(
        self, query: str, query_embedding: list[float] | None = None, top_k: int = 5
    ) -> list[SearchResult]:
        sparse_results = self.sparse.search(query, top_k=top_k * 2)

        if self.dense and query_embedding:
            dense_results = self.dense.search(query_embedding, top_k=top_k * 2)
        else:
            dense_results = []

        if self.fusion_method == "rrf":
            return self._rrf_fusion(sparse_results, dense_results, top_k)
        elif self.fusion_method == "weighted":
            return self._weighted_fusion(sparse_results, dense_results, top_k)
        else:
            return self._convex_fusion(sparse_results, dense_results, top_k)

    def _rrf_fusion(
        self,
        sparse_results: list[SearchResult],
        dense_results: list[SearchResult],
        top_k: int,
        k: int = 60,
    ) -> list[SearchResult]:
        scores: dict[str, float] = {}

        for rank, result in enumerate(sparse_results, start=1):
            chunk_id = result.chunk.chunk_id
            scores[chunk_id] = (
                scores.get(chunk_id, 0.0) + (1.0 / (k + rank)) * self.sparse_weight
            )

        for rank, result in enumerate(dense_results, start=1):
            chunk_id = result.chunk.chunk_id
            scores[chunk_id] = (
                scores.get(chunk_id, 0.0) + (1.0 / (k + rank)) * self.dense_weight
            )

        chunk_id_to_result: dict[str, SearchResult] = {
            r.chunk.chunk_id: r for r in sparse_results + dense_results
        }

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results: list[SearchResult] = []
        for chunk_id, score in sorted_scores[:top_k]:
            if chunk_id in chunk_id_to_result:
                original = chunk_id_to_result[chunk_id]
                results.append(
                    SearchResult(
                        chunk=original.chunk,
                        score=score,
                        retrieval_type="hybrid_rrf",
                    )
                )

        return results

    def _weighted_fusion(
        self,
        sparse_results: list[SearchResult],
        dense_results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        max_sparse = max((r.score for r in sparse_results), default=1.0)
        max_dense = max((r.score for r in dense_results), default=1.0)

        scores: dict[str, float] = {}
        chunk_id_to_result: dict[str, SearchResult] = {}

        for result in sparse_results:
            chunk_id = result.chunk.chunk_id
            normalized = result.score / max_sparse if max_sparse > 0 else 0
            scores[chunk_id] = (
                scores.get(chunk_id, 0.0) + normalized * self.sparse_weight
            )
            chunk_id_to_result[chunk_id] = result

        for result in dense_results:
            chunk_id = result.chunk.chunk_id
            normalized = result.score / max_dense if max_dense > 0 else 0
            scores[chunk_id] = (
                scores.get(chunk_id, 0.0) + normalized * self.dense_weight
            )
            if chunk_id not in chunk_id_to_result:
                chunk_id_to_result[chunk_id] = result

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results: list[SearchResult] = []
        for chunk_id, score in sorted_scores[:top_k]:
            if chunk_id in chunk_id_to_result:
                original = chunk_id_to_result[chunk_id]
                results.append(
                    SearchResult(
                        chunk=original.chunk,
                        score=score,
                        retrieval_type="hybrid_weighted",
                    )
                )

        return results

    def _convex_fusion(
        self,
        sparse_results: list[SearchResult],
        dense_results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        all_results = sparse_results + dense_results

        sparse_max = max((r.score for r in sparse_results), default=1.0)
        dense_max = max((r.score for r in dense_results), default=1.0)

        convex_scores: dict[str, float] = {}
        chunk_id_to_result: dict[str, SearchResult] = {}

        for result in all_results:
            chunk_id = result.chunk.chunk_id
            chunk_id_to_result[chunk_id] = result

            if result.retrieval_type == "sparse":
                normalized = (result.score / sparse_max) if sparse_max > 0 else 0
                convex_scores[chunk_id] = (
                    convex_scores.get(chunk_id, 0.0) + self.sparse_weight * normalized
                )
            else:
                normalized = (result.score / dense_max) if dense_max > 0 else 0
                convex_scores[chunk_id] = (
                    convex_scores.get(chunk_id, 0.0) + self.dense_weight * normalized
                )

        sorted_scores = sorted(convex_scores.items(), key=lambda x: x[1], reverse=True)

        results: list[SearchResult] = []
        for chunk_id, score in sorted_scores[:top_k]:
            if chunk_id in chunk_id_to_result:
                original = chunk_id_to_result[chunk_id]
                results.append(
                    SearchResult(
                        chunk=original.chunk,
                        score=score,
                        retrieval_type="hybrid_convex",
                    )
                )

        return results
