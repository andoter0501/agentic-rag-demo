from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .loader import Chunk


_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_\u4e00-\u9fff]+")


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


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
                scored.append(SearchResult(chunk=chunk, score=score))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]
