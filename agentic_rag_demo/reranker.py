from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .loader import Chunk
from .retriever import SearchResult


@dataclass(frozen=True)
class RerankResult:
    chunk: Chunk
    original_score: float
    rerank_score: float
    rerank_type: str = "cross_encoder"


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[RerankResult]:
        raise NotImplementedError


class CrossEncoderReranker(Reranker):
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise RuntimeError("请安装 sentence-transformers: pip install sentence-transformers")

        self.model = CrossEncoder(model_name, max_length=512, device=device)
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[RerankResult]:
        if not results:
            return []

        pairs = [(query, r.chunk.content) for r in results]
        scores = self.model.predict(pairs, show_progress_bar=False)

        if not hasattr(scores, "__iter__"):
            scores = [float(scores)]

        scored_results: list[tuple[SearchResult, float]] = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        reranked: list[RerankResult] = []
        for result, score in scored_results[:top_k]:
            reranked.append(
                RerankResult(
                    chunk=result.chunk,
                    original_score=result.score,
                    rerank_score=float(score),
                    rerank_type="cross_encoder",
                )
            )

        return reranked


class LLMCoherenceReranker(Reranker):
    def __init__(
        self,
        llm: Any | None = None,
        use_api: bool = False,
    ) -> None:
        self.llm = llm
        self.use_api = use_api

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[RerankResult]:
        if not results:
            return []

        if self.llm and self.use_api:
            return self._rerank_with_llm(query, results, top_k)

        return self._heuristic_rerank(query, results, top_k)

    def _heuristic_rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[RerankResult]:
        query_terms = set(query.lower().split())

        scored_results: list[tuple[SearchResult, float]] = []

        for result in results:
            content_lower = result.chunk.content.lower()
            term_matches = sum(1 for term in query_terms if term in content_lower)
            position_score = 1.0 / (results.index(result) + 1)
            term_bonus = term_matches * 0.1
            combined = result.score + position_score + term_bonus
            scored_results.append((result, combined))

        scored_results.sort(key=lambda x: x[1], reverse=True)

        reranked: list[RerankResult] = []
        for result, score in scored_results[:top_k]:
            reranked.append(
                RerankResult(
                    chunk=result.chunk,
                    original_score=result.score,
                    rerank_score=score,
                    rerank_type="llm_heuristic",
                )
            )

        return reranked

    def _rerank_with_llm(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[RerankResult]:
        from .llm import Evidence, LLM

        evidence_list = [
            Evidence(
                chunk_id=r.chunk.chunk_id,
                content=r.chunk.content,
                score=r.score,
            )
            for r in results
        ]

        ranking_prompt = self._build_ranking_prompt(query, evidence_list)
        try:
            ranking_text = self.llm.generate(ranking_prompt, [])
            chunk_scores = self._parse_ranking(ranking_text, [r.chunk.chunk_id for r in results])
        except Exception:
            return self._heuristic_rerank(query, results, top_k)

        chunk_id_to_result: dict[str, SearchResult] = {r.chunk.chunk_id: r for r in results}
        chunk_id_to_score: dict[str, float] = chunk_scores

        sorted_chunks = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        reranked: list[RerankResult] = []
        for chunk_id, score in sorted_chunks:
            if chunk_id in chunk_id_to_result:
                result = chunk_id_to_result[chunk_id]
                reranked.append(
                    RerankResult(
                        chunk=result.chunk,
                        original_score=result.score,
                        rerank_score=score,
                        rerank_type="llm_judged",
                    )
                )

        return reranked

    def _build_ranking_prompt(self, query: str, evidence: list) -> str:
        chunks_text = "\n".join(
            f"[{i + 1}] {e.chunk_id}: {e.content[:200]}..." for i, e in enumerate(evidence)
        )
        return (
            f"Query: {query}\n\n"
            f"Candidates:\n{chunks_text}\n\n"
            "请根据候选内容与查询的相关性，给每个候选打分(0-10)，返回格式: "
            "每行 [chunk_id]: score"
        )

    def _parse_ranking(self, text: str, chunk_ids: list[str]) -> dict[str, float]:
        scores: dict[str, float] = {cid: 0.0 for cid in chunk_ids}
        for line in text.split("\n"):
            line = line.strip()
            if ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    chunk_id = parts[0].strip("[] ")
                    score_str = parts[1].strip()
                    try:
                        score = float(score_str)
                        if chunk_id in scores:
                            scores[chunk_id] = score
                    except ValueError:
                        continue
        return scores


class SentenceTransformerReranker(Reranker):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError("请安装 sentence-transformers: pip install sentence-transformers")

        self.model = SentenceTransformer(model_name, device=device)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[RerankResult]:
        if not results:
            return []

        try:
            import numpy as np
        except ImportError:
            return self._simple_rerank(query, results, top_k)

        query_emb = self.model.encode([query])
        docs = [r.chunk.content for r in results]
        doc_embs = self.model.encode(docs)

        similarities = np.dot(query_emb, doc_embs.T).flatten()

        scored_results: list[tuple[SearchResult, float]] = list(zip(results, similarities.tolist()))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        reranked: list[RerankResult] = []
        for result, score in scored_results[:top_k]:
            reranked.append(
                RerankResult(
                    chunk=result.chunk,
                    original_score=result.score,
                    rerank_score=float(score),
                    rerank_type="sent_transformer",
                )
            )

        return reranked

    def _simple_rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[RerankResult]:
        query_terms = set(query.lower().split())

        scored_results: list[tuple[SearchResult, float]] = []
        for result in results:
            content_lower = result.chunk.content.lower()
            term_matches = sum(1 for term in query_terms if term in content_lower)
            score = result.score + (term_matches * 0.1)
            scored_results.append((result, score))

        scored_results.sort(key=lambda x: x[1], reverse=True)

        reranked: list[RerankResult] = []
        for result, score in scored_results[:top_k]:
            reranked.append(
                RerankResult(
                    chunk=result.chunk,
                    original_score=result.score,
                    rerank_score=float(score),
                    rerank_type="sent_transformer_fallback",
                )
            )

        return reranked


def build_reranker(
    rerank_type: str = "cross_encoder",
    model_name: str | None = None,
    device: str = "cpu",
    llm: Any | None = None,
) -> Reranker:
    if rerank_type == "cross_encoder":
        return CrossEncoderReranker(
            model_name=model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device=device,
        )
    elif rerank_type == "sentence_transformer":
        return SentenceTransformerReranker(
            model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2",
            device=device,
        )
    elif rerank_type == "llm_coherence":
        return LLMCoherenceReranker(llm=llm, use_api=True)
    else:
        return LLMCoherenceReranker(llm=llm, use_api=False)
