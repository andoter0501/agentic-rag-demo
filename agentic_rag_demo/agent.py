from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .agent_loop import AgentLoop, AgentState, build_agent_loop
from .embedding import build_embedding_model, EmbeddingModel
from .loader import Chunk, load_chunks
from .llm import Evidence, LLM, LocalHeuristicLLM
from .reranker import Reranker, build_reranker
from .retriever import (
    DenseRetriever,
    HybridRetriever,
    SearchResult,
    SparseRetriever,
)
from .vector_index import VectorIndex


@dataclass
class AgentResponse:
    answer: str
    citations: list[str]
    trace: list[str]
    agent_state: AgentState | None = None
    retrieval_type: str = "hybrid"


class AgenticRAG:
    def __init__(
        self,
        chunks: list[Chunk],
        retriever: HybridRetriever | None = None,
        reranker: Reranker | None = None,
        llm: LLM | None = None,
        embedding_model: EmbeddingModel | None = None,
        max_iterations: int = 5,
    ) -> None:
        self.chunks = chunks
        self.embedding_model = embedding_model

        if retriever:
            self.retriever = retriever
        else:
            sparse = SparseRetriever(chunks)
            dense = None
            if embedding_model:
                embeddings = embedding_model.embed_batch([c.content for c in chunks])
                dense = DenseRetriever(chunks, embeddings)
            self.retriever = HybridRetriever(chunks, sparse=sparse, dense=dense)

        self.reranker = reranker
        self.llm = llm or LocalHeuristicLLM()
        self.max_iterations = max_iterations

        self._agent_loop: AgentLoop | None = None

    @classmethod
    def from_markdown(
        cls,
        path: str,
        llm: LLM | None = None,
        embedding_provider: str = "auto",
        embedding_model_name: str | None = None,
        enable_reranker: bool = False,
        max_iterations: int = 5,
    ) -> "AgenticRAG":
        chunks = load_chunks(path)
        embed_model = build_embedding_model(
            provider=embedding_provider,
            model_name=embedding_model_name,
        )
        reranker = build_reranker("cross_encoder") if enable_reranker else None
        return cls(
            chunks=chunks,
            llm=llm,
            embedding_model=embed_model,
            reranker=reranker,
            max_iterations=max_iterations,
        )

    @classmethod
    def from_documents(
        cls,
        paths: list[str],
        llm: LLM | None = None,
        embedding_provider: str = "auto",
        embedding_model_name: str | None = None,
        enable_reranker: bool = False,
        max_iterations: int = 5,
    ) -> "AgenticRAG":
        all_chunks: list[Chunk] = []
        for path in paths:
            chunks = load_chunks(path)
            all_chunks.extend(chunks)

        embed_model = build_embedding_model(
            provider=embedding_provider,
            model_name=embedding_model_name,
        )
        reranker = build_reranker("cross_encoder") if enable_reranker else None

        return cls(
            chunks=all_chunks,
            llm=llm,
            embedding_model=embed_model,
            reranker=reranker,
            max_iterations=max_iterations,
        )

    @property
    def agent_loop(self) -> AgentLoop:
        if self._agent_loop is None:
            self._agent_loop = build_agent_loop(
                retriever=self.retriever,
                reranker=self.reranker,
                llm=self.llm,
                max_iterations=self.max_iterations,
            )
        return self._agent_loop

    @staticmethod
    def _rewrite_query(question: str) -> list[str]:
        cleaned = re.sub(r"[？?。,.!！]", " ", question).strip()
        tokens = [t for t in cleaned.split() if t]
        keyword_query = " ".join(tokens[:6]) if tokens else question
        variants = [question]
        if keyword_query and keyword_query != question:
            variants.append(keyword_query)
        return variants

    @staticmethod
    def _dedupe_results(
        results: list[SearchResult], top_k: int = 4
    ) -> list[SearchResult]:
        merged: dict[str, SearchResult] = {}
        for item in results:
            existing = merged.get(item.chunk.chunk_id)
            if existing is None or item.score > existing.score:
                merged[item.chunk.chunk_id] = item
        ranked = sorted(merged.values(), key=lambda x: x.score, reverse=True)
        return ranked[:top_k]

    def _get_query_embedding(self, query: str) -> list[float] | None:
        if self.embedding_model:
            return self.embedding_model.embed(query)
        return None

    def ask(self, question: str, use_agent_loop: bool = False) -> AgentResponse:
        trace: list[str] = ["step1: 接收问题"]

        if use_agent_loop and self.llm:
            return self._ask_with_agent(question, trace)

        return self._ask_simple(question, trace)

    def _ask_simple(self, question: str, trace: list[str]) -> AgentResponse:
        queries = self._rewrite_query(question)
        trace.append(f"step2: 生成查询变体 -> {queries}")

        all_hits: list[SearchResult] = []
        query_embedding = self._get_query_embedding(question)

        for query in queries:
            hits = self.retriever.search(
                query, query_embedding=query_embedding, top_k=3
            )
            trace.append(f"tool:retrieve('{query}') -> {len(hits)} hits")
            all_hits.extend(hits)

        chosen = self._dedupe_results(all_hits)
        trace.append(f"step3: 合并去重后保留 {len(chosen)} 条证据")

        if not chosen:
            return AgentResponse(
                answer="我没有在知识库中找到足够证据。请补充文档或换一种问法。",
                citations=[],
                trace=trace,
                retrieval_type="none",
            )

        if self.reranker and chosen:
            from .reranker import RerankResult

            reranked = self.reranker.rerank(question, chosen, top_k=3)
            if reranked:
                chosen = [
                    SearchResult(
                        chunk=r.chunk, score=r.rerank_score, retrieval_type="reranked"
                    )
                    for r in reranked
                ]
                trace.append(f"step3.5: 重排序后保留 {len(chosen)} 条")

        if chosen[0].score < 0.12:
            trace.append("step4: 相关性偏低，触发澄清策略")
            return AgentResponse(
                answer="我找到了弱相关内容，但不足以稳定回答。请给出更具体的问题关键词。",
                citations=[f"{r.chunk.chunk_id}({r.score:.3f})" for r in chosen],
                trace=trace,
                retrieval_type="weak",
            )

        trace.append("step4: 证据充足，调用 llm 生成回答")
        evidence = [
            Evidence(chunk_id=r.chunk.chunk_id, content=r.chunk.content, score=r.score)
            for r in chosen
        ]
        try:
            answer = self.llm.generate(question, evidence)
        except Exception as exc:
            trace.append(f"step5: llm 调用失败，降级本地回答 -> {exc}")
            answer = LocalHeuristicLLM().generate(question, evidence)

        citations = [f"{r.chunk.chunk_id}({r.score:.3f})" for r in chosen]
        return AgentResponse(
            answer=answer,
            citations=citations,
            trace=trace,
            retrieval_type=chosen[0].retrieval_type if chosen else "unknown",
        )

    def _ask_with_agent(self, question: str, trace: list[str]) -> AgentResponse:
        trace.append("step2: 启用 Agent Loop")

        context = []
        for chunk in self.chunks[:10]:
            context.append({"chunk": chunk, "score": 1.0})

        state = self.agent_loop.run(question, context)

        trace.append(f"agent: 执行 {state.iterations} 次迭代")

        for action in state.actions:
            trace.append(f"  - {action.action_type.value}: {action.thought[:50]}...")

        if state.reflections:
            trace.append(f"  反思: {state.reflections[-1][:50]}...")

        answer = state.final_answer or "无法生成回答"
        citations = [
            e.get("chunk").chunk_id if isinstance(e, dict) else str(e)
            for e in state.evidence[:5]
            if isinstance(e, dict) and "chunk" in e
        ]

        return AgentResponse(
            answer=answer,
            citations=citations,
            trace=trace,
            agent_state=state,
            retrieval_type="agent_loop",
        )

    def batch_ask(
        self, questions: list[str], use_agent_loop: bool = False
    ) -> list[AgentResponse]:
        return [self.ask(q, use_agent_loop=use_agent_loop) for q in questions]

    def get_retrieval_stats(self) -> dict[str, Any]:
        return {
            "total_chunks": len(self.chunks),
            "retrieval_type": "hybrid",
            "has_reranker": self.reranker is not None,
            "has_embedding": self.embedding_model is not None,
            "embedding_dimension": self.embedding_model.dimension
            if self.embedding_model
            else 0,
        }
