from __future__ import annotations

import re
from dataclasses import dataclass

from .loader import Chunk, load_chunks
from .llm import Evidence, LLM, LocalHeuristicLLM
from .retriever import SearchResult, SparseRetriever


@dataclass
class AgentResponse:
    answer: str
    citations: list[str]
    trace: list[str]


class AgenticRAG:
    def __init__(self, chunks: list[Chunk], llm: LLM | None = None) -> None:
        self.retriever = SparseRetriever(chunks)
        self.llm = llm or LocalHeuristicLLM()

    @classmethod
    def from_markdown(cls, path: str, llm: LLM | None = None) -> "AgenticRAG":
        return cls(load_chunks(path), llm=llm)

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
    def _dedupe_results(results: list[SearchResult], top_k: int = 4) -> list[SearchResult]:
        merged: dict[str, SearchResult] = {}
        for item in results:
            existing = merged.get(item.chunk.chunk_id)
            if existing is None or item.score > existing.score:
                merged[item.chunk.chunk_id] = item
        ranked = sorted(merged.values(), key=lambda x: x.score, reverse=True)
        return ranked[:top_k]

    def ask(self, question: str) -> AgentResponse:
        trace: list[str] = ["step1: 接收问题"]

        queries = self._rewrite_query(question)
        trace.append(f"step2: 生成查询变体 -> {queries}")

        all_hits: list[SearchResult] = []
        for query in queries:
            hits = self.retriever.search(query, top_k=3)
            trace.append(f"tool:retrieve('{query}') -> {len(hits)} hits")
            all_hits.extend(hits)

        chosen = self._dedupe_results(all_hits)
        trace.append(f"step3: 合并去重后保留 {len(chosen)} 条证据")

        if not chosen:
            return AgentResponse(
                answer="我没有在知识库中找到足够证据。请补充文档或换一种问法。",
                citations=[],
                trace=trace,
            )

        # Reflect: if all scores are weak, ask for clarification.
        if chosen[0].score < 0.12:
            trace.append("step4: 相关性偏低，触发澄清策略")
            return AgentResponse(
                answer="我找到了弱相关内容，但不足以稳定回答。请给出更具体的问题关键词。",
                citations=[f"{r.chunk.chunk_id}({r.score:.3f})" for r in chosen],
                trace=trace,
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
        return AgentResponse(answer=answer, citations=citations, trace=trace)
