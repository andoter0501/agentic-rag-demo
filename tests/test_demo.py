from pathlib import Path

from agentic_rag_demo import AgenticRAG


class FakeLLM:
    def generate(self, question: str, evidence: list) -> str:
        return f"Q:{question};E:{len(evidence)}"


def test_answer_has_citations_for_relevant_question() -> None:
    kb = Path(__file__).parent.parent / "data" / "knowledge_base.md"
    agent = AgenticRAG.from_markdown(str(kb), llm=FakeLLM())
    resp = agent.ask("Agentic RAG 和普通 RAG 的区别是什么？")

    assert "Q:Agentic RAG 和普通 RAG 的区别是什么？;E:" in resp.answer
    assert len(resp.citations) > 0


def test_unknown_question_falls_back() -> None:
    kb = Path(__file__).parent.parent / "data" / "knowledge_base.md"
    agent = AgenticRAG.from_markdown(str(kb))
    resp = agent.ask("量子纠缠实验的最新参数是什么？")

    assert "不足" in resp.answer or "没有" in resp.answer
