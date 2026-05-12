from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Evidence:
    chunk_id: str
    content: str
    score: float


class LLM:
    def generate(self, question: str, evidence: list[Evidence]) -> str:
        raise NotImplementedError


class LocalHeuristicLLM(LLM):
    def generate(self, question: str, evidence: list[Evidence]) -> str:
        evidence_lines = [f"- [{e.chunk_id}] {e.content}" for e in evidence]
        summary = "\n".join(evidence_lines)
        return (
            "基于知识库证据，结论如下：\n"
            "1. Agentic RAG = RAG + 代理决策（规划/工具调用/反思）。\n"
            "2. 实践上建议先做最小闭环，并保留检索日志与引用。\n"
            "3. 评估时要同时看命中率、忠实性、延迟和成本。\n\n"
            f"问题：{question}\n\n"
            f"证据：\n{summary}"
        )


class OpenAICompatibleLLM(LLM):
    def __init__(
        self,
        model: str = "gpt-5.2",
        api_key: str | None = None,
        base_url: str | None = None,
        api_mode: Literal["auto", "responses", "chat"] = "auto",
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "缺少 openai 依赖。请先在虚拟环境执行: pip install openai"
            ) from exc
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.api_mode = api_mode

    @staticmethod
    def _build_prompt(question: str, evidence: list[Evidence]) -> str:
        evidence_text = "\n".join(
            f"[{e.chunk_id}] score={e.score:.3f} {e.content}" for e in evidence
        )
        return (
            "你是一个 RAG 问答助手。只能依据给定证据回答，不得编造。"
            "如果证据不足，请明确说证据不足。回答使用中文，先给结论，再列关键依据。"
            f"\n\n问题:\n{question}\n\n证据:\n{evidence_text}"
        )

    def _generate_by_responses(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )
        text = getattr(response, "output_text", None)
        if text:
            return text.strip()
        return "模型返回为空，请重试。"

    def _generate_by_chat(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.choices and response.choices[0].message:
            content = response.choices[0].message.content
            if isinstance(content, str) and content.strip():
                return content.strip()
        return "模型返回为空，请重试。"

    def generate(self, question: str, evidence: list[Evidence]) -> str:
        prompt = self._build_prompt(question, evidence)
        if self.api_mode == "responses":
            return self._generate_by_responses(prompt)
        if self.api_mode == "chat":
            return self._generate_by_chat(prompt)
        try:
            return self._generate_by_responses(prompt)
        except Exception:
            return self._generate_by_chat(prompt)


def build_llm(
    use_openai: bool,
    model: str | None = None,
    provider: Literal["openai", "opencode"] = "openai",
    api_mode: Literal["auto", "responses", "chat"] = "auto",
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLM:
    if not use_openai:
        return LocalHeuristicLLM()
    if provider == "opencode":
        chosen_model = model or os.getenv("OPENCODE_MODEL", "minimax-2.5")
        resolved_key = api_key or os.getenv("OPENCODE_API_KEY") or os.getenv("OPENAI_API_KEY")
        resolved_base = (
            base_url
            or os.getenv("OPENCODE_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
        )
    else:
        chosen_model = model or os.getenv("OPENAI_MODEL", "gpt-5.2")
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        resolved_base = base_url or os.getenv("OPENAI_BASE_URL")
    return OpenAICompatibleLLM(
        model=chosen_model,
        api_key=resolved_key,
        base_url=resolved_base,
        api_mode=api_mode,
    )
