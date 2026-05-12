from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal
from abc import ABC, abstractmethod


@dataclass(frozen=True)
class Evidence:
    chunk_id: str
    content: str
    score: float


class LLM(ABC):
    @abstractmethod
    def generate(self, question: str, evidence: list[Evidence]) -> str:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


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
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("缺少 openai 依赖。请先执行: pip install openai") from exc
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.api_mode = api_mode
        self.max_tokens = max_tokens
        self.temperature = temperature

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
            max_output_tokens=self.max_tokens,
        )
        text = getattr(response, "output_text", None)
        if text:
            return text.strip()
        return "模型返回为空，请重试。"

    def _generate_by_chat(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
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


class AnthropicLLM(LLM):
    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError("请执行: pip install anthropic") from exc
        self.client = Anthropic(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @property
    def name(self) -> str:
        return f"Anthropic/{self.model}"

    def _build_prompt(self, question: str, evidence: list[Evidence]) -> str:
        evidence_text = "\n".join(f"[{e.chunk_id}] {e.content}" for e in evidence)
        return (
            "你是一个 RAG 问答助手。只能依据给定证据回答，不得编造。"
            "回答使用中文，先给结论，再列关键依据。\n\n"
            f"问题: {question}\n\n证据:\n{evidence_text}"
        )

    def generate(self, question: str, evidence: list[Evidence]) -> str:
        prompt = self._build_prompt(question, evidence)
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            if response.content and response.content[0].text:
                return response.content[0].text.strip()
            return "模型返回为空，请重试。"
        except Exception as exc:
            return f"调用 Anthropic 失败: {exc}"


class AzureOpenAILLM(LLM):
    def __init__(
        self,
        model: str = "gpt-5.2",
        api_key: str | None = None,
        api_version: str = "2024-02-01",
        azure_endpoint: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> None:
        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise RuntimeError("请执行: pip install openai") from exc
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @property
    def name(self) -> str:
        return f"Azure/{self.model}"

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

    def generate(self, question: str, evidence: list[Evidence]) -> str:
        prompt = self._build_prompt(question, evidence)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content
                if isinstance(content, str) and content.strip():
                    return content.strip()
            return "模型返回为空，请重试。"
        except Exception as exc:
            return f"调用 Azure OpenAI 失败: {exc}"


class OllamaLLM(LLM):
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        options: dict[str, Any] | None = None,
    ) -> None:
        import httpx

        self.client = httpx.Client(base_url=base_url, timeout=timeout)
        self.model = model
        self.options = options or {}

    @property
    def name(self) -> str:
        return f"Ollama/{self.model}"

    def _build_prompt(self, question: str, evidence: list[Evidence]) -> str:
        evidence_text = "\n".join(f"[{e.chunk_id}] {e.content}" for e in evidence)
        return (
            "你是一个 RAG 问答助手。基于以下证据回答，不得编造。\n\n"
            f"问题: {question}\n\n证据:\n{evidence_text}\n\n回答:"
        )

    def generate(self, question: str, evidence: list[Evidence]) -> str:
        prompt = self._build_prompt(question, evidence)
        try:
            response = self.client.post(
                "/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    **self.options,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip() or "模型返回为空"
        except Exception as exc:
            return f"调用 Ollama 失败: {exc}"

    def list_models(self) -> list[str]:
        try:
            response = self.client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []


class GroqLLM(LLM):
    def __init__(
        self,
        model: str = "llama-3.1-70b-versatile",
        api_key: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> None:
        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError("请执行: pip install groq") from exc
        self.client = Groq(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @property
    def name(self) -> str:
        return f"Groq/{self.model}"

    @staticmethod
    def _build_prompt(question: str, evidence: list[Evidence]) -> str:
        evidence_text = "\n".join(f"[{e.chunk_id}] {e.content}" for e in evidence)
        return (
            "你是一个 RAG 问答助手。只能依据给定证据回答，不得编造。"
            "回答使用中文。\n\n"
            f"问题: {question}\n\n证据:\n{evidence_text}"
        )

    def generate(self, question: str, evidence: list[Evidence]) -> str:
        prompt = self._build_prompt(question, evidence)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content
                if isinstance(content, str) and content.strip():
                    return content.strip()
            return "模型返回为空，请重试。"
        except Exception as exc:
            return f"调用 Groq 失败: {exc}"


class TogetherLLM(LLM):
    def __init__(
        self,
        model: str = "meta-llama/Llama-3.2-70B-Instruct-Turbo",
        api_key: str | None = None,
        base_url: str = "https://api.together.xyz",
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("请执行: pip install openai") from exc
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @property
    def name(self) -> str:
        return f"Together/{self.model}"

    @staticmethod
    def _build_prompt(question: str, evidence: list[Evidence]) -> str:
        evidence_text = "\n".join(f"[{e.chunk_id}] {e.content}" for e in evidence)
        return (
            "你是一个 RAG 问答助手。只能依据给定证据回答，不得编造。"
            "回答使用中文。\n\n"
            f"问题: {question}\n\n证据:\n{evidence_text}"
        )

    def generate(self, question: str, evidence: list[Evidence]) -> str:
        prompt = self._build_prompt(question, evidence)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content
                if isinstance(content, str) and content.strip():
                    return content.strip()
            return "模型返回为空，请重试。"
        except Exception as exc:
            return f"调用 Together AI 失败: {exc}"


class VLLM(LLM):
    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("请执行: pip install openai") from exc
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @property
    def name(self) -> str:
        return f"vLLM/{self.model}"

    @staticmethod
    def _build_prompt(question: str, evidence: list[Evidence]) -> str:
        evidence_text = "\n".join(f"[{e.chunk_id}] {e.content}" for e in evidence)
        return (
            "你是一个 RAG 问答助手。只能依据给定证据回答，不得编造。"
            "回答使用中文。\n\n"
            f"问题: {question}\n\n证据:\n{evidence_text}"
        )

    def generate(self, question: str, evidence: list[Evidence]) -> str:
        prompt = self._build_prompt(question, evidence)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content
                if isinstance(content, str) and content.strip():
                    return content.strip()
            return "模型返回为空，请重试。"
        except Exception as exc:
            return f"调用 vLLM 失败: {exc}"


def build_llm(
    use_openai: bool = False,
    model: str | None = None,
    provider: Literal[
        "openai", "opencode", "anthropic", "azure", "ollama", "groq", "together", "vllm"
    ] = "openai",
    api_mode: Literal["auto", "responses", "chat"] = "auto",
    api_key: str | None = None,
    base_url: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> LLM:
    if not use_openai:
        return LocalHeuristicLLM()

    resolved_key = (
        api_key or os.getenv(f"{provider.upper()}_API_KEY") or os.getenv("OPENAI_API_KEY")
    )
    resolved_base = (
        base_url or os.getenv(f"{provider.upper()}_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    )

    if provider == "opencode":
        return OpenAICompatibleLLM(
            model=model or os.getenv("OPENCODE_MODEL", "minimax-2.5"),
            api_key=resolved_key,
            base_url=resolved_base or os.getenv("OPENAI_BASE_URL"),
            api_mode=api_mode,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    if provider == "anthropic":
        return AnthropicLLM(
            model=model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            api_key=resolved_key or os.getenv("ANTHROPIC_API_KEY"),
            base_url=os.getenv("ANTHROPIC_BASE_URL"),
            max_tokens=max_tokens,
            temperature=temperature,
        )

    if provider == "azure":
        return AzureOpenAILLM(
            model=model or os.getenv("AZURE_MODEL", "gpt-5.2"),
            api_key=resolved_key or os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION", "2024-02-01"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            max_tokens=max_tokens,
            temperature=temperature,
        )

    if provider == "ollama":
        return OllamaLLM(
            model=model or os.getenv("OLLAMA_MODEL", "llama3.2"),
            base_url=resolved_base or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    if provider == "groq":
        return GroqLLM(
            model=model or os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"),
            api_key=resolved_key or os.getenv("GROQ_API_KEY"),
            max_tokens=max_tokens,
            temperature=temperature,
        )

    if provider == "together":
        return TogetherLLM(
            model=model or os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.2-70B-Instruct-Turbo"),
            api_key=resolved_key or os.getenv("TOGETHER_API_KEY"),
            base_url=resolved_base or "https://api.together.xyz",
            max_tokens=max_tokens,
            temperature=temperature,
        )

    if provider == "vllm":
        return VLLM(
            model=model or os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            base_url=resolved_base or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=api_key or os.getenv("VLLM_API_KEY", "EMPTY"),
            max_tokens=max_tokens,
            temperature=temperature,
        )

    return OpenAICompatibleLLM(
        model=model or os.getenv("OPENAI_MODEL", "gpt-5.2"),
        api_key=resolved_key,
        base_url=resolved_base,
        api_mode=api_mode,
        max_tokens=max_tokens,
        temperature=temperature,
    )


SUPPORTED_PROVIDERS = {
    "openai": "OpenAI (GPT-4, GPT-3.5)",
    "opencode": "OpenCode (MiniMax 等)",
    "anthropic": "Anthropic (Claude)",
    "azure": "Azure OpenAI",
    "ollama": "Ollama (本地模型)",
    "groq": "Groq (高速推理)",
    "together": "Together AI",
    "vllm": "vLLM (本地部署)",
}


def list_providers() -> dict[str, str]:
    return SUPPORTED_PROVIDERS
