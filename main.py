from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agentic_rag_demo import AgenticRAG, build_llm


@dataclass(frozen=True)
class RuntimeConfig:
    use_openai: bool
    provider: Literal["openai", "opencode"]
    api_mode: Literal["auto", "responses", "chat"]
    model: str | None
    api_key: str | None
    base_url: str | None


def load_project_env() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        return
    load_dotenv()


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    env_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    provider = args.provider or ("opencode" if env_provider == "opencode" else "openai")

    env_mode = os.getenv("LLM_API_MODE", "auto").lower()
    api_mode = args.api_mode or (
        env_mode if env_mode in {"auto", "responses", "chat"} else "auto"
    )

    use_openai = (
        True
        if args.openai
        else _to_bool(os.getenv("LLM_ENABLE"), default=False)
    )

    model = args.model
    api_key = args.api_key
    base_url = args.base_url

    if provider == "opencode":
        model = model or os.getenv("OPENCODE_MODEL") or os.getenv("OPENAI_MODEL")
        api_key = api_key or os.getenv("OPENCODE_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENCODE_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    else:
        model = model or os.getenv("OPENAI_MODEL")
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_BASE_URL")

    return RuntimeConfig(
        use_openai=use_openai,
        provider=provider,
        api_mode=api_mode,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )


def run_demo(
    use_openai: bool = False,
    model: str | None = None,
    provider: Literal["openai", "opencode"] = "openai",
    api_mode: Literal["auto", "responses", "chat"] = "auto",
    api_key: str | None = None,
    base_url: str | None = None,
) -> None:
    kb_path = Path(__file__).parent / "data" / "knowledge_base.md"
    try:
        llm = build_llm(
            use_openai=use_openai,
            model=model,
            provider=provider,
            api_mode=api_mode,
            api_key=api_key,
            base_url=base_url,
        )
    except Exception as exc:
        print(f"[warn] OpenAI 初始化失败，降级本地模式: {exc}")
        llm = build_llm(use_openai=False)
        use_openai = False
    agent = AgenticRAG.from_markdown(str(kb_path), llm=llm)

    runtime = f"{provider}/{api_mode}" if use_openai else "local"
    print(f"Agentic RAG Demo [{runtime}] (输入 q 退出)\n")
    while True:
        question = input("你: ").strip()
        if question.lower() in {"q", "quit", "exit"}:
            print("bye")
            return
        if not question:
            continue

        response = agent.ask(question)
        print("\n助手:")
        print(response.answer)
        print("\n引用:", ", ".join(response.citations) if response.citations else "无")
        print("\nTrace:")
        for line in response.trace:
            print(f"- {line}")
        print()


if __name__ == "__main__":
    load_project_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai", action="store_true", help="启用 OpenAI 兼容 API")
    parser.add_argument("--provider", choices=["openai", "opencode"], default=None)
    parser.add_argument("--api-mode", choices=["auto", "responses", "chat"], default=None)
    parser.add_argument("--model", default=None, help="模型 ID，例如 gpt-5.2 或 minimax-2.5")
    parser.add_argument("--api-key", default=None, help="可选，直接传 API Key")
    parser.add_argument("--base-url", default=None, help="可选，OpenAI 兼容网关地址")
    args = parser.parse_args()
    config = resolve_runtime_config(args)
    run_demo(
        use_openai=config.use_openai,
        model=config.model,
        provider=config.provider,
        api_mode=config.api_mode,
        api_key=config.api_key,
        base_url=config.base_url,
    )
