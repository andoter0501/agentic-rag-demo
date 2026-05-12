from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agentic_rag_demo import (
    AgenticRAG,
    build_llm,
    build_embedding_model,
    build_reranker,
    build_cache,
    RAGCache,
    Evaluator,
    list_providers,
)


LLMProvider = Literal[
    "openai", "opencode", "anthropic", "azure", "ollama", "groq", "together", "vllm"
]


@dataclass(frozen=True)
class RuntimeConfig:
    use_openai: bool
    provider: LLMProvider
    api_mode: Literal["auto", "responses", "chat"]
    model: str | None
    api_key: str | None
    base_url: str | None
    enable_reranker: bool
    use_agent_loop: bool
    use_cache: bool
    embedding_provider: str
    cache_type: Literal["lru", "ttl", "persistent"]
    max_tokens: int
    temperature: float


def load_project_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    env_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    provider = args.provider or env_provider

    env_mode = os.getenv("LLM_API_MODE", "auto").lower()
    api_mode = args.api_mode or (env_mode if env_mode in {"auto", "responses", "chat"} else "auto")

    use_openai = True if args.openai else _to_bool(os.getenv("LLM_ENABLE"), default=False)

    max_tokens = args.max_tokens or int(os.getenv("LLM_MAX_TOKENS", "2048"))
    temperature = args.temperature or float(os.getenv("LLM_TEMPERATURE", "0.7"))

    return RuntimeConfig(
        use_openai=use_openai,
        provider=provider,
        api_mode=api_mode,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        enable_reranker=args.enable_reranker,
        use_agent_loop=args.use_agent_loop,
        use_cache=args.use_cache,
        embedding_provider=args.embedding_provider or "auto",
        cache_type=args.cache_type or "lru",
        max_tokens=max_tokens,
        temperature=temperature,
    )


def run_demo(
    use_openai: bool = False,
    model: str | None = None,
    provider: LLMProvider = "openai",
    api_mode: Literal["auto", "responses", "chat"] = "auto",
    api_key: str | None = None,
    base_url: str | None = None,
    enable_reranker: bool = False,
    use_agent_loop: bool = False,
    use_cache: bool = False,
    embedding_provider: str = "auto",
    cache_type: Literal["lru", "ttl", "persistent"] = "lru",
    max_tokens: int = 2048,
    temperature: float = 0.7,
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
            max_tokens=max_tokens,
            temperature=temperature,
        )
        print(f"[info] LLM: {llm.name}")
    except Exception as exc:
        print(f"[warn] LLM 初始化失败，降级本地模式: {exc}")
        llm = build_llm(use_openai=False)
        use_openai = False

    print(f"[info] 加载知识库: {kb_path}")

    embedding_model = build_embedding_model(provider=embedding_provider)
    print(
        f"[info] Embedding 模型: {type(embedding_model).__name__}, 维度={embedding_model.dimension}"
    )

    reranker = build_reranker("cross_encoder") if enable_reranker else None
    if reranker:
        print("[info] 启用 Reranker: CrossEncoder")

    cache: RAGCache | None = None
    if use_cache:
        cache = build_cache(cache_type=cache_type)
        print(f"[info] 启用缓存: {cache_type}")

    agent = AgenticRAG.from_markdown(
        str(kb_path),
        llm=llm,
        embedding_provider=embedding_provider,
        enable_reranker=enable_reranker,
        max_iterations=5,
    )

    runtime = f"{provider}/{api_mode}" if use_openai else "local"
    features = []
    if enable_reranker:
        features.append("reranker")
    if use_agent_loop:
        features.append("agent_loop")
    if use_cache:
        features.append("cache")

    feature_str = f" [{', '.join(features)}]" if features else ""
    print(f"Agentic RAG Demo [{runtime}]{feature_str} (输入 q 退出, h 帮助)\n")

    while True:
        question = input("你: ").strip()
        if question.lower() in {"q", "quit", "exit"}:
            if cache:
                cache.print_stats()
            print("bye")
            return
        if question.lower() == "h":
            print("\n命令:")
            print("  q/quit/exit - 退出")
            print("  h - 帮助")
            print("  stats - 显示检索统计")
            print("  eval - 运行评估")
            print("  models - 显示支持的模型")
            continue
        if question.lower() == "stats":
            stats = agent.get_retrieval_stats()
            print(f"\n检索统计: {stats}")
            continue
        if question.lower() == "eval":
            run_evaluation(agent)
            continue
        if question.lower() == "models":
            print("\n支持的 LLM Provider:")
            for p, desc in list_providers().items():
                print(f"  {p}: {desc}")
            print()
            continue
        if not question:
            continue

        if cache:
            cached_answer = cache.get_cached_answer(question)
            if cached_answer:
                print("\n[缓存命中]")
                print(f"答案: {cached_answer.get('answer', 'N/A')}")
                print(f"引用: {', '.join(cached_answer.get('citations', []))}")
                print()
                continue

        response = agent.ask(question, use_agent_loop=use_agent_loop)

        print("\n助手:")
        print(response.answer)
        print(f"\n引用: {', '.join(response.citations) if response.citations else '无'}")
        print(f"检索类型: {response.retrieval_type}")

        if response.agent_state:
            print(f"Agent 迭代: {response.agent_state.iterations}")

        print("\nTrace:")
        for line in response.trace:
            print(f"  - {line}")
        print()

        if cache:
            cache.cache_answer(
                question,
                {
                    "answer": response.answer,
                    "citations": response.citations,
                },
            )


def run_evaluation(agent: AgenticRAG) -> None:
    print("\n[评估模式]")

    test_questions = [
        "什么是 RAG?",
        "Agentic RAG 和传统 RAG 有什么区别?",
        "RAG 评估指标有哪些?",
    ]

    evaluator = Evaluator(llm=agent.llm)

    for question in test_questions:
        response = agent.ask(question, use_agent_loop=False)

        retrieved_chunks = [c.split("(")[0] for c in response.citations]

        metrics = evaluator.evaluate_rag(
            question=question,
            answer=response.answer,
            retrieved_chunks=retrieved_chunks,
            relevant_chunks=retrieved_chunks,
            evidence_chunks=[r.split("(")[0] for r in response.citations],
        )

        print(f"\n问题: {question}")
        print(f"  Hit Rate: {metrics.hit_rate:.2%}")
        print(f"  MRR: {metrics.mrr:.3f}")
        print(f"  Faithfulness: {metrics.faithfulness:.2%}")


if __name__ == "__main__":
    load_project_env()

    providers = list(list_providers().keys())

    parser = argparse.ArgumentParser(description="Agentic RAG Demo")

    parser.add_argument("--openai", action="store_true", help="启用 LLM API")
    parser.add_argument("--provider", choices=providers, default=None, help="LLM 提供商")
    parser.add_argument("--api-mode", choices=["auto", "responses", "chat"], default=None)
    parser.add_argument("--model", default=None, help="模型 ID")
    parser.add_argument("--api-key", default=None, help="API Key")
    parser.add_argument("--base-url", default=None, help="API 基础 URL")
    parser.add_argument("--max-tokens", type=int, default=None, help="最大输出 tokens (默认 2048)")
    parser.add_argument("--temperature", type=float, default=None, help="温度参数 (默认 0.7)")

    parser.add_argument("--reranker", action="store_true", help="启用 Reranker")
    parser.add_argument("--agent-loop", action="store_true", help="启用 Agent Loop")
    parser.add_argument("--cache", action="store_true", help="启用缓存")
    parser.add_argument("--cache-type", choices=["lru", "ttl", "persistent"], default="lru")
    parser.add_argument("--embedding-provider", default="auto")

    args = parser.parse_args()
    config = resolve_runtime_config(args)

    run_demo(
        use_openai=config.use_openai,
        model=config.model,
        provider=config.provider,
        api_mode=config.api_mode,
        api_key=config.api_key,
        base_url=config.base_url,
        enable_reranker=config.enable_reranker,
        use_agent_loop=config.use_agent_loop,
        use_cache=config.use_cache,
        embedding_provider=config.embedding_provider,
        cache_type=config.cache_type,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )
