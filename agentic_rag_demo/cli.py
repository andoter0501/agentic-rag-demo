"""命令行接口"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import get_settings, Settings
from .agent import AgenticRAG
from .llm import build_llm, list_providers
from .embedding import build_embedding_model
from .reranker import build_reranker
from .cache import build_cache
from .evaluator import Evaluator


def create_parser() -> argparse.ArgumentParser:
    settings = get_settings()

    providers = list(list_providers().keys())

    parser = argparse.ArgumentParser(
        description="Agentic RAG Demo - 智能问答系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--version", action="version", version=f"%(prog)s {settings.version}")
    parser.add_argument("--config", type=Path, help="配置文件路径 (YAML/JSON)")

    # LLM 配置
    llm_group = parser.add_argument_group("LLM 配置")
    llm_group.add_argument("--openai", action="store_true", help="启用 LLM API")
    llm_group.add_argument("--provider", choices=providers, default=None, help="LLM 提供商")
    llm_group.add_argument("--model", default=None, help="模型 ID")
    llm_group.add_argument("--api-key", default=None, help="API Key")
    llm_group.add_argument("--base-url", default=None, help="API 基础 URL")
    llm_group.add_argument("--max-tokens", type=int, default=None, help="最大输出 tokens")
    llm_group.add_argument("--temperature", type=float, default=None, help="温度参数")
    llm_group.add_argument("--api-mode", choices=["auto", "responses", "chat"], default=None)

    # Embedding 配置
    embed_group = parser.add_argument_group("Embedding 配置")
    embed_group.add_argument("--embedding-provider", default=None, help="Embedding 提供商")

    # 解析器配置
    parser_group = parser.add_argument_group("文档解析器配置")
    parser_group.add_argument(
        "--parser-type", choices=["native", "llamaindex", "langchain", "mineru"], default=None
    )
    parser_group.add_argument("--chunk-size", type=int, default=None, help="分块大小")
    parser_group.add_argument("--chunk-overlap", type=int, default=None, help="分块重叠")

    # 检索配置
    retrieval_group = parser.add_argument_group("检索配置")
    retrieval_group.add_argument("--reranker", action="store_true", help="启用 Reranker")
    retrieval_group.add_argument("--top-k", type=int, default=None, help="检索数量")

    # Agent 配置
    agent_group = parser.add_argument_group("Agent 配置")
    agent_group.add_argument("--agent-loop", action="store_true", help="启用 Agent Loop")
    agent_group.add_argument("--max-iterations", type=int, default=None, help="最大迭代次数")

    # 缓存配置
    cache_group = parser.add_argument_group("缓存配置")
    cache_group.add_argument("--cache", action="store_true", help="启用缓存")
    cache_group.add_argument("--cache-type", choices=["lru", "ttl", "persistent"], default=None)

    # 评估配置
    eval_group = parser.add_argument_group("评估配置")
    eval_group.add_argument("--eval", action="store_true", help="运行评估")

    # 其他
    parser.add_argument(
        "--knowledge-base", type=Path, default=Path("data/knowledge_base.md"), help="知识库路径"
    )
    parser.add_argument("--debug", action="store_true", help="调试模式")

    return parser


def resolve_config(args: argparse.Namespace) -> Settings:
    """从命令行参数和配置文件解析配置"""
    settings = get_settings()

    if args.config and args.config.exists():
        if args.config.suffix in {".yaml", ".yml"}:
            settings = Settings.from_yaml(args.config)
        elif args.config.suffix == ".json":
            settings = Settings.from_json(args.config)

    # LLM 配置
    if args.openai or args.provider:
        settings.llm.provider = args.provider or settings.llm.provider
    settings.llm.model = args.model or settings.llm.model
    settings.llm.api_key = args.api_key or settings.llm.api_key
    settings.llm.base_url = args.base_url or settings.llm.base_url
    settings.llm.max_tokens = args.max_tokens or settings.llm.max_tokens
    settings.llm.temperature = args.temperature or settings.llm.temperature
    if args.api_mode:
        settings.llm.api_mode = args.api_mode

    # Embedding 配置
    if args.embedding_provider:
        settings.embedding.provider = args.embedding_provider

    # 解析器配置
    if args.parser_type:
        settings.parser.parser_type = args.parser_type
    if args.chunk_size:
        settings.parser.chunk_size = args.chunk_size
    if args.chunk_overlap:
        settings.parser.chunk_overlap = args.chunk_overlap

    # 检索配置
    if args.reranker:
        settings.retrieval.enable_reranker = True
    if args.top_k:
        settings.retrieval.top_k = args.top_k

    # Agent 配置
    if args.agent_loop:
        settings.agent.use_agent_loop = True
    if args.max_iterations:
        settings.agent.max_iterations = args.max_iterations

    # 缓存配置
    if args.cache:
        settings.cache.enabled = True
    if args.cache_type:
        settings.cache.cache_type = args.cache_type

    # 评估配置
    if args.eval:
        settings.evaluator.enabled = True

    # 调试模式
    if args.debug:
        settings.debug = True

    return settings


def run_demo(settings: Settings, kb_path: Path) -> None:
    """运行演示"""
    llm = build_llm(
        use_openai=True,
        model=settings.llm.model,
        provider=settings.llm.provider,
        api_mode=settings.llm.api_mode,
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        max_tokens=settings.llm.max_tokens,
        temperature=settings.llm.temperature,
    )
    print(f"[info] LLM: {llm.name}")

    embedding_model = build_embedding_model(provider=settings.embedding.provider)
    print(f"[info] Embedding: {type(embedding_model).__name__}, 维度={embedding_model.dimension}")

    reranker = None
    if settings.retrieval.enable_reranker:
        reranker = build_reranker(settings.retrieval.reranker_type)
        print(f"[info] Reranker: {settings.retrieval.reranker_type}")

    cache = None
    if settings.cache.enabled:
        cache = build_cache(cache_type=settings.cache.cache_type)
        print(f"[info] 缓存: {settings.cache.cache_type}")

    agent = AgenticRAG.from_markdown(
        str(kb_path),
        llm=llm,
        embedding_provider=settings.embedding.provider,
        enable_reranker=settings.retrieval.enable_reranker,
        max_iterations=settings.agent.max_iterations,
    )

    print(f"\nAgentic RAG Demo [{settings.llm.provider}] (输入 q 退出, h 帮助)\n")

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
            print("  stats - 检索统计")
            print("  eval - 评估模式")
            print("  models - 支持的 LLM")
            continue

        if question.lower() == "stats":
            print(f"\n{agent.get_retrieval_stats()}")
            continue

        if question.lower() == "eval":
            run_evaluation(agent, settings)
            continue

        if question.lower() == "models":
            from .llm import list_providers

            print("\n支持的 LLM Provider:")
            for p, desc in list_providers().items():
                print(f"  {p}: {desc}")
            print()
            continue

        if not question:
            continue

        if cache:
            cached = cache.get_cached_answer(question)
            if cached:
                print(f"\n[缓存命中] {cached.get('answer', '')[:100]}...\n")
                continue

        response = agent.ask(question, use_agent_loop=settings.agent.use_agent_loop)

        print(f"\n助手: {response.answer}")
        print(f"引用: {', '.join(response.citations) or '无'}")
        print(f"检索类型: {response.retrieval_type}\n")

        if cache:
            cache.cache_answer(
                question, {"answer": response.answer, "citations": response.citations}
            )


def run_evaluation(agent: AgenticRAG, settings: Settings) -> None:
    """运行评估"""
    print("\n[评估模式]")

    test_questions = [
        "什么是 RAG?",
        "Agentic RAG 和传统 RAG 有什么区别?",
        "RAG 的评估指标有哪些?",
    ]

    evaluator = Evaluator(llm=agent.llm)

    for question in test_questions:
        response = agent.ask(question)
        retrieved = [c.split("(")[0] for c in response.citations]

        metrics = evaluator.evaluate_rag(
            question=question,
            answer=response.answer,
            retrieved_chunks=retrieved,
            relevant_chunks=retrieved,
            evidence_chunks=retrieved,
        )

        print(f"\n问题: {question}")
        print(f"  Hit Rate: {metrics.hit_rate:.2%}")
        print(f"  MRR: {metrics.mrr:.3f}")
        print(f"  Faithfulness: {metrics.faithfulness:.2%}")


def main() -> None:
    """主入口"""
    parser = create_parser()
    args = parser.parse_args()

    try:
        settings = resolve_config(args)

        kb_path = args.knowledge_base
        if not kb_path.exists():
            print(f"[错误] 知识库文件不存在: {kb_path}")
            sys.exit(1)

        run_demo(settings, kb_path)

    except KeyboardInterrupt:
        print("\n\nbye")
        sys.exit(0)
    except Exception as exc:
        print(f"[错误] {exc}")
        if args.debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
