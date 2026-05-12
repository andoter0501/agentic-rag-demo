#!/usr/bin/env python3
"""演示增强后的 Agentic RAG 所有新功能"""

from __future__ import annotations

import time

from agentic_rag_demo import (
    AgenticRAG,
    build_llm,
    build_embedding_model,
    build_reranker,
    build_cache,
    load_semantic_chunks,
    parse_document,
    Evaluator,
    BatchEvaluator,
    SearchResult,
)
from agentic_rag_demo.loader import SemanticChunker, RecursiveChunker


def demo_basic_retrieval():
    print("\n" + "=" * 60)
    print("1. 基础检索演示 (混合检索)")
    print("=" * 60)

    kb_path = "data/knowledge_base.md"
    agent = AgenticRAG.from_markdown(kb_path)

    questions = [
        "什么是 RAG?",
        "Agentic RAG 是什么?",
    ]

    for q in questions:
        print(f"\n问题: {q}")
        response = agent.ask(q)
        print(f"答案: {response.answer[:100]}...")
        print(f"引用: {response.citations}")
        print(f"检索类型: {response.retrieval_type}")


def demo_semantic_chunks():
    print("\n" + "=" * 60)
    print("2. 语义分块演示")
    print("=" * 60)

    kb_path = "data/knowledge_base.md"

    semantic_chunks = load_semantic_chunks(kb_path, strategy="semantic", max_chars=200)
    print(f"\n语义分块结果: {len(semantic_chunks)} 个块")

    recursive_chunks = load_semantic_chunks(kb_path, strategy="recursive", max_chars=200)
    print(f"递归分块结果: {len(recursive_chunks)} 个块")

    for i, chunk in enumerate(semantic_chunks[:3]):
        print(f"\n  块 {i + 1}: {chunk.chunk_id}")
        print(f"  内容: {chunk.content[:80]}...")
        print(f"  元数据: {chunk.metadata}")


def demo_document_parsing():
    print("\n" + "=" * 60)
    print("3. 多格式文档解析演示")
    print("=" * 60)

    from agentic_rag_demo import (
        MarkdownParser,
        TXTParser,
        get_parser_for_file,
    )

    print("\nMarkdown 解析器:")
    parser = MarkdownParser()
    docs = parser.parse("data/knowledge_base.md")
    print(f"  解析文档数: {len(docs)}")
    print(f"  内容长度: {len(docs[0].content) if docs else 0}")

    print("\n自动检测文件类型:")
    detected_parser = get_parser_for_file("test.md")
    print(f"  .md -> {type(detected_parser).__name__}")

    detected_parser = get_parser_for_file("test.txt")
    print(f"  .txt -> {type(detected_parser).__name__}")


def demo_reranker():
    print("\n" + "=" * 60)
    print("4. Reranker 重排序演示")
    print("=" * 60)

    try:
        reranker = build_reranker("cross_encoder")
        print(f"Reranker 类型: {type(reranker).__name__}")

        from agentic_rag_demo import SearchResult, Chunk

        test_chunks = [
            Chunk(chunk_id="test-1", title="Test", content="RAG 是检索增强生成技术"),
            Chunk(chunk_id="test-2", title="Test", content="深度学习是机器学习的子领域"),
            Chunk(chunk_id="test-3", title="Test", content="RAG 系统由检索器和生成器组成"),
        ]

        results = [SearchResult(chunk=c, score=0.5 + i * 0.1) for i, c in enumerate(test_chunks)]

        query = "什么是 RAG"
        reranked = reranker.rerank(query, results, top_k=3)

        print(f"\n查询: {query}")
        print("重排序结果:")
        for r in reranked:
            print(f"  {r.chunk.chunk_id}: {r.rerank_score:.4f} ({r.rerank_type})")

    except Exception as e:
        print(f"Reranker 需要额外依赖: {e}")
        print("运行: pip install sentence-transformers")


def demo_caching():
    print("\n" + "=" * 60)
    print("5. 缓存机制演示")
    print("=" * 60)

    cache = build_cache(cache_type="lru", max_size=100)

    test_queries = [
        "什么是 RAG",
        "什么是 RAG",
        "Agentic RAG",
        "什么是 RAG",
    ]

    print("\n缓存查询测试:")
    for q in test_queries:
        cached = cache.get_cached_answer(q)
        if cached:
            print(f"  [命中] {q[:20]}...")
        else:
            print(f"  [未命中] {q[:20]}...")
            cache.cache_answer(q, {"answer": f"缓存的答案: {q}"})

    cache.print_stats()


def demo_evaluation():
    print("\n" + "=" * 60)
    print("6. 评估框架演示")
    print("=" * 60)

    evaluator = Evaluator()

    test_cases = [
        {
            "question": "什么是 RAG?",
            "answer": "RAG 是检索增强生成技术，结合检索和生成。",
            "retrieved_chunks": ["chunk-1", "chunk-2"],
            "relevant_chunks": ["chunk-1"],
            "evidence_chunks": ["RAG 是检索增强生成"],
        },
        {
            "question": "Agentic RAG 是什么?",
            "answer": "Agentic RAG 在传统 RAG 上增加代理行为。",
            "retrieved_chunks": ["chunk-3"],
            "relevant_chunks": ["chunk-3"],
            "evidence_chunks": ["Agentic RAG 增加规划、反思等能力"],
        },
    ]

    batch_evaluator = BatchEvaluator(evaluator)
    results = batch_evaluator.evaluate_batch(test_cases)
    batch_evaluator.print_summary(results)


def demo_agent_loop():
    print("\n" + "=" * 60)
    print("7. Agent Loop 演示")
    print("=" * 60)

    kb_path = "data/knowledge_base.md"
    llm = build_llm(use_openai=False)

    agent = AgenticRAG.from_markdown(kb_path, llm=llm)

    print("\n使用 Agent Loop 模式:")
    response = agent.ask("什么是 Agentic RAG?", use_agent_loop=True)

    print(f"\n答案: {response.answer[:100]}...")
    print(f"引用: {response.citations}")
    print(f"Agent 迭代次数: {response.agent_state.iterations if response.agent_state else 0}")

    if response.agent_state:
        print("\nAgent 动作:")
        for action in response.agent_state.actions:
            print(f"  {action.action_type.value}: {action.thought[:50]}...")


def demo_retrieval_stats():
    print("\n" + "=" * 60)
    print("8. 检索统计信息")
    print("=" * 60)

    kb_path = "data/knowledge_base.md"
    agent = AgenticRAG.from_markdown(kb_path)

    stats = agent.get_retrieval_stats()
    print("\n系统统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def demo_embedding():
    print("\n" + "=" * 60)
    print("9. Embedding 模型演示")
    print("=" * 60)

    try:
        embed_model = build_embedding_model(provider="local")
        print(f"Embedding 模型: {type(embed_model).__name__}")
        print(f"向量维度: {embed_model.dimension}")

        test_texts = ["RAG 是检索增强生成", "深度学习"]
        embeddings = embed_model.embed_batch(test_texts)

        print(f"\n生成 {len(embeddings)} 个向量")
        print(f"向量长度: {len(embeddings[0])}")

    except Exception as e:
        print(f"Embedding 需要额外依赖: {e}")


def main():
    print("=" * 60)
    print("Agentic RAG 增强功能演示")
    print("=" * 60)

    demo_basic_retrieval()
    demo_semantic_chunks()
    demo_document_parsing()
    demo_reranker()
    demo_caching()
    demo_evaluation()
    demo_agent_loop()
    demo_retrieval_stats()
    demo_embedding()

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
