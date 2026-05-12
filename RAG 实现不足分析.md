# RAG 实现不足分析

## 1. 检索层缺陷

- **仅用稀疏检索**: `SparseRetriever` 是 BM25/TF-IDF，缺少语义向量检索（如 Embedding + FAISS/Qdrant）
- **无混合检索**: 没有结合关键词+语义的 Hybrid Search
- **无重排序**: 检索后直接返回，缺少 Reranker 优化

## 2. Agent 代理能力不足

当前 `AgenticRAG.ask()` 只是简单线性流程，无真正的代理决策：

- ❌ 无规划 (Planning)
- ❌ 无多工具调用
- ❌ 无反思 (Reflection) 机制
- ❌ 无查询澄清后的重试逻辑
- ❌ 无多轮对话上下文

## 3. 生成层问题

- `LocalHeuristicLLM` 是硬编码模板，不是真正的 LLM 生成
- 无 Streaming 输出
- 无引用验证（是否真的引用了证据）

## 4. 文档处理简陋

- 仅支持 Markdown
- 固定字符数切分（280），不考虑语义边界
- 无 Metadata 提取（来源、更新时间等）

## 5. 基础设施缺失

| 功能 | 状态 |
|------|------|
| 持久化向量存储 | ❌ |
| 评估框架 | ❌ |
| 缓存机制 | ❌ |
| 错误重试 | ❌ |
| 增量索引 | ❌ |

## 优化优先级建议

1. **短期**: 添加 Embedding + 向量数据库（Milvus/Qdrant）做语义检索

2. **中期**: 实现真正的 Agent Loop（规划→工具调用→反思）

3. **长期**: 添加评估 pipeline、缓存、多模态支持