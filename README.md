# Agentic RAG Demo

一个功能完整的 Agentic RAG 实现，支持混合检索、重排序、Agent Loop、评估和缓存。

## 核心特性

| 特性 | 描述 |
|------|------|
| 混合检索 | BM25 + Embedding 融合 (RRF/加权/凸组合) |
| 重排序 | CrossEncoder / LLM / SentenceTransformer |
| Agent Loop | 规划 → 工具调用 → 反思 → 终止判断 |
| 多格式支持 | Markdown / PDF / HTML / JSON / DOCX / TXT |
| 语义分块 | 语义切分 / 递归切分 / 重叠切分 |
| 评估框架 | Hit Rate / MRR / NDCG / 忠实性 |
| 多级缓存 | LRU / TTL / Persistent 三种模式 |

## 快速开始

### 安装依赖

```bash
pip install openai python-dotenv

# 可选依赖
pip install agentic-rag-demo[embedding]    # 向量模型
pip install agentic-rag-demo[vector]        # FAISS 加速
pip install agentic-rag-demo[parser]        # 多格式解析
pip install agentic-rag-demo[all]           # 全部依赖
```

### 运行命令

```bash
# 基础模式 (仅混合检索)
python3 main.py

# 启用重排序
python3 main.py --reranker

# 启用 Agent Loop (规划/反思)
python3 main.py --agent-loop

# 启用缓存
python3 main.py --cache --cache-type lru

# 完整功能
python3 main.py --reranker --agent-loop --cache

# 指定 Embedding 提供商
python3 main.py --embedding-provider openai
python3 main.py --embedding-provider huggingface

# 使用远程 LLM
python3 main.py --openai --provider openai --model gpt-5.2
python3 main.py --openai --provider opencode --model minimax-2.5

# 演示所有功能
python3 demo_all_features.py
```

### 交互命令

| 命令 | 说明 |
|------|------|
| `q` / `quit` / `exit` | 退出程序 |
| `h` | 显示帮助 |
| `stats` | 显示检索统计 |
| `eval` | 运行评估测试 |

## 项目结构

```
agentic-rag-demo/
├── agentic_rag_demo/          # 核心包
│   ├── agent.py              # AgenticRAG 主类
│   ├── agent_loop.py         # Agent Loop 控制器
│   ├── retriever.py          # 检索器 (稀疏/稠密/混合)
│   ├── reranker.py           # 重排序模块
│   ├── embedding.py          # 向量嵌入模型
│   ├── vector_index.py       # 向量索引
│   ├── loader.py             # 文档加载与分块
│   ├── document_parser.py    # 多格式文档解析
│   ├── evaluator.py          # 评估框架
│   ├── cache.py              # 缓存机制
│   └── llm.py                # LLM 接口
├── data/
│   └── knowledge_base.md     # 示例知识库
├── main.py                   # CLI 入口
├── demo_all_features.py      # 功能演示
└── tests/                    # 测试用例
```

## 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                         用户查询                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Query Processing                         │
│  查询改写 → 查询扩展 → 意图识别                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Retrieval Layer                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ 稀疏检索 │  │ 稠密检索 │  │ 混合检索 │  │ 重排序   │   │
│  │ BM25     │  │ Embedding│  │ RRF      │  │ CrossEnc │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Agent Controller                          │
│  规划 → 工具调度 → 反思 → 终止判断                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Generation Layer                           │
│  Prompt 构造 → LLM 生成 → 引用标注                            │
└─────────────────────────────────────────────────────────────┘
```

## 配置优先级

```
CLI 参数 > .env 环境变量 > 代码默认值
```

### .env 配置示例

```bash
# LLM 配置
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-5.2

# Opencode 配置
OPENCODE_API_KEY=your_opencode_key
OPENCODE_BASE_URL=https://api.opencode.ai/v1
OPENCODE_MODEL=minimax-2.5

# LLM 提供商
LLM_PROVIDER=opencode
LLM_API_MODE=auto
```

## 示例问题

- `什么是 RAG?`
- `Agentic RAG 和传统 RAG 有什么区别?`
- `RAG 的评估指标有哪些?`
- `实践建议是什么?`

## 测试

```bash
# 运行所有测试
python3 -m pytest

# 快速测试
python3 -m pytest tests/test_demo.py -q
```

## 扩展指南

### 添加新的 Embedding 模型

```python
from agentic_rag_demo import build_embedding_model

# 使用 OpenAI Embedding
embed = build_embedding_model(provider="openai", model_name="text-embedding-3-small")

# 使用 HuggingFace 模型
embed = build_embedding_model(provider="huggingface", model_name="BAAI/bge-small-zh-v1.5")
```

### 自定义 Reranker

```python
from agentic_rag_demo import build_reranker

reranker = build_reranker("cross_encoder", model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
```

### 批量评估

```python
from agentic_rag_demo import BatchEvaluator, Evaluator, AgenticRAG

evaluator = BatchEvaluator(Evaluator())
results = evaluator.evaluate_batch(test_cases)
evaluator.print_summary(results)
```