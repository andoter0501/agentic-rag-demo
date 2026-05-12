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
| 多 LLM 支持 | OpenAI / Claude / Ollama / Groq 等 |
| 多解析器支持 | Native / LlamaIndex / LangChain / MinerU |

## 支持的 LLM 模型

| Provider | 模型示例 | 描述 |
|----------|---------|------|
| `openai` | gpt-4o, gpt-4-turbo, gpt-3.5-turbo | OpenAI 官方模型 |
| `opencode` | minimax-2.5, minimax-abab6.5 | MiniMax OpenCode |
| `anthropic` | claude-3-5-sonnet, claude-3-opus | Anthropic Claude |
| `azure` | gpt-4o, gpt-4 | Azure OpenAI |
| `ollama` | llama3.2, qwen2.5, mistral | 本地部署模型 |
| `groq` | llama-3.1-70b, mixtral-8x7b | Groq 高速推理 |
| `together` | meta-llama/Llama-3.2-70B | Together AI |
| `vllm` | Qwen/Qwen2.5-7B | vLLM 本地部署 |

## 支持的文档解析器

| 解析器 | 说明 | 优势 |
|--------|------|------|
| `native` | 内置解析器 | 轻量、无依赖 |
| `llamaindex` | LlamaIndex | 功能丰富、语义分块 |
| `langchain` | LangChain | 生态完整、格式多样 |
| `mineru` | MinerU | 优秀 PDF/文档解析 |

## 快速开始

### 安装依赖

```bash
pip install openai python-dotenv

# 可选依赖
pip install agentic-rag-demo[embedding]    # 向量模型
pip install agentic-rag-demo[vector]      # FAISS 加速
pip install agentic-rag-demo[parser]      # 多格式解析
pip install agentic-rag-demo[all]         # 全部依赖
```

### LLM 运行命令

```bash
# 本地模式 (无需 API)
python3 main.py

# OpenAI
python3 main.py --openai --provider openai --model gpt-4o

# OpenCode (MiniMax)
python3 main.py --openai --provider opencode --model minimax-2.5

# Anthropic (Claude)
python3 main.py --openai --provider anthropic --model claude-3-5-sonnet-20241022

# Ollama (本地)
python3 main.py --openai --provider ollama --model llama3.2

# Groq (高速)
python3 main.py --openai --provider groq --model llama-3.1-70b-versatile

# Azure OpenAI
python3 main.py --openai --provider azure --model gpt-4o
```

### 功能选项

```bash
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

# 自定义生成参数
python3 main.py --openai --max-tokens 4096 --temperature 0.5

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
| `models` | 显示支持的 LLM |

## .env 配置示例

```bash
# OpenAI
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o

# OpenCode (MiniMax)
OPENCODE_API_KEY=xxx
OPENCODE_BASE_URL=https://api.opencode.ai/v1
OPENCODE_MODEL=minimax-2.5

# Anthropic
ANTHROPIC_API_KEY=sk-ant-xxx
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Azure
AZURE_API_KEY=xxx
AZURE_ENDPOINT=https://xxx.openai.azure.com
AZURE_MODEL=gpt-4o

# Ollama (本地)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# Groq
GROQ_API_KEY=gsk_xxx
GROQ_MODEL=llama-3.1-70b-versatile

# Together AI
TOGETHER_API_KEY=xxx
TOGETHER_MODEL=meta-llama/Llama-3.2-70B-Instruct-Turbo

# vLLM (本地)
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct

# 全局设置
LLM_PROVIDER=openai
LLM_API_MODE=auto
LLM_MAX_TOKENS=2048
LLM_TEMPERATURE=0.7
```

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
│   └── llm.py                # LLM 接口 (多模型支持)
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
│  支持: OpenAI / Claude / Ollama / Groq 等                    │
└─────────────────────────────────────────────────────────────┘
```

## 配置优先级

```
CLI 参数 > .env 环境变量 > 代码默认值
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

### 使用不同的 LLM Provider

```python
from agentic_rag_demo import build_llm

# OpenAI
llm = build_llm(use_openai=True, provider="openai", model="gpt-4o")

# Claude
llm = build_llm(use_openai=True, provider="anthropic", model="claude-3-5-sonnet-20241022")

# Ollama
llm = build_llm(use_openai=True, provider="ollama", model="llama3.2")

# Groq
llm = build_llm(use_openai=True, provider="groq", model="llama-3.1-70b-versatile")

# 查看支持的模型
from agentic_rag_demo import list_providers
print(list_providers())
```

### 自定义 Embedding 模型

```python
from agentic_rag_demo import build_embedding_model

embed = build_embedding_model(provider="openai", model_name="text-embedding-3-small")
embed = build_embedding_model(provider="huggingface", model_name="BAAI/bge-small-zh-v1.5")
```

### 使用不同的文档解析器

```python
from agentic_rag_demo import build_parser, parse_document_with, list_parsers

# 查看支持的解析器
print(list_parsers())

# 使用内置解析器 (默认)
chunks = parse_document_with("data/report.pdf", parser_type="native")

# 使用 LlamaIndex 解析器
chunks = parse_document_with(
    "data/report.pdf",
    parser_type="llamaindex",
    chunk_size=512,
    chunk_overlap=50,
)

# 使用 MinerU 解析器 (优秀的 PDF 解析)
chunks = parse_document_with(
    "data/report.pdf",
    parser_type="mineru",
    parse_mode="hybrid",  # hybrid/ocr/txt
    extract_tables=True,
)

# 使用 LangChain 解析器
chunks = parse_document_with(
    "data/report.docx",
    parser_type="langchain",
    chunk_size=500,
)

# 直接创建解析器
parser = build_parser("mineru", parse_mode="ocr")
documents = parser.parse("data/report.pdf")
```

### 批量评估

```python
from agentic_rag_demo import BatchEvaluator, Evaluator

evaluator = BatchEvaluator(Evaluator())
results = evaluator.evaluate_batch(test_cases)
evaluator.print_summary(results)
```

## 统一配置管理

### 使用 Settings 类

```python
from agentic_rag_demo import Settings, get_settings

# 获取全局配置
settings = get_settings()
print(f"LLM: {settings.llm.provider}")
print(f"Parser: {settings.parser.parser_type}")

# 从环境变量加载
settings = Settings.from_env()

# 从 YAML 文件加载
settings = Settings.from_yaml("config.yaml")

# 从 JSON 文件加载
settings = Settings.from_json("config.json")

# 保存配置
settings.to_yaml("my_config.yaml")
```

### 环境变量配置

```bash
# LLM 配置
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o
export LLM_API_KEY=sk-xxx
export LLM_MAX_TOKENS=4096

# Embedding 配置
export EMBEDDING_PROVIDER=huggingface
export EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5

# 解析器配置
export PARSER_TYPE=mineru
export PARSER_CHUNK_SIZE=512

# 检索配置
export RETRIEVAL_TOP_K=5
export RETRIEVAL_RERANKER=true

# Agent 配置
export AGENT_USE_LOOP=true
export AGENT_MAX_ITERATIONS=10

# 缓存配置
export CACHE_ENABLED=true
export CACHE_TYPE=lru
```

### 配置文件模板

```python
from agentic_rag_demo import create_config_template

# 创建配置模板
create_config_template("config.yaml")
```

生成 `config.yaml`:

```yaml
project_name: Agentic RAG Demo
version: 0.3.0
llm:
  provider: openai
  model: gpt-4o
  max_tokens: 2048
embedding:
  provider: auto
parser:
  parser_type: native
  chunk_size: 280
retrieval:
  top_k: 5
  enable_reranker: false
agent:
  max_iterations: 5
  use_agent_loop: false
cache:
  enabled: false
  cache_type: lru
```