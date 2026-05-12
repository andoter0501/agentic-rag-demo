# Agentic RAG Demo

一个最小可运行的 `agentic RAG` 示例，特点：

- 本地知识库（`data/knowledge_base.md`）
- 文档切分 + 稀疏向量检索（TF-IDF + cosine）
- 简单代理流程：查询改写 -> 多次检索 -> 去重聚合 -> 反思 -> 引用回答
- CLI 交互演示（含 trace）

## 目录

- `agentic_rag_demo/loader.py`: 文档读取与切分
- `agentic_rag_demo/retriever.py`: 检索器
- `agentic_rag_demo/agent.py`: Agentic RAG 核心流程
- `main.py`: 交互入口
- `tests/test_demo.py`: 基础测试

## 运行

```bash
python3 main.py
```

推荐：使用 `.env`（dotenv 自动加载）

```bash
. .venv/bin/activate
pip install openai python-dotenv
cp .env.example .env
# 编辑 .env，填入 OPENCODE_API_KEY / OPENCODE_BASE_URL / OPENCODE_MODEL 等配置
python3 main.py
```

显式使用 OpenAI：

```bash
. .venv/bin/activate
python3 main.py --openai --provider openai --model gpt-5.2 --api-mode auto
```

显式使用 Opencode（例如 MiniMax2.5）：

```bash
. .venv/bin/activate
python3 main.py --openai --provider opencode --model minimax-2.5 --api-mode auto
```

示例提问：

- `Agentic RAG 和 RAG 有什么区别？`
- `Agentic RAG 的评估指标有哪些？`

退出输入：`q`

## 测试

```bash
. .venv/bin/activate
python -m pytest -q
```

说明：
- `--api-mode auto` 会优先走 `responses`，失败后自动回退 `chat/completions`。
- 具体模型 ID 以你在 Opencode 控制台可见名称为准，可替换 `minimax-2.5`。
- 配置优先级：`CLI 参数 > .env 环境变量 > 代码默认值`。
