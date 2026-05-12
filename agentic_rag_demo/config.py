"""统一配置管理"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

try:
    from pydantic import BaseModel, Field
    from pydantic_settings import BaseSettings, SettingsConfigDict

    HAS_PYDANTIC = True
except ImportError:
    try:
        from pydantic import BaseModel, Field

        HAS_PYDANTIC = False
        BaseSettings = BaseModel
        SettingsConfigDict = {}
    except ImportError:
        BaseModel = object
        Field = lambda default, **kw: default
        HAS_PYDANTIC = False
        BaseSettings = object
        SettingsConfigDict = {}


LLMProvider = Literal[
    "openai", "opencode", "anthropic", "azure", "ollama", "groq", "together", "vllm"
]
EmbeddingProvider = Literal["auto", "openai", "huggingface"]
ParserType = Literal["native", "llamaindex", "langchain", "mineru"]
FusionMethod = Literal["rrf", "weighted", "convex"]
CacheType = Literal["lru", "ttl", "persistent"]


@dataclass
class LLMConfig:
    provider: LLMProvider = "openai"
    model: str = "gpt-4o"
    api_key: str | None = None
    base_url: str | None = None
    api_mode: Literal["auto", "responses", "chat"] = "auto"
    max_tokens: int = 2048
    temperature: float = 0.7


@dataclass
class EmbeddingConfig:
    provider: EmbeddingProvider = "auto"
    model: str | None = None
    dimension: int = 384
    batch_size: int = 100


@dataclass
class ParserConfig:
    parser_type: ParserType = "native"
    chunk_size: int = 280
    chunk_overlap: int = 50
    parse_mode: Literal["hybrid", "ocr", "txt"] = "hybrid"
    extract_images: bool = False
    extract_tables: bool = True


@dataclass
class RetrievalConfig:
    top_k: int = 5
    sparse_weight: float = 0.4
    dense_weight: float = 0.6
    fusion_method: FusionMethod = "rrf"
    enable_reranker: bool = False
    reranker_type: Literal["cross_encoder", "sentence_transformer", "llm_coherence"] = (
        "cross_encoder"
    )
    rerank_top_k: int = 3


@dataclass
class AgentConfig:
    max_iterations: int = 5
    use_agent_loop: bool = False
    enable_reflection: bool = True
    query_rewrite: bool = True
    query_variants: int = 2


@dataclass
class CacheConfig:
    enabled: bool = False
    cache_type: CacheType = "lru"
    max_size: int = 1000
    default_ttl: int = 3600
    cache_dir: str | None = None


@dataclass
class EvaluatorConfig:
    enabled: bool = False
    metrics: list[str] = field(default_factory=lambda: ["hit_rate", "mrr", "ndcg", "faithfulness"])


@dataclass
class Settings:
    project_name: str = "Agentic RAG Demo"
    version: str = "0.3.0"
    debug: bool = False
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    parser: ParserConfig = field(default_factory=ParserConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)

    @classmethod
    def from_env(cls) -> Settings:
        """从环境变量加载配置"""
        return cls(
            llm=LLMConfig(
                provider=os.getenv("LLM_PROVIDER", "openai"),
                model=os.getenv("LLM_MODEL", "gpt-4o"),
                api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
                api_mode=os.getenv("LLM_API_MODE", "auto"),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            ),
            embedding=EmbeddingConfig(
                provider=os.getenv("EMBEDDING_PROVIDER", "auto"),
                model=os.getenv("EMBEDDING_MODEL"),
            ),
            parser=ParserConfig(
                parser_type=os.getenv("PARSER_TYPE", "native"),
                chunk_size=int(os.getenv("PARSER_CHUNK_SIZE", "280")),
                chunk_overlap=int(os.getenv("PARSER_CHUNK_OVERLAP", "50")),
            ),
            retrieval=RetrievalConfig(
                top_k=int(os.getenv("RETRIEVAL_TOP_K", "5")),
                enable_reranker=os.getenv("RETRIEVAL_RERANKER", "false").lower() == "true",
            ),
            agent=AgentConfig(
                max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "5")),
                use_agent_loop=os.getenv("AGENT_USE_LOOP", "false").lower() == "true",
            ),
            cache=CacheConfig(
                enabled=os.getenv("CACHE_ENABLED", "false").lower() == "true",
                cache_type=os.getenv("CACHE_TYPE", "lru"),
            ),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> Settings:
        """从 YAML 文件加载配置"""
        try:
            import yaml

            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return cls(**data) if data else cls()
        except ImportError:
            return cls.from_json(path.with_suffix(".json"))

    @classmethod
    def from_json(cls, path: str | Path) -> Settings:
        """从 JSON 文件加载配置"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data) if data else cls()

    def to_yaml(self, path: str | Path) -> None:
        """保存配置到 YAML 文件"""
        try:
            import yaml

            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        except ImportError:
            self.to_json(path.with_suffix(".json"))

    def to_json(self, path: str | Path) -> None:
        """保存配置到 JSON 文件"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "project_name": self.project_name,
            "version": self.version,
            "debug": self.debug,
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "api_key": self.llm.api_key,
                "base_url": self.llm.base_url,
                "api_mode": self.llm.api_mode,
                "max_tokens": self.llm.max_tokens,
                "temperature": self.llm.temperature,
            },
            "embedding": {
                "provider": self.embedding.provider,
                "model": self.embedding.model,
                "dimension": self.embedding.dimension,
            },
            "parser": {
                "parser_type": self.parser.parser_type,
                "chunk_size": self.parser.chunk_size,
                "chunk_overlap": self.parser.chunk_overlap,
            },
            "retrieval": {
                "top_k": self.retrieval.top_k,
                "enable_reranker": self.retrieval.enable_reranker,
            },
            "agent": {
                "max_iterations": self.agent.max_iterations,
                "use_agent_loop": self.agent.use_agent_loop,
            },
            "cache": {
                "enabled": self.cache.enabled,
                "cache_type": self.cache.cache_type,
            },
        }


@lru_cache
def get_settings() -> Settings:
    """获取全局配置实例 (单例)"""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    return Settings.from_env()


def reload_settings() -> Settings:
    """重新加载配置"""
    get_settings.cache_clear()
    return get_settings()


def create_config_template(path: str | Path = "config.yaml") -> None:
    """创建配置模板文件"""
    settings = Settings()
    settings.to_yaml(path)
    print(f"配置模板已创建: {path}")
    print(f"可使用以下环境变量覆盖配置:")
    print("  LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, ...")
    print("  EMBEDDING_PROVIDER, EMBEDDING_MODEL, ...")
    print("  PARSER_TYPE, PARSER_CHUNK_SIZE, ...")
    print("  RETRIEVAL_TOP_K, RETRIEVAL_RERANKER, ...")
    print("  AGENT_MAX_ITERATIONS, AGENT_USE_LOOP, ...")
    print("  CACHE_ENABLED, CACHE_TYPE, ...")


__all__ = [
    "Settings",
    "LLMConfig",
    "EmbeddingConfig",
    "ParserConfig",
    "RetrievalConfig",
    "AgentConfig",
    "CacheConfig",
    "EvaluatorConfig",
    "get_settings",
    "reload_settings",
    "create_config_template",
    "LLMProvider",
    "EmbeddingProvider",
    "ParserType",
    "FusionMethod",
    "CacheType",
]
