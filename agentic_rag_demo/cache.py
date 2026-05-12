from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class CacheEntry:
    key: str
    value: Any
    timestamp: float
    hit_count: int = 0
    ttl: float | None = None

    def is_expired(self, current_time: float | None = None) -> bool:
        if self.ttl is None:
            return False
        t = current_time or time.time()
        return (t - self.timestamp) > self.ttl

    def access(self) -> None:
        self.hit_count += 1


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate,
        }


class LRUCache:
    def __init__(
        self,
        max_size: int = 1000,
        ttl: float | None = None,
        name: str = "default",
    ) -> None:
        self.max_size = max_size
        self.ttl = ttl
        self.name = name
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats()

    def get(self, key: str) -> Any | None:
        if key not in self._cache:
            self._stats.misses += 1
            return None

        entry = self._cache[key]

        if entry.is_expired():
            self._evict(key)
            self._stats.misses += 1
            return None

        entry.access()
        self._cache.move_to_end(key)
        self._stats.hits += 1
        return entry.value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)

        entry_ttl = ttl if ttl is not None else self.ttl
        self._cache[key] = CacheEntry(
            key=key,
            value=value,
            timestamp=time.time(),
            ttl=entry_ttl,
        )

        if len(self._cache) > self.max_size:
            self._evict_lru()

    def _evict(self, key: str) -> None:
        if key in self._cache:
            del self._cache[key]
            self._stats.evictions += 1

    def _evict_lru(self) -> None:
        if self._cache:
            oldest_key = next(iter(self._cache))
            self._evict(oldest_key)

    def clear(self) -> None:
        self._cache.clear()

    @property
    def stats(self) -> CacheStats:
        return self._stats

    def __len__(self) -> int:
        return len(self._cache)


class TTLCache:
    def __init__(
        self,
        default_ttl: float = 3600,
        max_size: int = 10000,
    ) -> None:
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._cache: dict[str, CacheEntry] = {}
        self._stats = CacheStats()

    def get(self, key: str) -> Any | None:
        if key not in self._cache:
            self._stats.misses += 1
            return None

        entry = self._cache[key]

        if entry.is_expired():
            del self._cache[key]
            self._stats.misses += 1
            return None

        entry.access()
        self._stats.hits += 1
        return entry.value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        entry_ttl = ttl if ttl is not None else self.default_ttl
        self._cache[key] = CacheEntry(
            key=key,
            value=value,
            timestamp=time.time(),
            ttl=entry_ttl,
        )

        if len(self._cache) > self.max_size:
            self._cleanup_expired()
            if len(self._cache) > self.max_size:
                oldest = min(self._cache.values(), key=lambda e: e.timestamp)
                del self._cache[oldest.key]
                self._stats.evictions += 1

    def _cleanup_expired(self) -> None:
        current = time.time()
        expired_keys = [k for k, v in self._cache.items() if v.is_expired(current)]
        for key in expired_keys:
            del self._cache[key]

    def clear(self) -> None:
        self._cache.clear()

    @property
    def stats(self) -> CacheStats:
        return self._stats


class PersistentCache:
    def __init__(
        self,
        cache_dir: str | Path,
        max_size: int = 10000,
        default_ttl: float = 86400,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._memory_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats()
        self._load_index()

    def _load_index(self) -> None:
        index_file = self.cache_dir / "index.json"
        if index_file.exists():
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for key, entry_data in data.items():
                        if not entry_data.get("expired", False):
                            self._memory_cache[key] = CacheEntry(
                                key=key,
                                value=entry_data.get("value"),
                                timestamp=entry_data.get("timestamp", 0),
                                ttl=entry_data.get("ttl", self.default_ttl),
                                hit_count=entry_data.get("hit_count", 0),
                            )
            except (json.JSONDecodeError, IOError):
                pass

    def _save_index(self) -> None:
        index_file = self.cache_dir / "index.json"
        data = {}
        for key, entry in self._memory_cache.items():
            if not entry.is_expired():
                data[key] = {
                    "value": entry.value,
                    "timestamp": entry.timestamp,
                    "ttl": entry.ttl,
                    "hit_count": entry.hit_count,
                }
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def _get_cache_file(self, key: str) -> Path:
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.json"

    def get(self, key: str) -> Any | None:
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if entry.is_expired():
                self._evict(key)
                self._stats.misses += 1
                return None

            entry.access()
            self._memory_cache.move_to_end(key)
            self._stats.hits += 1
            return entry.value

        cache_file = self._get_cache_file(key)
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    entry_data = json.load(f)
                    entry = CacheEntry(
                        key=key,
                        value=entry_data.get("value"),
                        timestamp=entry_data.get("timestamp", 0),
                        ttl=entry_data.get("ttl", self.default_ttl),
                    )
                    if entry.is_expired():
                        cache_file.unlink()
                        self._stats.misses += 1
                        return None

                    self._memory_cache[key] = entry
                    self._memory_cache.move_to_end(key)
                    self._stats.hits += 1
                    return entry.value
            except (json.JSONDecodeError, IOError):
                self._stats.misses += 1
                return None

        self._stats.misses += 1
        return None

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        entry_ttl = ttl if ttl is not None else self.default_ttl
        self._memory_cache[key] = CacheEntry(
            key=key,
            value=value,
            timestamp=time.time(),
            ttl=entry_ttl,
        )
        self._memory_cache.move_to_end(key)

        cache_file = self._get_cache_file(key)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "key": key,
                    "value": value,
                    "timestamp": time.time(),
                    "ttl": entry_ttl,
                },
                f,
                ensure_ascii=False,
            )

        if len(self._memory_cache) > self.max_size:
            self._evict_lru()
            self._save_index()

    def _evict(self, key: str) -> None:
        if key in self._memory_cache:
            del self._memory_cache[key]
            cache_file = self._get_cache_file(key)
            if cache_file.exists():
                cache_file.unlink()
            self._stats.evictions += 1

    def _evict_lru(self) -> None:
        if self._memory_cache:
            oldest_key = next(iter(self._memory_cache))
            self._evict(oldest_key)

    def clear(self) -> None:
        self._memory_cache.clear()
        for f in self.cache_dir.glob("*.json"):
            if f.name != "index.json":
                f.unlink()

    @property
    def stats(self) -> CacheStats:
        return self._stats


class QueryCache:
    def __init__(
        self,
        cache_type: Literal["lru", "ttl", "persistent"] = "lru",
        max_size: int = 1000,
        ttl: float = 3600,
        cache_dir: str | None = None,
    ) -> None:
        self.cache_type = cache_type

        if cache_type == "lru":
            self._cache = LRUCache(max_size=max_size, ttl=ttl)
        elif cache_type == "ttl":
            self._cache = TTLCache(default_ttl=ttl, max_size=max_size)
        elif cache_type == "persistent":
            cache_path = cache_dir or "./.cache"
            self._cache = PersistentCache(
                cache_path, max_size=max_size, default_ttl=ttl
            )
        else:
            self._cache = LRUCache(max_size=max_size, ttl=ttl)

    @staticmethod
    def _normalize_query(query: str) -> str:
        normalized = query.lower().strip()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    @staticmethod
    def _hash_query(query: str) -> str:
        normalized = QueryCache._normalize_query(query)
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, query: str) -> Any | None:
        key = self._hash_query(query)
        return self._cache.get(key)

    def set(self, query: str, value: Any, ttl: float | None = None) -> None:
        key = self._hash_query(query)
        self._cache.set(key, value, ttl=ttl)

    def clear(self) -> None:
        self._cache.clear()

    @property
    def stats(self) -> CacheStats:
        return self._cache.stats


import re


class RAGCache:
    def __init__(
        self,
        query_cache: QueryCache | None = None,
        embedding_cache: QueryCache | None = None,
        retrieval_cache: QueryCache | None = None,
    ) -> None:
        self.query_cache = query_cache or QueryCache(cache_type="lru", max_size=1000)
        self.embedding_cache = embedding_cache or QueryCache(
            cache_type="lru", max_size=500
        )
        self.retrieval_cache = retrieval_cache or QueryCache(
            cache_type="ttl", max_size=2000
        )

    def get_cached_answer(self, query: str) -> Any | None:
        return self.query_cache.get(query)

    def cache_answer(self, query: str, answer: Any, ttl: float = 3600) -> None:
        self.query_cache.set(query, answer, ttl=ttl)

    def get_cached_embedding(self, text: str) -> Any | None:
        return self.embedding_cache.get(text)

    def cache_embedding(
        self, text: str, embedding: list[float], ttl: float = 86400
    ) -> None:
        self.embedding_cache.set(text, embedding, ttl=ttl)

    def get_cached_retrieval(self, query: str) -> Any | None:
        return self.retrieval_cache.get(query)

    def cache_retrieval(self, query: str, results: Any, ttl: float = 1800) -> None:
        self.retrieval_cache.set(query, results, ttl=ttl)

    def clear_all(self) -> None:
        self.query_cache.clear()
        self.embedding_cache.clear()
        self.retrieval_cache.clear()

    def print_stats(self) -> None:
        print("\n=== RAG Cache Statistics ===")
        print(
            f"Query Cache:      hits={self.query_cache.stats.hits}, misses={self.query_cache.stats.misses}, hit_rate={self.query_cache.stats.hit_rate:.2%}"
        )
        print(
            f"Embedding Cache:  hits={self.embedding_cache.stats.hits}, misses={self.embedding_cache.stats.misses}, hit_rate={self.embedding_cache.stats.hit_rate:.2%}"
        )
        print(
            f"Retrieval Cache:  hits={self.retrieval_cache.stats.hits}, misses={self.retrieval_cache.stats.misses}, hit_rate={self.retrieval_cache.stats.hit_rate:.2%}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_cache": self.query_cache.stats.to_dict(),
            "embedding_cache": self.embedding_cache.stats.to_dict(),
            "retrieval_cache": self.retrieval_cache.stats.to_dict(),
        }


def build_cache(
    cache_type: Literal["lru", "ttl", "persistent"] = "lru",
    max_size: int = 1000,
    ttl: float = 3600,
    cache_dir: str | None = None,
) -> RAGCache:
    return RAGCache(
        query_cache=QueryCache(cache_type=cache_type, max_size=max_size, ttl=ttl),
        embedding_cache=QueryCache(
            cache_type=cache_type, max_size=max_size // 2, ttl=ttl * 24
        ),
        retrieval_cache=QueryCache(
            cache_type="ttl", max_size=max_size * 2, ttl=ttl / 2
        ),
    )
