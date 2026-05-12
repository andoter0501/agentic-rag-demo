from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any


class EmbeddingModel(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError


class OpenAIEmbedding(EmbeddingModel):
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 100,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("请安装 openai: pip install openai")

        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )
        self.model = model
        self.batch_size = batch_size
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = self._get_dimension()
        return self._dimension

    def _get_dimension(self) -> int:
        dim_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dim_map.get(self.model, 1536)

    def embed(self, text: str) -> list[float]:
        results = self.embed_batch([text])
        return results[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            embeddings.extend([e.embedding for e in response.data])

        return embeddings


class LocalEmbedding(EmbeddingModel):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            self.model = None
            self.normalize = normalize
            self._dimension = 384
            return

        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = normalize

    @property
    def dimension(self) -> int:
        if self.model is not None:
            return self.model.get_sentence_embedding_dimension()
        return self._dimension

    def embed(self, text: str) -> list[float]:
        if self.model is None:
            return self._simple_embed(text)
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.model is None:
            return [self._simple_embed(t) for t in texts]
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def _simple_embed(self, text: str) -> list[float]:
        import hashlib

        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        return [((hash_val >> (i * 4)) & 0xFF) / 255.0 for i in range(min(self._dimension, 32))] + [
            0.0
        ] * max(0, self._dimension - 32)


class HuggingFaceEmbedding(EmbeddingModel):
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        device: str = "cpu",
        normalize: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            self.model = None
            self.tokenizer = None
            self.device = None
            self.normalize = normalize
            self._dimension = 768
            return

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.normalize = normalize

    @property
    def dimension(self) -> int:
        if self.model is not None:
            return self.model.config.hidden_size
        return self._dimension

    def embed(self, text: str) -> list[float]:
        if self.model is None or self.tokenizer is None:
            return self._simple_embed(text)
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.model is None or self.tokenizer is None:
            return [self._simple_embed(t) for t in texts]

        import torch

        all_embeddings: list[list[float]] = []
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)
                embeddings = outputs.last_hidden_state[:, 0]

                if self.normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.extend(embeddings.cpu().numpy().tolist())

        return all_embeddings

    def _simple_embed(self, text: str) -> list[float]:
        import hashlib

        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        return [((hash_val >> (i * 4)) & 0xFF) / 255.0 for i in range(min(self._dimension, 32))] + [
            0.0
        ] * max(0, self._dimension - 32)


def build_embedding_model(
    provider: str = "auto",
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    device: str = "cpu",
) -> EmbeddingModel:
    if provider == "openai":
        return OpenAIEmbedding(
            model=model_name or "text-embedding-3-small",
            api_key=api_key,
            base_url=base_url,
        )
    elif provider == "huggingface":
        return HuggingFaceEmbedding(
            model_name=model_name or "BAAI/bge-small-zh-v1.5",
            device=device,
        )
    else:
        if os.getenv("OPENAI_API_KEY"):
            return OpenAIEmbedding(api_key=api_key, base_url=base_url)
        return LocalEmbedding(model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2")
