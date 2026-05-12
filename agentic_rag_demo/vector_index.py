from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False


class VectorIndex:
    def __init__(
        self,
        dimension: int,
        index_path: str | Path | None = None,
    ) -> None:
        self.dimension = dimension
        self.index_path = Path(index_path) if index_path else None
        self._index: Any | None = None
        self._id_map: dict[int, str] = {}
        self._reverse_map: dict[str, int] = {}

    @property
    def index(self) -> Any:
        if self._index is None:
            self._load_or_create()
        return self._index

    def _load_or_create(self) -> None:
        if self.index_path and self.index_path.exists():
            self._load()
        else:
            self._create()

    def _create(self) -> None:
        try:
            import faiss

            self._index = faiss.IndexFlatL2(self.dimension)
        except ImportError:
            if NUMPY_AVAILABLE:
                self._index = _NumpyIndex(self.dimension)
            else:
                self._index = _SimpleIndex()

    def _load(self) -> None:
        try:
            import faiss
        except ImportError:
            raise RuntimeError("请安装 faiss: pip install faiss-cpu 或 faiss-gpu")

        index_file = self.index_path / "index.faiss"
        id_map_file = self.index_path / "id_map.json"

        if index_file.exists():
            self._index = faiss.read_index(str(index_file))

        if id_map_file.exists():
            with open(id_map_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._id_map = {int(k): v for k, v in data["id_map"].items()}
                self._reverse_map = data["reverse_map"]

    def save(self, path: str | Path | None = None) -> None:
        target = Path(path) if path else self.index_path
        if not target:
            raise ValueError("必须指定保存路径")

        target.mkdir(parents=True, exist_ok=True)

        try:
            import faiss

            faiss.write_index(self._index, str(target / "index.faiss"))
        except (ImportError, AttributeError):
            if hasattr(self._index, "save"):
                with open(target / "index.npy", "wb") as f:
                    np.save(f, self._index.vectors)
                with open(target / "ids.pkl", "wb") as f:
                    pickle.dump(list(self._id_map.values()), f)

        id_map_file = target / "id_map.json"
        with open(id_map_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "id_map": {str(k): v for k, v in self._id_map.items()},
                    "reverse_map": self._reverse_map,
                    "dimension": self.dimension,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def add(self, chunk_ids: list[str], embeddings: list[list[float]]) -> None:
        if not chunk_ids or not embeddings:
            return

        if NUMPY_AVAILABLE and np is not None:
            vectors = np.array(embeddings, dtype=np.float32)
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            self.index.add(vectors)
        else:
            self.index.add(embeddings)

        start_id = len(self._id_map)
        for i, chunk_id in enumerate(chunk_ids):
            self._id_map[start_id + i] = chunk_id
            self._reverse_map[chunk_id] = start_id + i

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        if not self._index:
            return []

        try:
            if NUMPY_AVAILABLE and np is not None:
                query_vec = np.array([query_embedding], dtype=np.float32)
                distances, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))
            else:
                indices, distances = self.index.search(
                    query_embedding, min(top_k, self.index.ntotal)
                )
        except Exception:
            return []

        results: list[tuple[str, float]] = []

        if NUMPY_AVAILABLE and np is not None:
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0:
                    continue
                chunk_id = self._id_map.get(int(idx))
                if chunk_id:
                    score = float(1.0 / (1.0 + dist))
                    results.append((chunk_id, score))
        else:
            for idx, dist in zip(indices, distances):
                if idx < 0:
                    continue
                chunk_id = self._id_map.get(int(idx))
                if chunk_id:
                    score = float(1.0 / (1.0 + dist))
                    results.append((chunk_id, score))

        return results

    def __len__(self) -> int:
        return self.index.ntotal if hasattr(self.index, "ntotal") else len(self._id_map)


class _NumpyIndex:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.vectors: list[list[float]] = []

    def add(self, vectors: Any) -> None:
        if NUMPY_AVAILABLE and np is not None:
            self.vectors.extend(np.array(vectors).tolist())
        else:
            self.vectors.extend(vectors)

    def search(self, query: Any, k: int) -> tuple[Any, Any]:
        if NUMPY_AVAILABLE and np is not None:
            vectors = np.array(self.vectors)
            if len(vectors) == 0:
                return np.array([[-1] * k]), np.array([[float("inf")] * k])

            distances = np.linalg.norm(vectors - query, axis=1)
            indices = np.argsort(distances)[:k]

            return indices.reshape(1, -1), distances[indices].reshape(1, -1)
        return [-1] * k, [float("inf")] * k

    @property
    def ntotal(self) -> int:
        return len(self.vectors)


class _SimpleIndex:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.vectors: list[list[float]] = []

    def add(self, vectors: list[list[float]]) -> None:
        self.vectors.extend(vectors)

    def search(self, query: list[float], k: int) -> tuple[list[int], list[float]]:
        if not self.vectors:
            return [-1] * k, [float("inf")] * k

        distances: list[tuple[int, float]] = []
        for i, vec in enumerate(self.vectors):
            dist = sum((q - v) ** 2 for q, v in zip(query, vec))
            distances.append((i, dist))

        distances.sort(key=lambda x: x[1])
        top_k = distances[:k]

        indices = [idx for idx, _ in top_k]
        dists = [dist for _, dist in top_k]

        return indices, dists

    @property
    def ntotal(self) -> int:
        return len(self.vectors)
