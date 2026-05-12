from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkStrategy:
    max_chars: int = 280
    overlap: int = 50
    min_chars: int = 50


@dataclass
class SemanticChunk:
    content: str
    title: str
    chunk_id: str
    sentence_count: int = 0
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def _split_sections(markdown_text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_title = "未命名章节"
    buffer: list[str] = []

    for line in markdown_text.splitlines():
        if line.startswith("## "):
            if buffer:
                sections.append((current_title, "\n".join(buffer).strip()))
                buffer = []
            current_title = line[3:].strip()
            continue
        if line.startswith("# "):
            continue
        buffer.append(line)

    if buffer:
        sections.append((current_title, "\n".join(buffer).strip()))

    return [s for s in sections if s[1]]


def _split_by_sentences(text: str) -> list[str]:
    sentence_endings = re.compile(r"[。！？.!?\n]+")
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


class SemanticChunker:
    def __init__(
        self,
        max_chars: int = 500,
        overlap: int = 100,
        overlap_sentences: int = 2,
        merge_threshold: float = 0.6,
    ) -> None:
        self.max_chars = max_chars
        self.overlap = overlap
        self.overlap_sentences = overlap_sentences
        self.merge_threshold = merge_threshold

    def chunk(self, text: str, title: str = "default") -> list[SemanticChunk]:
        sentences = _split_by_sentences(text)
        if not sentences:
            return []

        chunks: list[SemanticChunk] = []
        current_bucket: list[str] = []
        current_length = 0

        for idx, sentence in enumerate(sentences):
            sentence_len = len(sentence)

            if current_length + sentence_len > self.max_chars and current_bucket:
                chunk_content = "".join(current_bucket)

                chunks.append(
                    SemanticChunk(
                        content=chunk_content,
                        title=title,
                        chunk_id=f"{title}-chunk-{len(chunks) + 1}",
                        sentence_count=len(current_bucket),
                        token_count=_estimate_tokens(chunk_content),
                    )
                )

                overlap_text = "".join(current_bucket[-self.overlap_sentences :])
                current_bucket = [overlap_text]
                current_length = len(overlap_text)
            else:
                current_bucket.append(sentence)
                current_length += sentence_len

        if current_bucket:
            chunk_content = "".join(current_bucket)
            chunks.append(
                SemanticChunk(
                    content=chunk_content,
                    title=title,
                    chunk_id=f"{title}-chunk-{len(chunks) + 1}",
                    sentence_count=len(current_bucket),
                    token_count=_estimate_tokens(chunk_content),
                )
            )

        return self._merge_similar_chunks(chunks)

    def _merge_similar_chunks(self, chunks: list[SemanticChunk]) -> list[SemanticChunk]:
        if len(chunks) <= 1:
            return chunks

        merged: list[SemanticChunk] = []
        current = chunks[0]

        for next_chunk in chunks[1:]:
            if self._should_merge(current, next_chunk):
                merged_content = current.content + next_chunk.content
                current = SemanticChunk(
                    content=merged_content,
                    title=current.title,
                    chunk_id=current.chunk_id,
                    sentence_count=current.sentence_count + next_chunk.sentence_count,
                    token_count=_estimate_tokens(merged_content),
                    metadata={**current.metadata, **next_chunk.metadata},
                )
            else:
                merged.append(current)
                current = next_chunk

        merged.append(current)
        return merged

    def _should_merge(self, chunk1: SemanticChunk, chunk2: SemanticChunk) -> bool:
        if len(chunk1.content) + len(chunk2.content) > self.max_chars * 1.5:
            return False

        common_words = set(chunk1.content.split()) & set(chunk2.content.split())
        total_words = set(chunk1.content.split()) | set(chunk2.content.split())

        if not total_words:
            return False

        overlap_ratio = len(common_words) / len(total_words)
        return overlap_ratio >= self.merge_threshold


class RecursiveChunker:
    def __init__(
        self,
        separators: list[str] | None = None,
        max_chars: int = 500,
        overlap: int = 50,
    ) -> None:
        self.separators = separators or [
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            ".",
            "!",
            "?",
            " ",
        ]
        self.max_chars = max_chars
        self.overlap = overlap

    def chunk(self, text: str, title: str = "default") -> list[SemanticChunk]:
        chunks: list[SemanticChunk] = []
        self._split_recursive(text, title, chunks, 0)
        return self._add_overlap(chunks)

    def _split_recursive(
        self,
        text: str,
        title: str,
        chunks: list[SemanticChunk],
        depth: int,
    ) -> None:
        if depth >= len(self.separators):
            if len(text) > self.max_chars:
                for i in range(0, len(text), self.max_chars - self.overlap):
                    chunk_text = text[i : i + self.max_chars]
                    if chunk_text.strip():
                        chunks.append(
                            SemanticChunk(
                                content=chunk_text,
                                title=title,
                                chunk_id=f"{title}-chunk-{len(chunks) + 1}",
                                token_count=_estimate_tokens(chunk_text),
                            )
                        )
            elif text.strip():
                chunks.append(
                    SemanticChunk(
                        content=text,
                        title=title,
                        chunk_id=f"{title}-chunk-{len(chunks) + 1}",
                        token_count=_estimate_tokens(text),
                    )
                )
            return

        separator = self.separators[depth]
        parts = text.split(separator)
        current_bucket = ""

        for part in parts:
            candidate = current_bucket + separator + part if current_bucket else part

            if len(candidate) > self.max_chars and current_bucket:
                chunks.append(
                    SemanticChunk(
                        content=current_bucket,
                        title=title,
                        chunk_id=f"{title}-chunk-{len(chunks) + 1}",
                        token_count=_estimate_tokens(current_bucket),
                    )
                )
                current_bucket = part
            else:
                current_bucket = candidate

        if current_bucket.strip():
            self._split_recursive(current_bucket, title, chunks, depth + 1)

    def _add_overlap(self, chunks: list[SemanticChunk]) -> list[SemanticChunk]:
        if len(chunks) <= 1 or self.overlap <= 0:
            return chunks

        result: list[SemanticChunk] = []

        for idx, chunk in enumerate(chunks):
            if idx > 0:
                prev_chunk = chunks[idx - 1]
                overlap_start = max(0, len(prev_chunk.content) - self.overlap)
                overlap_text = prev_chunk.content[overlap_start:]
                chunk = SemanticChunk(
                    content=overlap_text + chunk.content,
                    title=chunk.title,
                    chunk_id=chunk.chunk_id,
                    sentence_count=chunk.sentence_count,
                    token_count=chunk.token_count,
                    metadata={**chunk.metadata, "has_overlap": True},
                )
            result.append(chunk)

        return result


def load_chunks(path: str | Path, max_chars: int = 280) -> list[Chunk]:
    raw = Path(path).read_text(encoding="utf-8")
    sections = _split_sections(raw)
    chunks: list[Chunk] = []

    for title, content in sections:
        paragraphs = [p.strip() for p in content.split("\n") if p.strip()]
        merged: list[str] = []
        bucket = ""

        for para in paragraphs:
            candidate = f"{bucket}\n{para}".strip() if bucket else para
            if len(candidate) > max_chars and bucket:
                merged.append(bucket)
                bucket = para
            else:
                bucket = candidate

        if bucket:
            merged.append(bucket)

        for idx, item in enumerate(merged, start=1):
            chunk_id = f"{title}-{idx}"
            chunks.append(Chunk(chunk_id=chunk_id, title=title, content=item))

    return chunks


def load_semantic_chunks(
    path: str | Path,
    strategy: str = "semantic",
    max_chars: int = 500,
    overlap: int = 100,
) -> list[Chunk]:
    raw = Path(path).read_text(encoding="utf-8")
    sections = _split_sections(raw)
    chunks: list[Chunk] = []

    if strategy == "semantic":
        chunker = SemanticChunker(max_chars=max_chars, overlap=overlap)
    elif strategy == "recursive":
        chunker = RecursiveChunker(max_chars=max_chars, overlap=overlap)
    else:
        chunker = SemanticChunker(max_chars=max_chars, overlap=overlap)

    for title, content in sections:
        semantic_chunks = chunker.chunk(content, title=title)

        for sc in semantic_chunks:
            chunks.append(
                Chunk(
                    chunk_id=sc.chunk_id,
                    title=sc.title,
                    content=sc.content,
                    metadata={
                        "sentence_count": sc.sentence_count,
                        "token_count": sc.token_count,
                        "strategy": strategy,
                    },
                )
            )

    return chunks
