from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    title: str
    content: str


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
