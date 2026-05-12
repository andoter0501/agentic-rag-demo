from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .document_parser_core import (
    Document,
    ParsedChunk,
    DocumentParser,
    MarkdownParser,
    PDFParser,
    HTMLParser,
    JSONParser,
    TXTParser,
    DocxParser,
    get_parser_for_file,
    parse_document,
    parse_documents,
)


class LlamaIndexParser(DocumentParser):
    def __init__(
        self,
        chunk_size: int = 280,
        chunk_overlap: int = 50,
        llm: Any | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm = llm
        self._loader: Any | None = None
        self._splitter: Any | None = None

    def _ensure_llama_index(self) -> None:
        if self._loader is not None:
            return
        try:
            from llama_index.core import SimpleDirectoryReader
            from llama_index.core.node_parser import SentenceSplitter

            self._loader = SimpleDirectoryReader
            self._splitter = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        except ImportError:
            raise RuntimeError("请安装 llama-index: pip install llama-index")

    def parse(self, path: str | Path) -> list[Document]:
        self._ensure_llama_index()

        p = Path(path)
        try:
            from llama_index.core import SimpleDirectoryReader

            reader = SimpleDirectoryReader(input_files=[str(p)])
            documents = reader.load_data()

            return [
                Document(
                    content=doc.text,
                    metadata={**doc.metadata, "source": str(p)},
                    source=str(p),
                )
                for doc in documents
            ]
        except Exception as exc:
            raise RuntimeError(f"LlamaIndex 解析失败: {exc}")

    def extract_chunks(
        self,
        document: Document,
        max_chars: int = 280,
        overlap: int = 50,
    ) -> list[ParsedChunk]:
        try:
            from llama_index.core.node_parser import SentenceSplitter

            splitter = SentenceSplitter(chunk_size=max_chars, chunk_overlap=overlap)
            nodes = splitter.get_nodes_from_documents(
                [{"text": document.content, "metadata": document.metadata}]
            )

            return [
                ParsedChunk(
                    content=node.text,
                    metadata=node.metadata,
                    chunk_id=node.id_,
                )
                for node in nodes
            ]
        except Exception:
            return self._fallback_chunking(document, max_chars, overlap)

    def _fallback_chunking(
        self,
        document: Document,
        max_chars: int,
        overlap: int,
    ) -> list[ParsedChunk]:
        paragraphs = [p.strip() for p in document.content.split("\n") if p.strip()]
        chunks: list[ParsedChunk] = []
        bucket = ""

        for idx, para in enumerate(paragraphs, start=1):
            candidate = f"{bucket}\n{para}" if bucket else para
            if len(candidate) > max_chars and bucket:
                chunks.append(
                    ParsedChunk(
                        content=bucket,
                        metadata={**document.metadata, "chunk_index": len(chunks) + 1},
                        chunk_id=f"chunk-{len(chunks) + 1}",
                    )
                )
                bucket = para[-overlap:] if overlap > 0 and len(para) > overlap else para
            else:
                bucket = candidate

        if bucket:
            chunks.append(
                ParsedChunk(
                    content=bucket,
                    metadata={**document.metadata, "chunk_index": len(chunks) + 1},
                    chunk_id=f"chunk-{len(chunks) + 1}",
                )
            )

        return chunks


class LangChainParser(DocumentParser):
    def __init__(
        self,
        chunk_size: int = 280,
        chunk_overlap: int = 50,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _ensure_langchain(self) -> bool:
        try:
            from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            return True
        except ImportError:
            return False

    def parse(self, path: str | Path) -> list[Document]:
        if not self._ensure_langchain():
            raise RuntimeError("请安装 langchain: pip install langchain langchain-community")

        p = Path(path)
        suffix = p.suffix.lower()

        try:
            if suffix == ".pdf":
                from langchain_community.document_loaders import PyPDFLoader

                loader = PyPDFLoader(str(p))
            elif suffix in {".html", ".htm"}:
                from langchain_community.document_loaders import UnstructuredHTMLLoader

                loader = UnstructuredHTMLLoader(str(p))
            elif suffix == ".docx":
                from langchain_community.document_loaders import UnstructuredWordDocumentLoader

                loader = UnstructuredWordDocumentLoader(str(p))
            elif suffix == ".txt":
                from langchain_community.document_loaders import TextLoader

                loader = TextLoader(str(p), encoding="utf-8")
            else:
                from langchain_community.document_loaders import TextLoader

                loader = TextLoader(str(p), encoding="utf-8")

            docs = loader.load()
            return [
                Document(
                    content=doc.page_content,
                    metadata={**doc.metadata, "source": str(p)},
                    source=str(p),
                )
                for doc in docs
            ]
        except Exception as exc:
            raise RuntimeError(f"LangChain 解析失败: {exc}")

    def extract_chunks(
        self,
        document: Document,
        max_chars: int = 280,
        overlap: int = 50,
    ) -> list[ParsedChunk]:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_chars,
                chunk_overlap=overlap,
                separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " "],
            )

            from langchain_core.documents import Document as LCDocument

            lc_doc = LCDocument(page_content=document.content, metadata=document.metadata)
            chunks = splitter.split_documents([lc_doc])

            return [
                ParsedChunk(
                    content=chunk.page_content,
                    metadata=chunk.metadata,
                    chunk_id=f"chunk-{i}",
                )
                for i, chunk in enumerate(chunks)
            ]
        except Exception:
            return self._fallback_chunking(document, max_chars, overlap)

    def _fallback_chunking(
        self,
        document: Document,
        max_chars: int,
        overlap: int,
    ) -> list[ParsedChunk]:
        paragraphs = [p.strip() for p in document.content.split("\n") if p.strip()]
        chunks: list[ParsedChunk] = []
        bucket = ""

        for idx, para in enumerate(paragraphs, start=1):
            candidate = f"{bucket}\n{para}" if bucket else para
            if len(candidate) > max_chars and bucket:
                chunks.append(
                    ParsedChunk(
                        content=bucket,
                        metadata={**document.metadata, "chunk_index": len(chunks) + 1},
                        chunk_id=f"chunk-{len(chunks) + 1}",
                    )
                )
                bucket = para[-overlap:] if overlap > 0 and len(para) > overlap else para
            else:
                bucket = candidate

        if bucket:
            chunks.append(
                ParsedChunk(
                    content=bucket,
                    metadata={**document.metadata, "chunk_index": len(chunks) + 1},
                    chunk_id=f"chunk-{len(chunks) + 1}",
                )
            )

        return chunks


class MinerUParser(DocumentParser):
    def __init__(
        self,
        parse_mode: Literal["hybrid", "ocr", "txt"] = "hybrid",
        extract_images: bool = False,
        extract_tables: bool = True,
    ) -> None:
        self.parse_mode = parse_mode
        self.extract_images = extract_images
        self.extract_tables = extract_tables

    def _ensure_mineru(self) -> bool:
        try:
            import mineru

            return True
        except ImportError:
            return False

    def parse(self, path: str | Path) -> list[Document]:
        if not self._ensure_mineru():
            raise RuntimeError("请安装 mineru: pip install mineru")

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"文件不存在: {p}")

        try:
            import mineru

            model_list = mineru.list_models()
            available_models = [m for m in model_list if "pdf" in m.lower() or "doc" in m.lower()]

            if not available_models:
                model_name = "DocLayoutAnalysis"
            else:
                model_name = available_models[0]

            client = mineru.init_model(model_name, model_type="layout")

            result = mineru.parse_pdf(
                file_path=str(p),
                parse_mode=self.parse_mode,
                extract_images=self.extract_images,
                extract_tables=self.extract_tables,
            )

            if isinstance(result, dict):
                pages = result.get("pages", [])
            elif hasattr(result, "pages"):
                pages = result.pages
            else:
                pages = [result]

            documents: list[Document] = []
            for i, page in enumerate(pages):
                if isinstance(page, dict):
                    text = page.get("text", page.get("content", ""))
                    blocks = page.get("blocks", [])
                elif hasattr(page, "text"):
                    text = page.text
                    blocks = getattr(page, "blocks", [])
                else:
                    text = str(page)
                    blocks = []

                metadata = {
                    "source": str(p),
                    "page": i + 1,
                    "total_pages": len(pages),
                    "parse_mode": self.parse_mode,
                    "blocks_count": len(blocks),
                }

                if text.strip():
                    documents.append(
                        Document(
                            content=text,
                            metadata=metadata,
                            source=str(p),
                        )
                    )

            return documents

        except Exception as exc:
            raise RuntimeError(f"MinerU 解析失败: {exc}")

    def extract_chunks(
        self,
        document: Document,
        max_chars: int = 280,
        overlap: int = 50,
    ) -> list[ParsedChunk]:
        paragraphs = [p.strip() for p in document.content.split("\n\n") if p.strip()]
        chunks: list[ParsedChunk] = []
        bucket = ""
        page = document.metadata.get("page", 1)

        for idx, para in enumerate(paragraphs, start=1):
            candidate = f"{bucket}\n{para}" if bucket else para
            if len(candidate) > max_chars and bucket:
                chunks.append(
                    ParsedChunk(
                        content=bucket,
                        metadata={**document.metadata, "chunk_index": len(chunks) + 1},
                        chunk_id=f"page{page}-chunk-{len(chunks) + 1}",
                    )
                )
                bucket = para[-overlap:] if overlap > 0 and len(para) > overlap else para
            else:
                bucket = candidate

        if bucket:
            chunks.append(
                ParsedChunk(
                    content=bucket,
                    metadata={**document.metadata, "chunk_index": len(chunks) + 1},
                    chunk_id=f"page{page}-chunk-{len(chunks) + 1}",
                )
            )

        return chunks


def build_parser(
    parser_type: Literal["native", "llamaindex", "langchain", "mineru"] = "native",
    **kwargs: Any,
) -> DocumentParser:
    if parser_type == "native":
        suffix = kwargs.get("suffix", ".txt")
        return get_parser_for_file(suffix)

    elif parser_type == "llamaindex":
        return LlamaIndexParser(
            chunk_size=kwargs.get("chunk_size", 280),
            chunk_overlap=kwargs.get("chunk_overlap", 50),
            llm=kwargs.get("llm"),
        )

    elif parser_type == "langchain":
        return LangChainParser(
            chunk_size=kwargs.get("chunk_size", 280),
            chunk_overlap=kwargs.get("chunk_overlap", 50),
        )

    elif parser_type == "mineru":
        return MinerUParser(
            parse_mode=kwargs.get("parse_mode", "hybrid"),
            extract_images=kwargs.get("extract_images", False),
            extract_tables=kwargs.get("extract_tables", True),
        )

    else:
        raise ValueError(f"不支持的解析器类型: {parser_type}")


def parse_document_with(
    path: str | Path,
    parser_type: Literal["native", "llamaindex", "langchain", "mineru"] = "native",
    max_chars: int = 280,
    overlap: int = 50,
    **kwargs: Any,
) -> list[ParsedChunk]:
    parser = build_parser(parser_type, **kwargs)
    documents = parser.parse(path)
    all_chunks: list[ParsedChunk] = []

    for doc in documents:
        chunks = parser.extract_chunks(doc, max_chars=max_chars, overlap=overlap)
        all_chunks.extend(chunks)

    return all_chunks


SUPPORTED_PARSERS = {
    "native": "内置解析器 (轻量，无额外依赖)",
    "llamaindex": "LlamaIndex 解析器 (功能丰富)",
    "langchain": "LangChain 解析器 (生态完整)",
    "mineru": "MinerU 解析器 (优秀 PDF/文档解析)",
}


def list_parsers() -> dict[str, str]:
    return SUPPORTED_PARSERS


__all__ = [
    "Document",
    "ParsedChunk",
    "DocumentParser",
    "MarkdownParser",
    "PDFParser",
    "HTMLParser",
    "JSONParser",
    "TXTParser",
    "DocxParser",
    "LlamaIndexParser",
    "LangChainParser",
    "MinerUParser",
    "get_parser_for_file",
    "parse_document",
    "parse_documents",
    "build_parser",
    "parse_document_with",
    "list_parsers",
    "SUPPORTED_PARSERS",
]
