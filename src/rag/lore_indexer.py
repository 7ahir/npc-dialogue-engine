"""Lore document ingestion pipeline: chunk, embed, store in ChromaDB."""

from pathlib import Path

import chromadb

from src.rag.embeddings import EmbeddingService
from src.utils.config import get_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[str]:
    """Split text into overlapping chunks at paragraph boundaries.

    Prefers splitting at paragraph breaks (double newline), then single
    newlines, then sentences. Falls back to hard character split.
    """
    separators = ["\n\n", "\n", ". "]
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Find the best split point near the chunk boundary
        best_split = end
        for sep in separators:
            # Look for separator in the last 25% of the chunk
            search_start = start + int(chunk_size * 0.75)
            pos = text.rfind(sep, search_start, end)
            if pos != -1:
                best_split = pos + len(sep)
                break

        chunk = text[start:best_split].strip()
        if chunk:
            chunks.append(chunk)

        start = best_split - chunk_overlap

    return chunks


def _extract_metadata(file_path: Path, chunk_text: str) -> dict[str, str]:
    """Extract metadata from chunk context."""
    # Find the most recent heading above or within this chunk
    section_title = ""
    for line in chunk_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            section_title = stripped.lstrip("#").strip()

    return {
        "source_file": file_path.name,
        "source_stem": file_path.stem,
        "section_title": section_title,
    }


class LoreIndexer:
    """Indexes lore documents into ChromaDB for RAG retrieval."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        chroma_client: chromadb.ClientAPI | None = None,
    ) -> None:
        self.config = get_config()
        self.embedding_service = embedding_service or EmbeddingService()

        if chroma_client is not None:
            self._client = chroma_client
        else:
            self._client = chromadb.PersistentClient(
                path=str(self.config.rag.chroma_persist_dir)
            )

    def index_directory(self, lore_dir: Path | None = None) -> int:
        """Index all markdown files in the lore directory.

        Returns the total number of chunks indexed.
        """
        lore_dir = lore_dir or self.config.rag.lore_dir
        collection = self._client.get_or_create_collection(
            name=self.config.rag.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Clear existing data for re-indexing
        existing = collection.count()
        if existing > 0:
            logger.info("clearing_existing_index", count=existing)
            all_ids = collection.get()["ids"]
            if all_ids:
                collection.delete(ids=all_ids)

        total_chunks = 0
        md_files = sorted(lore_dir.glob("*.md"))
        logger.info("indexing_lore_files", count=len(md_files))

        for file_path in md_files:
            text = file_path.read_text(encoding="utf-8")
            chunks = _chunk_text(
                text,
                chunk_size=self.config.rag.chunk_size,
                chunk_overlap=self.config.rag.chunk_overlap,
            )

            if not chunks:
                continue

            ids = [f"{file_path.stem}_{i}" for i in range(len(chunks))]
            metadatas = [_extract_metadata(file_path, chunk) for chunk in chunks]
            embeddings = self.embedding_service.embed_batch(chunks)

            collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            total_chunks += len(chunks)
            logger.info(
                "indexed_file",
                file=file_path.name,
                chunks=len(chunks),
            )

        logger.info("indexing_complete", total_chunks=total_chunks)
        return total_chunks
