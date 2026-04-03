"""Vector search over lore database for context-grounded NPC dialogue."""

import chromadb

from src.rag.embeddings import EmbeddingService
from src.utils.config import get_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class LoreChunk:
    """A retrieved lore chunk with relevance metadata."""

    __slots__ = ("text", "score", "source_file", "section_title")

    def __init__(
        self,
        text: str,
        score: float,
        source_file: str,
        section_title: str,
    ) -> None:
        self.text = text
        self.score = score
        self.source_file = source_file
        self.section_title = section_title

    def __repr__(self) -> str:
        return (
            f"LoreChunk(score={self.score:.3f}, "
            f"source='{self.source_file}', "
            f"section='{self.section_title}')"
        )


class LoreRetriever:
    """Retrieves relevant lore chunks for a given query.

    Uses cosine similarity between query embedding and indexed lore
    chunks. Applies a relevance threshold to skip low-quality matches
    — if no chunk exceeds the threshold, returns empty (no RAG context
    is better than bad RAG context for NPC dialogue).
    """

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

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        relevance_threshold: float | None = None,
    ) -> list[LoreChunk]:
        """Retrieve top-k relevant lore chunks for a query.

        Args:
            query: The player input or search text.
            top_k: Number of results to fetch (default from config).
            relevance_threshold: Minimum similarity score (default from config).

        Returns:
            List of LoreChunk objects sorted by relevance (highest first).
            Empty list if no chunks exceed the threshold.
        """
        top_k = top_k or self.config.rag.top_k
        relevance_threshold = relevance_threshold or self.config.rag.relevance_threshold

        collection = self._client.get_or_create_collection(
            name=self.config.rag.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        if collection.count() == 0:
            logger.warning("empty_lore_collection")
            return []

        query_embedding = self.embedding_service.embed(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        chunks: list[LoreChunk] = []
        documents = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results["distances"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []

        for doc, distance, metadata in zip(documents, distances, metadatas):
            # ChromaDB returns cosine distance, convert to similarity
            similarity = 1.0 - distance

            if similarity < relevance_threshold:
                continue

            chunks.append(
                LoreChunk(
                    text=doc,
                    score=similarity,
                    source_file=metadata.get("source_file", ""),
                    section_title=metadata.get("section_title", ""),
                )
            )

        logger.info(
            "lore_retrieved",
            query_preview=query[:50],
            total_results=len(documents),
            above_threshold=len(chunks),
        )

        return chunks

    def format_context(self, chunks: list[LoreChunk]) -> str:
        """Format retrieved chunks into a prompt-ready context string."""
        if not chunks:
            return ""

        parts: list[str] = []
        for chunk in chunks:
            header = f"[{chunk.source_file} — {chunk.section_title}]" if chunk.section_title else f"[{chunk.source_file}]"
            parts.append(f"{header}\n{chunk.text}")

        return "\n\n".join(parts)
