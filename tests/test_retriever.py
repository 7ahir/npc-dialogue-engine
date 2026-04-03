"""Tests for the RAG pipeline: embeddings, indexer, and retriever."""

from pathlib import Path

import chromadb
import pytest

from src.rag.embeddings import EmbeddingService
from src.rag.lore_indexer import LoreIndexer, _chunk_text
from src.rag.retriever import LoreRetriever

# ─── Chunking Tests ─────────────────────────────────────────────


class TestChunking:
    def test_short_text_single_chunk(self) -> None:
        chunks = _chunk_text("Hello world", chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_long_text_splits(self) -> None:
        text = "Word " * 200  # ~1000 chars
        chunks = _chunk_text(text, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1

    def test_empty_text(self) -> None:
        chunks = _chunk_text("", chunk_size=512)
        assert len(chunks) == 0

    def test_whitespace_only(self) -> None:
        chunks = _chunk_text("   \n\n  ", chunk_size=512)
        assert len(chunks) == 0

    def test_chunks_have_no_leading_trailing_whitespace(self) -> None:
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = _chunk_text(text, chunk_size=30, chunk_overlap=5)
        for chunk in chunks:
            assert chunk == chunk.strip()

    def test_overlap_creates_redundancy(self) -> None:
        text = "A" * 100 + "\n\n" + "B" * 100
        chunks_no_overlap = _chunk_text(text, chunk_size=110, chunk_overlap=0)
        chunks_with_overlap = _chunk_text(text, chunk_size=110, chunk_overlap=30)
        # With overlap, total text coverage should be larger
        total_no = sum(len(c) for c in chunks_no_overlap)
        total_with = sum(len(c) for c in chunks_with_overlap)
        assert total_with >= total_no


# ─── Embedding Tests ────────────────────────────────────────────


class TestEmbeddingService:
    @pytest.fixture(scope="class")
    def service(self) -> EmbeddingService:
        return EmbeddingService()

    def test_embed_returns_list_of_floats(self, service: EmbeddingService) -> None:
        result = service.embed("Hello world")
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], float)

    def test_embed_dimension(self, service: EmbeddingService) -> None:
        result = service.embed("Test")
        assert len(result) == service.dimension

    def test_embed_batch(self, service: EmbeddingService) -> None:
        texts = ["Hello", "World", "Test"]
        results = service.embed_batch(texts)
        assert len(results) == 3
        assert all(len(r) == service.dimension for r in results)

    def test_similar_texts_have_high_similarity(self, service: EmbeddingService) -> None:
        import numpy as np

        e1 = np.array(service.embed("The blacksmith forges a sword"))
        e2 = np.array(service.embed("The smith hammers a blade at the anvil"))
        e3 = np.array(service.embed("The weather is sunny today"))

        sim_related = float(np.dot(e1, e2))
        sim_unrelated = float(np.dot(e1, e3))

        assert sim_related > sim_unrelated


# ─── Indexer + Retriever Integration Tests ──────────────────────


class TestLoreIndexerAndRetriever:
    @pytest.fixture
    def chroma_client(self) -> chromadb.ClientAPI:
        """Create an ephemeral ChromaDB client for testing."""
        return chromadb.EphemeralClient()

    @pytest.fixture
    def embedding_service(self) -> EmbeddingService:
        return EmbeddingService()

    @pytest.fixture
    def indexed_collection(
        self,
        chroma_client: chromadb.ClientAPI,
        embedding_service: EmbeddingService,
        lore_dir: Path,
    ) -> int:
        """Index the real lore documents and return chunk count."""
        indexer = LoreIndexer(
            embedding_service=embedding_service,
            chroma_client=chroma_client,
        )
        return indexer.index_directory(lore_dir)

    def test_indexing_creates_chunks(self, indexed_collection: int) -> None:
        assert indexed_collection > 0

    def test_indexing_processes_all_files(self, indexed_collection: int, lore_dir: Path) -> None:
        # We have 5 lore files, each should produce at least 1 chunk
        assert indexed_collection >= 5

    def test_retriever_finds_relevant_chunks(
        self,
        indexed_collection: int,
        chroma_client: chromadb.ClientAPI,
        embedding_service: EmbeddingService,
    ) -> None:
        retriever = LoreRetriever(
            embedding_service=embedding_service,
            chroma_client=chroma_client,
        )

        # Query about Moonstone — should retrieve items_and_materials content
        chunks = retriever.retrieve("Tell me about Moonstone Ingots", top_k=3)
        assert len(chunks) > 0
        assert any("moonstone" in c.text.lower() for c in chunks)

    def test_retriever_returns_scored_chunks(
        self,
        indexed_collection: int,
        chroma_client: chromadb.ClientAPI,
        embedding_service: EmbeddingService,
    ) -> None:
        retriever = LoreRetriever(
            embedding_service=embedding_service,
            chroma_client=chroma_client,
        )

        chunks = retriever.retrieve("Shadow Cult activities", top_k=3)
        assert all(isinstance(c.score, float) for c in chunks)
        # Scores should be in descending order
        scores = [c.score for c in chunks]
        assert scores == sorted(scores, reverse=True)

    def test_retriever_respects_threshold(
        self,
        indexed_collection: int,
        chroma_client: chromadb.ClientAPI,
        embedding_service: EmbeddingService,
    ) -> None:
        retriever = LoreRetriever(
            embedding_service=embedding_service,
            chroma_client=chroma_client,
        )

        # Very high threshold should return fewer results
        chunks_low = retriever.retrieve("magic system", relevance_threshold=0.1)
        chunks_high = retriever.retrieve("magic system", relevance_threshold=0.8)
        assert len(chunks_low) >= len(chunks_high)

    def test_retriever_format_context(
        self,
        indexed_collection: int,
        chroma_client: chromadb.ClientAPI,
        embedding_service: EmbeddingService,
    ) -> None:
        retriever = LoreRetriever(
            embedding_service=embedding_service,
            chroma_client=chroma_client,
        )

        chunks = retriever.retrieve("Starfall Steel", top_k=2)
        context = retriever.format_context(chunks)

        if chunks:
            assert len(context) > 0
            # Should contain source file references
            assert "[" in context

    def test_retriever_empty_collection(
        self,
        chroma_client: chromadb.ClientAPI,
        embedding_service: EmbeddingService,
    ) -> None:
        retriever = LoreRetriever(
            embedding_service=embedding_service,
            chroma_client=chroma_client,
        )

        # Query against non-existent collection
        chunks = retriever.retrieve("anything")
        assert chunks == []

    def test_reindexing_clears_old_data(
        self,
        chroma_client: chromadb.ClientAPI,
        embedding_service: EmbeddingService,
        lore_dir: Path,
    ) -> None:
        indexer = LoreIndexer(
            embedding_service=embedding_service,
            chroma_client=chroma_client,
        )

        count1 = indexer.index_directory(lore_dir)
        count2 = indexer.index_directory(lore_dir)
        # Re-indexing should produce the same count (not double)
        assert count1 == count2
