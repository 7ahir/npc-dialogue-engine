"""SentenceBERT embedding service for semantic similarity."""

from sentence_transformers import SentenceTransformer

from src.utils.config import get_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Wraps SentenceBERT for text embedding generation.

    Uses all-MiniLM-L6-v2 by default: 384-dim embeddings, ~80MB model,
    sub-5ms per embedding — fast enough for game-ready latency.
    """

    def __init__(self, model_name: str | None = None) -> None:
        config = get_config()
        self._model_name = model_name or config.rag.embedding_model
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model on first use."""
        if self._model is None:
            logger.info("loading_embedding_model", model=self._model_name)
            self._model = SentenceTransformer(self._model_name)
            logger.info("embedding_model_loaded", model=self._model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed a batch of text strings."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
