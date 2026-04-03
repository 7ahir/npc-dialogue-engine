"""Application configuration using pydantic-settings."""

from pathlib import Path
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class ModelConfig(BaseSettings):
    """LLM model configuration."""

    model_config = SettingsConfigDict(env_prefix="MODEL_")

    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    lora_path: Path | None = None
    load_in_4bit: bool = True
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    device: str = "auto"


class RAGConfig(BaseSettings):
    """RAG pipeline configuration."""

    model_config = SettingsConfigDict(env_prefix="RAG_")

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    collection_name: str = "game_lore"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 3
    relevance_threshold: float = 0.35
    chroma_persist_dir: Path = PROJECT_ROOT / "chroma_data"
    lore_dir: Path = PROJECT_ROOT / "data" / "lore"


class APIConfig(BaseSettings):
    """API server configuration."""

    model_config = SettingsConfigDict(env_prefix="API_")

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default=["*"])
    max_session_history: int = 10
    session_ttl_seconds: int = 3600


class AppConfig(BaseSettings):
    """Top-level application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = PROJECT_ROOT
    characters_dir: Path = PROJECT_ROOT / "configs" / "characters"
    log_level: str = "INFO"
    log_format: str = "json"  # "json" or "console"

    model: ModelConfig = Field(default_factory=ModelConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    api: APIConfig = Field(default_factory=APIConfig)


@lru_cache
def get_config() -> AppConfig:
    """Get cached application configuration."""
    return AppConfig()
