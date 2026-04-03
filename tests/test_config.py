"""Tests for application configuration."""

from pathlib import Path

import yaml

from src.utils.config import AppConfig, ModelConfig, RAGConfig, APIConfig


class TestModelConfig:
    def test_default_values(self) -> None:
        config = ModelConfig()
        assert config.base_model == "Qwen/Qwen2.5-3B-Instruct"
        assert config.load_in_4bit is True
        assert config.max_new_tokens == 256
        assert config.temperature == 0.7
        assert config.device == "auto"

    def test_lora_path_optional(self) -> None:
        config = ModelConfig()
        assert config.lora_path is None


class TestRAGConfig:
    def test_default_values(self) -> None:
        config = RAGConfig()
        assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.top_k == 3
        assert config.relevance_threshold == 0.35
        assert config.chunk_size == 512

    def test_lore_dir_exists(self) -> None:
        config = RAGConfig()
        assert config.lore_dir.exists()


class TestAPIConfig:
    def test_default_values(self) -> None:
        config = APIConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.max_session_history == 10


class TestAppConfig:
    def test_nested_configs(self) -> None:
        config = AppConfig()
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.rag, RAGConfig)
        assert isinstance(config.api, APIConfig)

    def test_characters_dir_exists(self) -> None:
        config = AppConfig()
        assert config.characters_dir.exists()

    def test_project_root_valid(self) -> None:
        config = AppConfig()
        assert (config.project_root / "pyproject.toml").exists()
