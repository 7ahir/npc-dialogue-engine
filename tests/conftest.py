"""Shared test fixtures for the NPC Dialogue Engine."""

from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def characters_dir(project_root: Path) -> Path:
    """Return the characters config directory."""
    return project_root / "configs" / "characters"


@pytest.fixture
def lore_dir(project_root: Path) -> Path:
    """Return the lore documents directory."""
    return project_root / "data" / "lore"


@pytest.fixture
def sample_character(characters_dir: Path) -> dict:
    """Load the blacksmith character config as a sample."""
    with open(characters_dir / "blacksmith.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def all_characters(characters_dir: Path) -> dict[str, dict]:
    """Load all character configs."""
    characters = {}
    for path in characters_dir.glob("*.yaml"):
        with open(path) as f:
            data = yaml.safe_load(f)
            characters[data["id"]] = data
    return characters
