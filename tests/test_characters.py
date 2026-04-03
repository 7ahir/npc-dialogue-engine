"""Tests for character configuration integrity."""

from pathlib import Path

REQUIRED_FIELDS = [
    "id",
    "name",
    "role",
    "description",
    "personality_traits",
    "speech_patterns",
    "vocabulary_constraints",
    "knowledge_boundaries",
    "emotional_range",
    "example_phrases",
    "quest_hooks",
]


class TestCharacterConfigs:
    def test_all_characters_have_required_fields(self, all_characters: dict[str, dict]) -> None:
        for char_id, char_data in all_characters.items():
            for field in REQUIRED_FIELDS:
                assert field in char_data, f"Character '{char_id}' missing field: {field}"

    def test_character_ids_unique(self, all_characters: dict[str, dict]) -> None:
        ids = list(all_characters.keys())
        assert len(ids) == len(set(ids)), "Duplicate character IDs found"

    def test_minimum_character_count(self, all_characters: dict[str, dict]) -> None:
        count = len(all_characters)
        assert count >= 3, f"Expected at least 3 characters, got {count}"

    def test_example_phrases_non_empty(self, all_characters: dict[str, dict]) -> None:
        for char_id, char_data in all_characters.items():
            assert len(char_data["example_phrases"]) >= 3, (
                f"Character '{char_id}' has fewer than 3 example phrases"
            )

    def test_personality_traits_non_empty(self, all_characters: dict[str, dict]) -> None:
        for char_id, char_data in all_characters.items():
            assert len(char_data["personality_traits"]) >= 3, (
                f"Character '{char_id}' has fewer than 3 personality traits"
            )

    def test_quest_hooks_present(self, all_characters: dict[str, dict]) -> None:
        for char_id, char_data in all_characters.items():
            assert len(char_data["quest_hooks"]) >= 1, f"Character '{char_id}' has no quest hooks"

    def test_blacksmith_specific(self, sample_character: dict) -> None:
        assert sample_character["id"] == "blacksmith"
        assert sample_character["name"] == "Grenn Ironheart"
        assert any(
            "smith" in p.lower() or "forge" in p.lower() or "hammer" in p.lower()
            for p in sample_character["speech_patterns"]
        )


class TestLoreDocuments:
    def test_lore_dir_has_documents(self, lore_dir: Path) -> None:
        lore_files = list(lore_dir.glob("*.md"))
        assert len(lore_files) >= 3, f"Expected at least 3 lore documents, got {len(lore_files)}"

    def test_lore_documents_non_empty(self, lore_dir: Path) -> None:
        for path in lore_dir.glob("*.md"):
            content = path.read_text()
            assert len(content) > 100, f"Lore document '{path.name}' seems too short"

    def test_required_lore_topics(self, lore_dir: Path) -> None:
        filenames = {p.stem for p in lore_dir.glob("*.md")}
        required = {"world_history", "factions", "locations"}
        missing = required - filenames
        assert not missing, f"Missing required lore documents: {missing}"
