"""Tests for prompt template building and character loading."""

import pytest

from src.pipeline.prompt_templates import CharacterLoader, PromptBuilder


class TestCharacterLoader:
    @pytest.fixture
    def loader(self) -> CharacterLoader:
        return CharacterLoader()

    def test_load_blacksmith(self, loader: CharacterLoader) -> None:
        char = loader.load("blacksmith")
        assert char["name"] == "Grenn Ironheart"
        assert char["id"] == "blacksmith"

    def test_load_all_characters(self, loader: CharacterLoader) -> None:
        ids = loader.list_characters()
        assert "blacksmith" in ids
        assert "tavern_keeper" in ids
        assert "mysterious_sage" in ids

    def test_load_invalid_character_raises(self, loader: CharacterLoader) -> None:
        with pytest.raises(ValueError, match="not found"):
            loader.load("nonexistent_npc")

    def test_caching(self, loader: CharacterLoader) -> None:
        char1 = loader.load("blacksmith")
        char2 = loader.load("blacksmith")
        assert char1 is char2  # Same object from cache


class TestPromptBuilder:
    @pytest.fixture
    def builder(self) -> PromptBuilder:
        return PromptBuilder()

    def test_system_prompt_contains_character_name(self, builder: PromptBuilder) -> None:
        prompt = builder.build_system_prompt("blacksmith")
        assert "Grenn Ironheart" in prompt

    def test_system_prompt_contains_personality(self, builder: PromptBuilder) -> None:
        prompt = builder.build_system_prompt("blacksmith")
        assert "Gruff but fair" in prompt

    def test_system_prompt_with_lore_context(self, builder: PromptBuilder) -> None:
        lore = "Moonstone is found in the Frozen Mines."
        prompt = builder.build_system_prompt("blacksmith", lore_context=lore)
        assert "Moonstone" in prompt
        assert "Relevant World Knowledge" in prompt

    def test_system_prompt_without_lore_excludes_section(self, builder: PromptBuilder) -> None:
        prompt = builder.build_system_prompt("blacksmith", lore_context="")
        assert "Relevant World Knowledge" not in prompt

    def test_system_prompt_with_history(self, builder: PromptBuilder) -> None:
        history = "Player: Hello!\nNPC: Welcome, traveler."
        prompt = builder.build_system_prompt("blacksmith", conversation_history=history)
        assert "Recent Conversation" in prompt
        assert "Hello!" in prompt

    def test_build_chat_messages_structure(self, builder: PromptBuilder) -> None:
        messages = builder.build_chat_messages(
            character_id="tavern_keeper",
            player_message="What's the news?",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What's the news?"
        assert "Mira Hearthstone" in messages[0]["content"]

    def test_build_chat_messages_with_full_context(self, builder: PromptBuilder) -> None:
        messages = builder.build_chat_messages(
            character_id="mysterious_sage",
            player_message="What is the Sundering?",
            lore_context="The Sundering shattered Pangaethera 2000 years ago.",
            conversation_history="Player: Hello\nNPC: Knowledge is not given.",
        )
        system = messages[0]["content"]
        assert "Eldris the Veiled" in system
        assert "Sundering" in system
        assert "Knowledge is not given" in system

    def test_tot_prompt_contains_tones(self, builder: PromptBuilder) -> None:
        prompt = builder.build_tot_prompt("blacksmith", num_candidates=3)
        assert "3" in prompt
        assert "helpful and warm" in prompt
        assert "cautious and guarded" in prompt

    def test_tot_prompt_custom_tones(self, builder: PromptBuilder) -> None:
        tones = ["angry", "dismissive"]
        prompt = builder.build_tot_prompt("blacksmith", num_candidates=2, tones=tones)
        assert "angry" in prompt
        assert "dismissive" in prompt

    def test_all_characters_produce_valid_prompts(self, builder: PromptBuilder) -> None:
        loader = builder.character_loader
        for char_id in loader.list_characters():
            prompt = builder.build_system_prompt(char_id)
            assert len(prompt) > 100, f"Prompt for '{char_id}' seems too short"
            # Should contain the rules section
            assert "Stay in character" in prompt
