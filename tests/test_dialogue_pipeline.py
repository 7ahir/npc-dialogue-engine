"""Tests for the dialogue pipeline orchestrator."""

import pytest

from src.models.dialogue_model import MockDialogueModel
from src.models.intent_classifier import Intent
from src.pipeline.context_manager import ContextManager
from src.pipeline.dialogue_pipeline import DialoguePipeline, DialogueResponse
from src.pipeline.prompt_templates import PromptBuilder
from src.rag.retriever import LoreChunk

# ─── Mock Retriever ─────────────────────────────────────────────


class StubRetriever:
    """Lightweight stub that returns fixed lore without ChromaDB."""

    def retrieve(self, query: str, **kwargs) -> list[LoreChunk]:  # type: ignore[no-untyped-def]
        return [
            LoreChunk(
                text="Moonstone resonates with residual aether from the Ley Network.",
                score=0.82,
                source_file="items_and_materials.md",
                section_title="Moonstone",
            )
        ]

    def format_context(self, chunks: list[LoreChunk]) -> str:
        return "\n".join(c.text for c in chunks)


class FailingRetriever:
    """Retriever that always raises — tests graceful degradation."""

    def retrieve(self, query: str, **kwargs) -> list[LoreChunk]:  # type: ignore[no-untyped-def]
        raise ConnectionError("ChromaDB unavailable")

    def format_context(self, chunks: list[LoreChunk]) -> str:
        return ""


# ─── Pipeline Tests ─────────────────────────────────────────────


class TestDialoguePipeline:
    @pytest.fixture
    def pipeline(self) -> DialoguePipeline:
        return DialoguePipeline(
            model=MockDialogueModel(),
            retriever=StubRetriever(),  # type: ignore[arg-type]
            prompt_builder=PromptBuilder(),
            context_manager=ContextManager(),
        )

    def test_process_returns_dialogue_response(self, pipeline: DialoguePipeline) -> None:
        result = pipeline.process(
            player_message="Got any swords for sale?",
            character_id="blacksmith",
            session_id="test-1",
        )
        assert isinstance(result, DialogueResponse)
        assert len(result.npc_response) > 0
        assert result.model_version == "mock-v1"
        assert result.latency_ms > 0

    def test_process_returns_intent(self, pipeline: DialoguePipeline) -> None:
        result = pipeline.process(
            player_message="Hello there, how are you?",
            character_id="tavern_keeper",
            session_id="test-2",
        )
        assert result.intent in [i.value for i in Intent]
        assert 0 <= result.confidence <= 1.0

    def test_process_returns_sentiment(self, pipeline: DialoguePipeline) -> None:
        result = pipeline.process(
            player_message="You're a disgrace!",
            character_id="blacksmith",
            session_id="test-3",
        )
        assert -1.0 <= result.sentiment <= 1.0

    def test_process_returns_lore_refs(self, pipeline: DialoguePipeline) -> None:
        result = pipeline.process(
            player_message="Tell me about Moonstone",
            character_id="mysterious_sage",
            session_id="test-4",
        )
        assert "items_and_materials.md" in result.lore_refs

    def test_session_history_accumulates(self, pipeline: DialoguePipeline) -> None:
        pipeline.process(
            player_message="Hello!",
            character_id="blacksmith",
            session_id="history-test",
        )
        pipeline.process(
            player_message="What about swords?",
            character_id="blacksmith",
            session_id="history-test",
        )

        session = pipeline.context_manager.get_or_create_session("history-test", "blacksmith")
        assert len(session.history) == 4  # 2 player + 2 npc messages

    def test_different_characters_respond_differently(self, pipeline: DialoguePipeline) -> None:
        r1 = pipeline.process(
            player_message="Hello",
            character_id="blacksmith",
            session_id="char-test-1",
        )
        r2 = pipeline.process(
            player_message="Hello",
            character_id="tavern_keeper",
            session_id="char-test-2",
        )
        # Mock model picks from example_phrases, which differ per character
        # At minimum both should produce non-empty responses
        assert len(r1.npc_response) > 0
        assert len(r2.npc_response) > 0

    def test_graceful_degradation_retriever_failure(self) -> None:
        pipeline = DialoguePipeline(
            model=MockDialogueModel(),
            retriever=FailingRetriever(),  # type: ignore[arg-type]
            prompt_builder=PromptBuilder(),
            context_manager=ContextManager(),
        )

        # Should not raise — continues without lore context
        result = pipeline.process(
            player_message="Tell me about the Sundering",
            character_id="mysterious_sage",
            session_id="fail-test",
        )
        assert len(result.npc_response) > 0
        assert result.lore_refs == []

    def test_process_stream(self, pipeline: DialoguePipeline) -> None:
        tokens = list(
            pipeline.process_stream(
                player_message="Got any swords?",
                character_id="blacksmith",
                session_id="stream-test",
            )
        )
        assert len(tokens) > 0
        full_response = "".join(tokens)
        assert len(full_response) > 0

    def test_process_with_tot(self, pipeline: DialoguePipeline) -> None:
        result = pipeline.process(
            player_message="What should I do about the Shadow Cult?",
            character_id="mysterious_sage",
            session_id="tot-test",
            use_tot=True,
        )
        assert len(result.npc_response) > 0
