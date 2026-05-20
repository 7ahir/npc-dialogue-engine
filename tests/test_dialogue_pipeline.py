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

    def test_stream_events_emits_metadata_then_tokens_then_done(
        self, pipeline: DialoguePipeline
    ) -> None:
        """The SSE-shaped stream must front-load intent/lore and end with latency."""
        import json

        events = list(
            pipeline.stream_events(
                player_message="Got any swords?",
                character_id="blacksmith",
                session_id="stream-events-test",
            )
        )
        assert events, "stream_events yielded nothing"

        # First frame is metadata with the structured fields the client needs
        # to render before the first token arrives.
        assert events[0]["event"] == "metadata"
        meta = json.loads(events[0]["data"])
        assert {
            "intent",
            "confidence",
            "sentiment",
            "lore_refs",
            "model_version",
            "trace_id",
        } <= set(meta)
        assert meta["trace_id"], "trace_id must be a non-empty string"

        # The streaming trace must actually be persisted — not just decorative.
        # Rebuild the same store the pipeline used so we can look it up.
        stored = pipeline.trace_store.get(meta["trace_id"])
        assert stored is not None, "streaming trace was not recorded in the trace store"
        assert stored.metadata.get("stream") is True
        span_names = [s.name for s in stored.spans]
        assert "intent" in span_names and "generation" in span_names

        # Last frame is done with a numeric latency_ms.
        assert events[-1]["event"] == "done"
        done = json.loads(events[-1]["data"])
        assert isinstance(done.get("latency_ms"), (int, float))
        assert done["latency_ms"] >= 0

        # Everything in between is at least one token.
        token_events = [ev for ev in events[1:-1] if ev["event"] == "token"]
        assert token_events, "no token events emitted between metadata and done"

    def test_process_with_tot(self, pipeline: DialoguePipeline) -> None:
        result = pipeline.process(
            player_message="What should I do about the Shadow Cult?",
            character_id="mysterious_sage",
            session_id="tot-test",
            use_tot=True,
        )
        assert len(result.npc_response) > 0


# ─── Tree of Thoughts Scoring ────────────────────────────────────


class StubEmbeddingService:
    """Deterministic embedding stub: encodes by token-set vector.

    Embeds each text into a dict-of-counts (treated as a sparse vector),
    L2-normalizes, then converts back to a 16-dim dense vector by hashing
    tokens to slots. Cosine sim between two of these is high when the
    texts share many words. Good enough to make the scoring path
    discriminate predictably without loading sentence-transformers.
    """

    def __init__(self) -> None:
        self._dim = 16

    @staticmethod
    def _stable_hash(s: str) -> int:
        # Builtin hash() is randomized per-process via PYTHONHASHSEED — using
        # md5 keeps the embedding deterministic across pytest invocations so
        # ToT scoring tests don't flake on CI.
        import hashlib

        return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self._dim
        for tok in text.lower().split():
            v[self._stable_hash(tok) % self._dim] += 1.0
        norm = sum(x * x for x in v) ** 0.5
        return [x / norm if norm else 0.0 for x in v]

    def embed(self, text: str) -> list[float]:
        return self._vec(text)

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        return [self._vec(t) for t in texts]


class _ScriptedModel:
    """Returns a different fixed response each call, in order. Lets us assert
    *which* candidate ToT picked rather than relying on the mock model's
    keyword-overlap heuristic."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._idx = 0

    def generate(self, messages):  # type: ignore[no-untyped-def]
        out = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return out

    def generate_stream(self, messages):  # type: ignore[no-untyped-def]
        yield self.generate(messages)

    @property
    def model_version(self) -> str:
        return "scripted-test"


class TestToTScoring:
    """Lock down that ToT actually scores candidates and picks the best one,
    not the historical 'always return candidate[0]' bug."""

    def _make_pipeline(self, model, embedding_service=None) -> DialoguePipeline:
        return DialoguePipeline(
            model=model,
            retriever=StubRetriever(),  # type: ignore[arg-type]
            prompt_builder=PromptBuilder(),
            context_manager=ContextManager(),
            embedding_service=embedding_service,
        )

    def test_tot_picks_candidate_closest_to_persona_with_embeddings(self) -> None:
        # The blacksmith's example_phrases are full of "steel", "forge",
        # "cinder", etc. Candidate B reuses that vocabulary; A and C don't.
        # With the stub embedding service (token-overlap cosine), B should win.
        model = _ScriptedModel(
            responses=[
                "Greetings, traveler, I am most pleased to see you.",  # A: friendly, off-vocab
                "Good steel speaks for itself. Bring me the materials, "
                "I'll forge it. Always has been the deal.",  # B: blacksmith voice
                "The mists obscure all answers, child of the void.",  # C: cryptic, off-vocab
            ]
        )
        pipeline = self._make_pipeline(model, embedding_service=StubEmbeddingService())

        result = pipeline.process(
            player_message="Make me a sword",
            character_id="blacksmith",
            session_id="tot-cosine",
            use_tot=True,
        )

        assert "forge" in result.npc_response.lower()
        # The generation span should expose ToT bookkeeping
        trace = pipeline.trace_store.get(result.trace_id or "")
        assert trace is not None
        gen_span = next(s for s in trace.spans if s.name == "generation")
        assert gen_span.metadata["tot_candidates"] == 3
        assert gen_span.metadata["tot_scoring"] == "cosine"
        assert len(gen_span.metadata["tot_scores"]) == 3
        # Winner score should match the reported max
        assert gen_span.metadata["tot_winner_score"] == max(gen_span.metadata["tot_scores"])

    def test_tot_falls_back_to_lexical_without_embedding_service(self) -> None:
        # No embedding service → lexical overlap with example_phrases.
        # Candidate B borrows multiple words from the blacksmith's phrases.
        model = _ScriptedModel(
            responses=[
                "Hello there friend.",  # zero overlap
                "Good steel speaks. Bring materials, I'll forge it.",  # overlaps a lot
                "Mist and shadow.",  # zero overlap
            ]
        )
        pipeline = self._make_pipeline(model, embedding_service=None)

        result = pipeline.process(
            player_message="Make me a sword",
            character_id="blacksmith",
            session_id="tot-lexical",
            use_tot=True,
        )

        assert "forge" in result.npc_response.lower()
        trace = pipeline.trace_store.get(result.trace_id or "")
        assert trace is not None
        gen_span = next(s for s in trace.spans if s.name == "generation")
        assert gen_span.metadata["tot_scoring"] == "lexical"
        assert gen_span.metadata["tot_winner_score"] > 0  # actually scored, not zero-default

    def test_tot_metadata_attached_to_trace(self) -> None:
        model = _ScriptedModel(responses=["one alpha", "two beta steel forge", "three gamma"])
        pipeline = self._make_pipeline(model, embedding_service=StubEmbeddingService())
        result = pipeline.process(
            player_message="hi",
            character_id="blacksmith",
            session_id="tot-meta",
            use_tot=True,
        )

        trace = pipeline.trace_store.get(result.trace_id or "")
        assert trace is not None
        gen_meta = next(s.metadata for s in trace.spans if s.name == "generation")
        expected_keys = (
            "tot_candidates",
            "tot_winner_tone",
            "tot_winner_score",
            "tot_scores",
            "tot_scoring",
        )
        for key in expected_keys:
            assert key in gen_meta, f"missing {key} in generation span metadata"
        assert gen_meta["tot_winner_tone"] in (
            "helpful and warm",
            "cautious and guarded",
            "mysterious and cryptic",
        )

    def test_persona_embedding_is_cached(self) -> None:
        """The persona reference text never changes per character, so we
        should embed it exactly once even across many ToT requests."""

        class CountingStub(StubEmbeddingService):
            def __init__(self) -> None:
                super().__init__()
                self.embed_calls = 0

            def embed(self, text: str) -> list[float]:
                self.embed_calls += 1
                return super().embed(text)

        emb = CountingStub()
        model = _ScriptedModel(responses=["a", "b", "c"] * 10)
        pipeline = self._make_pipeline(model, embedding_service=emb)

        for i in range(3):
            pipeline.process(
                player_message=f"msg {i}",
                character_id="blacksmith",
                session_id=f"cache-{i}",
                use_tot=True,
            )

        # Persona embed should have been called exactly once for blacksmith
        assert emb.embed_calls == 1, (
            f"expected 1 persona embed across 3 ToT calls, got {emb.embed_calls}"
        )
