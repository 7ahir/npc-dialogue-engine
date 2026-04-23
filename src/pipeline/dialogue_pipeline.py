"""Dialogue pipeline orchestrator.

Wires the full NPC dialogue flow: intent classification → RAG retrieval →
prompt assembly → model generation. This is the architecturally central file
— it demonstrates how game-ready ML systems decompose into fast, composable
stages that the game engine can reason about independently.
"""

import time
from collections.abc import Iterator
from dataclasses import dataclass

from src.models.dialogue_model import DialogueModel, create_dialogue_model
from src.models.intent_classifier import Intent, IntentClassifier
from src.pipeline.context_manager import ContextManager
from src.pipeline.prompt_templates import PromptBuilder
from src.rag.embeddings import EmbeddingService
from src.rag.retriever import LoreRetriever
from src.utils.config import AppConfig, get_config
from src.utils.logging_config import get_logger
from src.utils.tracing import TraceRecorder, TraceStore, get_trace_store

logger = get_logger(__name__)


@dataclass
class DialogueResponse:
    """Complete response from the dialogue pipeline."""

    npc_response: str
    intent: str
    confidence: float
    sentiment: float
    lore_refs: list[str]
    latency_ms: float
    model_version: str
    trace_id: str | None = None


class DialoguePipeline:
    """Orchestrates the full NPC dialogue generation flow.

    Pipeline stages:
    1. Session management — get/create conversation context
    2. Intent classification — what is the player asking? (~5ms)
    3. RAG retrieval — what lore is relevant? (~10ms)
    4. Prompt assembly — build structured prompt (~1ms)
    5. Generation — produce NPC response (~200-800ms with LLM)
    6. Session update — record the exchange

    Each stage is independently replaceable via constructor injection,
    making the pipeline testable without a GPU or database.
    """

    def __init__(
        self,
        model: DialogueModel | None = None,
        retriever: LoreRetriever | None = None,
        classifier: IntentClassifier | None = None,
        prompt_builder: PromptBuilder | None = None,
        context_manager: ContextManager | None = None,
        config: AppConfig | None = None,
        trace_store: TraceStore | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.config = config or get_config()
        self.model = model or create_dialogue_model(self.config.model)
        self.retriever = retriever or LoreRetriever()
        self.classifier = classifier or IntentClassifier()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.context_manager = context_manager or ContextManager()
        # Default to the process-wide singleton so the API can read traces
        # without callers having to plumb a store through.
        self.trace_store = trace_store if trace_store is not None else get_trace_store()
        # Reuse the retriever's embedding service when possible — avoids
        # loading SentenceTransformer twice. Falls back to None, in which
        # case ToT scoring uses a lexical-overlap heuristic instead of
        # cosine similarity (still better than always returning candidate 0).
        self.embedding_service: EmbeddingService | None = embedding_service or getattr(
            self.retriever, "embedding_service", None
        )
        # Per-character persona embedding cache — embedding the persona
        # reference text once per character is plenty; it never changes.
        self._persona_cache: dict[str, list[float]] = {}

    def process(
        self,
        player_message: str,
        character_id: str,
        session_id: str = "default",
        use_tot: bool = False,
    ) -> DialogueResponse:
        """Run the full dialogue pipeline synchronously.

        Args:
            player_message: The player's input text.
            character_id: Which NPC to respond as (must match a config file).
            session_id: Session identifier for multi-turn context.
            use_tot: If True, generate multiple candidates and select best (slower).

        Returns:
            DialogueResponse with NPC response and metadata.
        """
        start = time.perf_counter()
        recorder = TraceRecorder()

        # 1. Session management
        with recorder.span("session"):
            session = self.context_manager.get_or_create_session(session_id, character_id)

        # 2. Intent classification (graceful degradation)
        with recorder.span("intent") as meta:
            intent, confidence, all_scores, sentiment = self._classify_safe(player_message)
            meta["label"] = intent
            meta["confidence"] = confidence
            meta["sentiment"] = sentiment

        # 3. RAG retrieval (graceful degradation)
        with recorder.span("retrieval") as meta:
            lore_context, lore_refs = self._retrieve_safe(player_message)
            meta["chunks"] = len(lore_refs)

        # 4. Conversation history
        history = session.format_history(max_turns=self.config.api.max_session_history)

        # 5. Prompt assembly
        with recorder.span("prompt"):
            messages = self.prompt_builder.build_chat_messages(
                character_id=character_id,
                player_message=player_message,
                lore_context=lore_context,
                conversation_history=history,
            )

        # 6. Generation
        with recorder.span("generation") as meta:
            meta["mode"] = "tot" if use_tot else "single"
            if use_tot:
                npc_response, tot_meta = self._generate_with_tot(messages, character_id)
                meta.update(tot_meta)
            else:
                npc_response = self.model.generate(messages)
            meta["response_chars"] = len(npc_response)

        # 7. Update session
        session.add_message("player", player_message)
        session.add_message("npc", npc_response)

        latency_ms = (time.perf_counter() - start) * 1000
        trace = recorder.finish(
            character_id=character_id,
            session_id=session_id,
            intent=intent,
            use_tot=use_tot,
            model_version=self.model.model_version,
        )
        self.trace_store.add(trace)

        logger.info(
            "dialogue_processed",
            character=character_id,
            intent=intent,
            confidence=confidence,
            sentiment=sentiment,
            lore_refs_count=len(lore_refs),
            latency_ms=round(latency_ms, 1),
            trace_id=trace.trace_id,
        )

        return DialogueResponse(
            npc_response=npc_response,
            intent=intent,
            confidence=confidence,
            sentiment=sentiment,
            lore_refs=lore_refs,
            latency_ms=round(latency_ms, 1),
            model_version=self.model.model_version,
            trace_id=trace.trace_id,
        )

    def process_stream(
        self,
        player_message: str,
        character_id: str,
        session_id: str = "default",
    ) -> Iterator[str]:
        """Run the pipeline and yield tokens as they're generated.

        Token-only stream — kept for callers that just want the text. The
        SSE endpoint uses :meth:`stream_events` instead so it can prefix
        the stream with an ``intent``/``lore_refs`` metadata frame.
        """
        for ev in self.stream_events(player_message, character_id, session_id):
            if ev["event"] == "token":
                yield ev["data"]

    def stream_events(
        self,
        player_message: str,
        character_id: str,
        session_id: str = "default",
    ) -> Iterator[dict[str, str]]:
        """Run the pipeline and yield typed SSE-shaped events.

        Yields dicts of ``{"event": <type>, "data": <payload>}``:

        * ``metadata`` — JSON string with ``intent``, ``confidence``,
          ``sentiment``, ``lore_refs``, ``model_version``, ``trace_id``.
          Emitted once before any token so the client can render NPC
          framing (intent badge, lore citations) and link to the trace
          inspector ahead of the typewriter.
        * ``token`` — one streamed token from the model.
        * ``done`` — JSON string with ``latency_ms`` (full pipeline,
          including upfront stages and stream draining) and ``trace_id``.

        Records a real trace in the trace store — same span shape as
        ``process()`` — so the ``trace_id`` returned in metadata is a
        live link, not a decorative field. Streaming was previously
        invisible to ``/traces``; now it shows up alongside sync
        requests with a ``stream=true`` marker on the trace metadata.
        """
        import json

        recorder = TraceRecorder()
        start = time.perf_counter()

        with recorder.span("session"):
            session = self.context_manager.get_or_create_session(session_id, character_id)

        with recorder.span("intent") as meta:
            intent, confidence, _all_scores, sentiment = self._classify_safe(player_message)
            meta["label"] = intent
            meta["confidence"] = confidence
            meta["sentiment"] = sentiment

        with recorder.span("retrieval") as meta:
            lore_context, lore_refs = self._retrieve_safe(player_message)
            meta["chunks"] = len(lore_refs)

        history = session.format_history(max_turns=self.config.api.max_session_history)

        yield {
            "event": "metadata",
            "data": json.dumps(
                {
                    "intent": intent,
                    "confidence": round(confidence, 4),
                    "sentiment": round(sentiment, 4),
                    "lore_refs": lore_refs,
                    "model_version": self.model.model_version,
                    "trace_id": recorder.trace_id,
                }
            ),
        }

        with recorder.span("prompt"):
            messages = self.prompt_builder.build_chat_messages(
                character_id=character_id,
                player_message=player_message,
                lore_context=lore_context,
                conversation_history=history,
            )

        full_response: list[str] = []
        with recorder.span("generation") as meta:
            meta["mode"] = "stream"
            for token in self.model.generate_stream(messages):
                full_response.append(token)
                yield {"event": "token", "data": token}
            meta["response_chars"] = sum(len(t) for t in full_response)
            meta["token_count"] = len(full_response)

        npc_response = "".join(full_response)
        session.add_message("player", player_message)
        session.add_message("npc", npc_response)

        latency_ms = (time.perf_counter() - start) * 1000
        trace = recorder.finish(
            character_id=character_id,
            session_id=session_id,
            intent=intent,
            stream=True,
            model_version=self.model.model_version,
        )
        self.trace_store.add(trace)

        yield {
            "event": "done",
            "data": json.dumps(
                {
                    "latency_ms": round(latency_ms, 1),
                    "response_chars": len(npc_response),
                    "trace_id": recorder.trace_id,
                }
            ),
        }

    def _classify_safe(self, text: str) -> tuple[str, float, dict[str, float], float]:
        """Classify intent with graceful fallback on error."""
        try:
            intent, confidence, all_scores, sentiment = self.classifier.classify_with_sentiment(
                text
            )
            return intent.value, confidence, all_scores, sentiment
        except Exception as e:
            logger.warning("intent_classification_failed", error=str(e))
            return Intent.SOCIAL.value, 0.0, {}, 0.0

    def _retrieve_safe(self, query: str) -> tuple[str, list[str]]:
        """Retrieve lore context with graceful fallback on error."""
        try:
            chunks = self.retriever.retrieve(query)
            context = self.retriever.format_context(chunks)
            refs = [c.source_file for c in chunks]
            return context, refs
        except Exception as e:
            logger.warning("lore_retrieval_failed", error=str(e))
            return "", []

    def _generate_with_tot(
        self, messages: list[dict[str, str]], character_id: str
    ) -> tuple[str, dict]:
        """Tree of Thoughts: generate multiple candidates, return the best.

        Generates 3 responses with distinct emotional tones (helpful, guarded,
        cryptic) by appending a tone instruction to the system prompt. Each
        candidate is scored against a persona-reference embedding (description
        + concatenated example_phrases) using cosine similarity in the same
        SentenceBERT space we already use for RAG. The highest-scoring
        candidate wins.

        Falls back to a lexical-overlap heuristic if no embedding service is
        wired — still beats "always pick candidate 0", which was the previous
        behavior.

        Returns:
            Tuple of (chosen response, metadata dict). Metadata is attached to
            the generation trace span so the inspector can show winner tone,
            per-candidate scores, and which scoring method ran.
        """
        tones = ["helpful and warm", "cautious and guarded", "mysterious and cryptic"]
        candidates: list[str] = []

        for tone in tones:
            modified = list(messages)  # shallow copy of the messages list
            modified[0] = {
                "role": "system",
                "content": messages[0]["content"] + f"\n\nRespond in a {tone} tone.",
            }
            candidates.append(self.model.generate(modified))

        if not candidates:
            return self.model.generate(messages), {"tot_candidates": 0}

        scores, scoring_method = self._score_candidates(candidates, character_id)
        best_idx = max(range(len(candidates)), key=lambda i: scores[i])

        meta = {
            "tot_candidates": len(candidates),
            "tot_winner_tone": tones[best_idx],
            "tot_winner_score": round(scores[best_idx], 4),
            "tot_scores": [round(s, 4) for s in scores],
            "tot_scoring": scoring_method,
        }
        logger.info(
            "tot_selected",
            character=character_id,
            winner_tone=tones[best_idx],
            winner_score=round(scores[best_idx], 4),
            scoring=scoring_method,
        )
        return candidates[best_idx], meta

    def _get_persona_embedding(self, character_id: str) -> list[float] | None:
        """Embed the character's persona reference text (cached per id).

        Reference text = description + example_phrases joined. This is the
        most distilled view of "what this NPC sounds like" available in the
        config without round-tripping through the model.
        """
        if self.embedding_service is None:
            return None
        if character_id in self._persona_cache:
            return self._persona_cache[character_id]
        try:
            character = self.prompt_builder.character_loader.load(character_id)
        except Exception as exc:  # noqa: BLE001 — best-effort enrichment
            logger.warning("persona_load_failed", character=character_id, error=str(exc))
            return None
        ref_parts: list[str] = []
        desc = character.get("description")
        if desc:
            ref_parts.append(str(desc).strip())
        ref_parts.extend(str(p) for p in character.get("example_phrases", []) if p)
        ref_text = " ".join(ref_parts).strip()
        if not ref_text:
            return None
        try:
            emb = self.embedding_service.embed(ref_text)
        except Exception as exc:  # noqa: BLE001 — never let scoring crash generation
            logger.warning("persona_embed_failed", character=character_id, error=str(exc))
            return None
        self._persona_cache[character_id] = emb
        return emb

    def _score_candidates(
        self, candidates: list[str], character_id: str
    ) -> tuple[list[float], str]:
        """Score each candidate against the character's persona profile.

        Returns:
            (scores, method) where method is "cosine" if embedding scoring
            ran successfully, or "lexical" for the fallback heuristic.

        Cosine path: SentenceBERT embeddings are L2-normalized by the
        EmbeddingService, so cosine similarity reduces to a dot product.
        Range is [-1, 1] but in practice [0, 1] for meaningfully related
        text in this model.
        """
        persona_emb = self._get_persona_embedding(character_id)
        if persona_emb is not None and self.embedding_service is not None:
            try:
                cand_embs = self.embedding_service.embed_batch(candidates)
                scores = [
                    sum(a * b for a, b in zip(persona_emb, ce, strict=True))
                    for ce in cand_embs
                ]
                return scores, "cosine"
            except Exception as exc:  # noqa: BLE001
                logger.warning("candidate_embed_failed", error=str(exc))

        # Fallback: word-overlap with the character's example_phrases. Crude
        # but deterministic and never zero-discriminating across candidates
        # that diverge in tone (which they will, given the tone-stamped prompts).
        try:
            character = self.prompt_builder.character_loader.load(character_id)
        except Exception:  # noqa: BLE001
            return [0.0] * len(candidates), "lexical"
        phrase_words: set[str] = set()
        for p in character.get("example_phrases", []):
            phrase_words.update(str(p).lower().split())
        if not phrase_words:
            return [0.0] * len(candidates), "lexical"
        scores = [
            len(set(c.lower().split()) & phrase_words) / len(phrase_words)
            for c in candidates
        ]
        return scores, "lexical"
