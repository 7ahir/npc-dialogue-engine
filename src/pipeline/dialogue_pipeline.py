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
                npc_response = self._generate_with_tot(messages, character_id)
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

        Performs intent classification and RAG retrieval upfront,
        then streams the model's response token-by-token.
        """
        # Pre-generation stages (same as process)
        session = self.context_manager.get_or_create_session(session_id, character_id)
        lore_context, _ = self._retrieve_safe(player_message)
        history = session.format_history(max_turns=self.config.api.max_session_history)

        messages = self.prompt_builder.build_chat_messages(
            character_id=character_id,
            player_message=player_message,
            lore_context=lore_context,
            conversation_history=history,
        )

        # Stream generation
        full_response: list[str] = []
        for token in self.model.generate_stream(messages):
            full_response.append(token)
            yield token

        # Update session after streaming completes
        npc_response = "".join(full_response)
        session.add_message("player", player_message)
        session.add_message("npc", npc_response)

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

    def _generate_with_tot(self, messages: list[dict[str, str]], character_id: str) -> str:
        """Tree of Thoughts: generate multiple candidates, select best.

        Generates 3 responses with different tones, scores each against
        the character's persona, and returns the highest-scoring one.
        Only used for complex dialogue scenarios where quality > speed.
        """
        tones = ["helpful and warm", "cautious and guarded", "mysterious and cryptic"]
        candidates: list[str] = []

        for tone in tones:
            # Modify the system prompt to request a specific tone
            modified = list(messages)  # shallow copy
            modified[0] = {
                "role": "system",
                "content": messages[0]["content"] + f"\n\nRespond in a {tone} tone.",
            }
            candidate = self.model.generate(modified)
            candidates.append(candidate)

        # For mock model, just return the first candidate
        # With a real model, we'd score each against persona embeddings
        # and return the best match. This scoring logic will be added
        # in the evaluation module.
        if candidates:
            return candidates[0]
        return self.model.generate(messages)
