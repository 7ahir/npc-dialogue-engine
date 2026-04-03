"""Player input intent and sentiment classification.

Uses zero-shot classification with a small transformer model for fast,
deterministic intent labels. The game engine can use these labels to
trigger animations, update quest state, or adjust NPC behavior before
the full dialogue response is generated.
"""

from enum import StrEnum

from transformers import pipeline as hf_pipeline

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Zero-shot classifier model — small and fast (~50ms on CPU)
_DEFAULT_CLASSIFIER_MODEL = "typeform/distilbert-base-uncased-mnli"


class Intent(StrEnum):
    """Player input intent categories."""

    QUEST = "quest"  # Asking about or accepting quests
    TRADE = "trade"  # Buying, selling, bartering
    LORE = "lore"  # Asking about world history, magic, events
    COMBAT = "combat"  # Combat-related dialogue, threats
    SOCIAL = "social"  # Greetings, small talk, relationship building
    HOSTILE = "hostile"  # Insults, aggression, provocation
    OFF_TOPIC = "off_topic"  # Meta-gaming, out-of-character, nonsense


# Labels used for zero-shot classification — phrased as natural language
# hypotheses for the MNLI model to score
_INTENT_HYPOTHESES = {
    Intent.QUEST: "This is about a quest, mission, task, or adventure request.",
    Intent.TRADE: "This is about buying, selling, trading, or pricing items.",
    Intent.LORE: (
        "This is a question about history, ancient knowledge, legends, lore, or world events."
    ),
    Intent.COMBAT: "This is about fighting, weapons, combat, or threats.",
    Intent.SOCIAL: "This is casual conversation, a greeting, or small talk.",
    Intent.HOSTILE: "This is aggressive, insulting, or hostile.",
    Intent.OFF_TOPIC: "This is off-topic, meta, or does not make sense in context.",
}


class IntentClassifier:
    """Classifies player input into intent categories with confidence scores.

    Uses zero-shot classification so no fine-tuning is needed — the model
    scores each intent hypothesis against the input text using natural
    language inference (NLI).
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or _DEFAULT_CLASSIFIER_MODEL
        self._classifier = None

    @property
    def classifier(self):  # type: ignore[no-untyped-def]
        """Lazy-load the zero-shot classification pipeline."""
        if self._classifier is None:
            logger.info("loading_intent_classifier", model=self._model_name)
            self._classifier = hf_pipeline(
                "zero-shot-classification",
                model=self._model_name,
                device=-1,  # CPU — fast enough for this model
            )
            logger.info("intent_classifier_loaded")
        return self._classifier

    def classify(self, text: str) -> tuple[Intent, float, dict[str, float]]:
        """Classify player input into an intent category.

        Args:
            text: The player's input message.

        Returns:
            Tuple of (top_intent, confidence, all_scores).
            - top_intent: The highest-scoring Intent enum value.
            - confidence: Score for the top intent (0.0-1.0).
            - all_scores: Dict mapping each intent to its score.
        """
        candidate_labels = list(_INTENT_HYPOTHESES.values())
        result = self.classifier(text, candidate_labels, multi_label=False)

        # Map hypothesis labels back to Intent enum values
        label_to_intent = {label: intent for intent, label in _INTENT_HYPOTHESES.items()}

        all_scores: dict[str, float] = {}
        for label, score in zip(result["labels"], result["scores"], strict=False):
            intent = label_to_intent[label]
            all_scores[intent.value] = round(score, 4)

        # Top result
        top_label = result["labels"][0]
        top_intent = label_to_intent[top_label]
        top_confidence = round(result["scores"][0], 4)

        logger.info(
            "intent_classified",
            text_preview=text[:50],
            intent=top_intent.value,
            confidence=top_confidence,
        )

        return top_intent, top_confidence, all_scores

    def classify_with_sentiment(self, text: str) -> tuple[Intent, float, dict[str, float], float]:
        """Classify intent and estimate sentiment from intent scores.

        Sentiment is derived from intent scores rather than a separate model,
        keeping latency low. Maps hostile/combat as negative, social as positive.

        Returns:
            Tuple of (intent, confidence, all_scores, sentiment).
            sentiment ranges from -1.0 (hostile) to 1.0 (friendly).
        """
        intent, confidence, all_scores = self.classify(text)

        # Derive sentiment from intent distribution
        positive_signals = all_scores.get("social", 0) + all_scores.get("trade", 0) * 0.5
        negative_signals = all_scores.get("hostile", 0) + all_scores.get("combat", 0) * 0.3
        sentiment = round(positive_signals - negative_signals, 4)
        sentiment = max(-1.0, min(1.0, sentiment))  # Clamp to [-1, 1]

        return intent, confidence, all_scores, sentiment
