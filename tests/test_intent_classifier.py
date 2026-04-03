"""Tests for the intent classification module."""

import pytest

from src.models.intent_classifier import Intent, IntentClassifier


@pytest.fixture(scope="module")
def classifier() -> IntentClassifier:
    """Shared classifier instance — model loading is expensive."""
    return IntentClassifier()


class TestIntentClassifier:
    @pytest.mark.slow
    def test_quest_intent(self, classifier: IntentClassifier) -> None:
        intent, confidence, _ = classifier.classify(
            "I need a quest. Do you have any missions for me?"
        )
        assert intent == Intent.QUEST
        assert confidence > 0.2

    @pytest.mark.slow
    def test_trade_intent(self, classifier: IntentClassifier) -> None:
        intent, confidence, _ = classifier.classify(
            "How much for that sword? I want to buy some armor."
        )
        assert intent == Intent.TRADE
        assert confidence > 0.2

    @pytest.mark.slow
    def test_lore_intent(self, classifier: IntentClassifier) -> None:
        intent, confidence, _ = classifier.classify(
            "What ancient legends do you know? Tell me about the Sundering and the old world."
        )
        assert intent == Intent.LORE
        assert confidence > 0.2

    @pytest.mark.slow
    def test_social_intent(self, classifier: IntentClassifier) -> None:
        intent, confidence, _ = classifier.classify("Hello there! How are you doing today?")
        assert intent == Intent.SOCIAL
        assert confidence > 0.2

    @pytest.mark.slow
    def test_hostile_intent(self, classifier: IntentClassifier) -> None:
        intent, confidence, _ = classifier.classify(
            "You're a fool and a liar. I should burn this place down."
        )
        assert intent == Intent.HOSTILE
        assert confidence > 0.2

    @pytest.mark.slow
    def test_all_scores_returned(self, classifier: IntentClassifier) -> None:
        _, _, all_scores = classifier.classify("Hello there!")
        assert len(all_scores) == len(Intent)
        assert all(isinstance(v, float) for v in all_scores.values())
        assert abs(sum(all_scores.values()) - 1.0) < 0.01  # Should sum to ~1

    @pytest.mark.slow
    def test_sentiment_friendly(self, classifier: IntentClassifier) -> None:
        _, _, _, sentiment = classifier.classify_with_sentiment(
            "Hello friend! Great day, isn't it?"
        )
        assert sentiment > 0  # Should be positive

    @pytest.mark.slow
    def test_sentiment_hostile(self, classifier: IntentClassifier) -> None:
        _, _, _, sentiment = classifier.classify_with_sentiment(
            "Get out of my way, you worthless fool."
        )
        assert sentiment < 0  # Should be negative
