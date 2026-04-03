"""Tests for conversation context management."""

import time

import pytest

from src.pipeline.context_manager import ContextManager, Session


class TestSession:
    def test_add_message(self) -> None:
        session = Session(session_id="test", character_id="blacksmith")
        session.add_message("player", "Hello!")
        session.add_message("npc", "Welcome, traveler.")
        assert len(session.history) == 2

    def test_get_recent_history(self) -> None:
        session = Session(session_id="test", character_id="blacksmith")
        for i in range(10):
            session.add_message("player", f"Message {i}")
        recent = session.get_recent_history(max_turns=3)
        assert len(recent) == 3
        assert recent[-1].content == "Message 9"

    def test_format_history(self) -> None:
        session = Session(session_id="test", character_id="blacksmith")
        session.add_message("player", "Got any swords?")
        session.add_message("npc", "Aye, finest steel.")
        formatted = session.format_history(max_turns=5)
        assert "Player: Got any swords?" in formatted
        assert "NPC: Aye, finest steel." in formatted

    def test_format_empty_history(self) -> None:
        session = Session(session_id="test", character_id="blacksmith")
        assert session.format_history(max_turns=5) == ""

    def test_add_message_updates_last_active(self) -> None:
        session = Session(session_id="test", character_id="blacksmith")
        initial = session.last_active
        time.sleep(0.01)
        session.add_message("player", "Hello")
        assert session.last_active > initial


class TestContextManager:
    @pytest.fixture
    def manager(self) -> ContextManager:
        return ContextManager()

    def test_create_session(self, manager: ContextManager) -> None:
        session = manager.get_or_create_session("s1", "blacksmith")
        assert session.session_id == "s1"
        assert session.character_id == "blacksmith"

    def test_get_existing_session(self, manager: ContextManager) -> None:
        s1 = manager.get_or_create_session("s1", "blacksmith")
        s1.add_message("player", "Hello")
        s2 = manager.get_or_create_session("s1", "blacksmith")
        assert len(s2.history) == 1  # Same session, history preserved

    def test_character_change_resets_history(self, manager: ContextManager) -> None:
        s1 = manager.get_or_create_session("s1", "blacksmith")
        s1.add_message("player", "Hello blacksmith")
        s2 = manager.get_or_create_session("s1", "tavern_keeper")
        assert len(s2.history) == 0  # New character, fresh history
        assert s2.character_id == "tavern_keeper"

    def test_reset_session(self, manager: ContextManager) -> None:
        manager.get_or_create_session("s1", "blacksmith")
        assert manager.reset_session("s1") is True
        assert manager.reset_session("s1") is False  # Already gone

    def test_active_session_count(self, manager: ContextManager) -> None:
        manager.get_or_create_session("s1", "blacksmith")
        manager.get_or_create_session("s2", "tavern_keeper")
        assert manager.active_session_count == 2

    def test_multiple_independent_sessions(self, manager: ContextManager) -> None:
        s1 = manager.get_or_create_session("s1", "blacksmith")
        s2 = manager.get_or_create_session("s2", "tavern_keeper")
        s1.add_message("player", "Hello smith")
        s2.add_message("player", "Hello keeper")
        assert len(s1.history) == 1
        assert len(s2.history) == 1
        assert s1.history[0].content != s2.history[0].content
