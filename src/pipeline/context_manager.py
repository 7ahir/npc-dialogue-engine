"""Conversation history and session state management.

Tracks multi-turn dialogue per session so the LLM sees prior context.
Uses in-memory storage with TTL expiry — appropriate for game sessions
where persistence isn't needed across restarts.
"""

import time
from dataclasses import dataclass, field

from src.utils.config import get_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Message:
    """A single message in the conversation."""

    role: str  # "player" or "npc"
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Session:
    """Tracks state for a single player-NPC conversation."""

    session_id: str
    character_id: str
    history: list[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def add_message(self, role: str, content: str) -> None:
        """Add a message and update last_active timestamp."""
        self.history.append(Message(role=role, content=content))
        self.last_active = time.time()

    def get_recent_history(self, max_turns: int) -> list[Message]:
        """Return the most recent messages, up to max_turns."""
        return self.history[-max_turns:]

    def format_history(self, max_turns: int) -> str:
        """Format recent history as a readable string for prompt injection."""
        recent = self.get_recent_history(max_turns)
        if not recent:
            return ""

        lines: list[str] = []
        for msg in recent:
            role_label = "Player" if msg.role == "player" else "NPC"
            lines.append(f"{role_label}: {msg.content}")
        return "\n".join(lines)


class ContextManager:
    """Manages conversation sessions with TTL-based expiry.

    The game engine sends a session_id with each request. The context
    manager maintains conversation history per session so multi-turn
    dialogue works without client-side state management.
    """

    def __init__(self) -> None:
        self.config = get_config()
        self._sessions: dict[str, Session] = {}

    def get_or_create_session(self, session_id: str, character_id: str) -> Session:
        """Get an existing session or create a new one."""
        self._expire_stale_sessions()

        if session_id in self._sessions:
            session = self._sessions[session_id]
            # If character changed, start fresh history
            if session.character_id != character_id:
                logger.info(
                    "session_character_changed",
                    session_id=session_id,
                    old_character=session.character_id,
                    new_character=character_id,
                )
                session = Session(session_id=session_id, character_id=character_id)
                self._sessions[session_id] = session
            return session

        session = Session(session_id=session_id, character_id=character_id)
        self._sessions[session_id] = session
        logger.info("session_created", session_id=session_id, character_id=character_id)
        return session

    def reset_session(self, session_id: str) -> bool:
        """Clear a session's conversation history. Returns True if session existed."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("session_reset", session_id=session_id)
            return True
        return False

    def _expire_stale_sessions(self) -> None:
        """Remove sessions that have exceeded their TTL."""
        now = time.time()
        ttl = self.config.api.session_ttl_seconds
        expired = [
            sid for sid, session in self._sessions.items() if now - session.last_active > ttl
        ]
        for sid in expired:
            del self._sessions[sid]
            logger.info("session_expired", session_id=sid)

    @property
    def active_session_count(self) -> int:
        """Return the number of active sessions."""
        self._expire_stale_sessions()
        return len(self._sessions)
