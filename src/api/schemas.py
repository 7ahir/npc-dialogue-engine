"""Pydantic request/response models for the NPC Dialogue API."""

from uuid import uuid4

from pydantic import BaseModel, Field


class DialogueRequest(BaseModel):
    """Request to generate NPC dialogue."""

    player_message: str = Field(
        ..., min_length=1, max_length=1000, description="The player's input message"
    )
    character_id: str = Field(..., pattern=r"^[a-z_]+$", description="NPC character identifier")
    session_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Session ID for multi-turn context",
    )
    use_tot: bool = Field(
        default=False,
        description="Use Tree of Thoughts for complex scenarios (slower)",
    )


class DialogueResponseSchema(BaseModel):
    """Response from NPC dialogue generation."""

    npc_response: str
    intent: str
    confidence: float
    sentiment: float
    lore_refs: list[str]
    latency_ms: float
    model_version: str


class CharacterSummary(BaseModel):
    """Brief character info for listing."""

    id: str
    name: str
    role: str


class CharacterDetail(CharacterSummary):
    """Full character info including personality and phrases."""

    description: str
    personality_traits: list[str]
    example_phrases: list[str]


class HealthResponse(BaseModel):
    """System health check response."""

    status: str  # "healthy" | "degraded"
    model_loaded: bool
    active_sessions: int
    version: str


class SessionResetResponse(BaseModel):
    """Response from session reset."""

    session_id: str
    reset: bool
