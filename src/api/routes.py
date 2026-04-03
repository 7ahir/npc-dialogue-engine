"""API route definitions for the NPC Dialogue Engine."""

from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sse_starlette.sse import EventSourceResponse

from src.api.middleware import ACTIVE_SESSIONS
from src.api.schemas import (
    CharacterDetail,
    CharacterSummary,
    DialogueRequest,
    DialogueResponseSchema,
    HealthResponse,
    SessionResetResponse,
)
from src.pipeline.dialogue_pipeline import DialoguePipeline
from src.pipeline.prompt_templates import CharacterLoader

router = APIRouter(prefix="/api/v1", tags=["dialogue"])


def _get_pipeline(request: Request) -> DialoguePipeline:
    """Extract the dialogue pipeline from app state."""
    return request.app.state.pipeline


def _get_character_loader(request: Request) -> CharacterLoader:
    """Extract the character loader from app state."""
    return request.app.state.character_loader


# ─── Dialogue Endpoints ────────────────────────────────────────


@router.post("/dialogue", response_model=DialogueResponseSchema)
async def generate_dialogue(body: DialogueRequest, request: Request) -> DialogueResponseSchema:
    """Generate an NPC dialogue response.

    Takes a player message, identifies the character, retrieves relevant
    lore, and generates a character-consistent response.
    """
    pipeline = _get_pipeline(request)
    loader = _get_character_loader(request)

    # Validate character exists
    if body.character_id not in loader.list_characters():
        raise HTTPException(
            status_code=404,
            detail=f"Character '{body.character_id}' not found. "
            f"Available: {loader.list_characters()}",
        )

    result = pipeline.process(
        player_message=body.player_message,
        character_id=body.character_id,
        session_id=body.session_id,
        use_tot=body.use_tot,
    )

    # Update active sessions gauge
    ACTIVE_SESSIONS.set(pipeline.context_manager.active_session_count)

    return DialogueResponseSchema(
        npc_response=result.npc_response,
        intent=result.intent,
        confidence=result.confidence,
        sentiment=result.sentiment,
        lore_refs=result.lore_refs,
        latency_ms=result.latency_ms,
        model_version=result.model_version,
    )


@router.post("/dialogue/stream")
async def stream_dialogue(body: DialogueRequest, request: Request) -> EventSourceResponse:
    """Stream NPC dialogue response token-by-token via SSE.

    Performs intent classification and RAG retrieval upfront, then streams
    the generated response. Ideal for typewriter-style UI effects in games.
    """
    pipeline = _get_pipeline(request)
    loader = _get_character_loader(request)

    if body.character_id not in loader.list_characters():
        raise HTTPException(
            status_code=404,
            detail=f"Character '{body.character_id}' not found.",
        )

    async def event_generator() -> AsyncIterator[dict[str, str]]:
        for token in pipeline.process_stream(
            player_message=body.player_message,
            character_id=body.character_id,
            session_id=body.session_id,
        ):
            yield {"event": "token", "data": token}
        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_generator())


# ─── Character Endpoints ───────────────────────────────────────


@router.get("/characters", response_model=list[CharacterSummary])
async def list_characters(request: Request) -> list[CharacterSummary]:
    """List all available NPC characters."""
    loader = _get_character_loader(request)
    characters = []
    for char_id in loader.list_characters():
        data = loader.load(char_id)
        characters.append(
            CharacterSummary(id=data["id"], name=data["name"], role=data["role"])
        )
    return characters


@router.get("/characters/{character_id}", response_model=CharacterDetail)
async def get_character(character_id: str, request: Request) -> CharacterDetail:
    """Get detailed information about a specific NPC character."""
    loader = _get_character_loader(request)

    if character_id not in loader.list_characters():
        raise HTTPException(
            status_code=404,
            detail=f"Character '{character_id}' not found.",
        )

    data = loader.load(character_id)
    return CharacterDetail(
        id=data["id"],
        name=data["name"],
        role=data["role"],
        description=data["description"],
        personality_traits=data["personality_traits"],
        example_phrases=data["example_phrases"],
    )


# ─── Session Endpoints ────────────────────────────────────────


@router.post("/sessions/{session_id}/reset", response_model=SessionResetResponse)
async def reset_session(session_id: str, request: Request) -> SessionResetResponse:
    """Clear conversation history for a session."""
    pipeline = _get_pipeline(request)
    was_reset = pipeline.context_manager.reset_session(session_id)
    ACTIVE_SESSIONS.set(pipeline.context_manager.active_session_count)
    return SessionResetResponse(session_id=session_id, reset=was_reset)


# ─── System Endpoints ─────────────────────────────────────────


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """System health check."""
    pipeline = _get_pipeline(request)
    return HealthResponse(
        status="healthy",
        model_loaded=pipeline.model is not None,
        active_sessions=pipeline.context_manager.active_session_count,
        version="0.1.0",
    )


@router.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
