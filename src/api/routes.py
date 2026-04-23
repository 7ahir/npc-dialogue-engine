"""API route definitions for the NPC Dialogue Engine."""

from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sse_starlette.sse import EventSourceResponse

from src.api.middleware import ACTIVE_SESSIONS, DIALOGUE_LATENCY
from src.api.schemas import (
    CharacterDetail,
    CharacterSummary,
    DialogueRequest,
    DialogueResponseSchema,
    HealthResponse,
    SessionResetResponse,
    SpanSchema,
    TraceListEntry,
    TraceSchema,
    TraceSummaryResponse,
)
from src.pipeline.dialogue_pipeline import DialoguePipeline
from src.pipeline.prompt_templates import CharacterLoader
from src.utils.tracing import TraceStore

router = APIRouter(prefix="/api/v1", tags=["dialogue"])


def _get_pipeline(request: Request) -> DialoguePipeline:
    """Extract the dialogue pipeline from app state."""
    return request.app.state.pipeline


def _get_character_loader(request: Request) -> CharacterLoader:
    """Extract the character loader from app state."""
    return request.app.state.character_loader


def _get_trace_store(request: Request) -> TraceStore:
    """Extract the trace store from app state."""
    return request.app.state.trace_store


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

    # Pipeline-only latency (excludes HTTP/middleware overhead). Lets the
    # Grafana dashboard split "model time" from "FastAPI time" — the latter
    # was already exposed via REQUEST_LATENCY, but the per-character cut on
    # generation latency was defined and never observed until now.
    DIALOGUE_LATENCY.labels(character_id=body.character_id).observe(result.latency_ms / 1000.0)

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
        trace_id=result.trace_id,
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
        # stream_events emits a `metadata` frame first (intent + lore_refs),
        # then `token` frames, then a final `done` frame with latency. We
        # forward each one verbatim so the client gets framing info before
        # the typewriter starts.
        import json as _json

        for ev in pipeline.stream_events(
            player_message=body.player_message,
            character_id=body.character_id,
            session_id=body.session_id,
        ):
            if ev["event"] == "done":
                # Mirror the sync endpoint's per-character latency observation
                # so streaming requests show up in the same Prometheus series.
                try:
                    payload = _json.loads(ev["data"])
                    DIALOGUE_LATENCY.labels(character_id=body.character_id).observe(
                        payload.get("latency_ms", 0.0) / 1000.0
                    )
                except (ValueError, KeyError):
                    pass
            yield ev

    return EventSourceResponse(event_generator())


# ─── Character Endpoints ───────────────────────────────────────


@router.get("/characters", response_model=list[CharacterSummary])
async def list_characters(request: Request) -> list[CharacterSummary]:
    """List all available NPC characters."""
    loader = _get_character_loader(request)
    characters = []
    for char_id in loader.list_characters():
        data = loader.load(char_id)
        characters.append(CharacterSummary(id=data["id"], name=data["name"], role=data["role"]))
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


# ─── Tracing Endpoints ────────────────────────────────────────────


@router.get("/traces", response_model=list[TraceListEntry])
async def list_traces(request: Request, limit: int = 50) -> list[TraceListEntry]:
    """List recent pipeline traces (most recent first).

    Compact view — use ``GET /traces/{id}`` for the full per-span breakdown.
    """
    store = _get_trace_store(request)
    traces = store.list(limit=max(1, min(limit, 500)))
    return [
        TraceListEntry(
            trace_id=t.trace_id,
            started_at=t.started_at,
            total_ms=round(t.total_ms, 3),
            character_id=t.metadata.get("character_id"),
            intent=t.metadata.get("intent"),
            span_count=len(t.spans),
        )
        for t in traces
    ]


@router.get("/traces/summary", response_model=TraceSummaryResponse)
async def trace_summary(request: Request) -> TraceSummaryResponse:
    """Aggregate latency stats across the trace buffer (p50/p95 per stage).

    Useful sanity check during load testing — answers "where does the time
    actually go?" without scraping Prometheus.
    """
    store = _get_trace_store(request)
    s = store.summary()
    return TraceSummaryResponse(
        count=s["count"],
        total_ms=s["total_ms"],
        spans=s["spans"],
    )


@router.get("/traces/{trace_id}", response_model=TraceSchema)
async def get_trace(trace_id: str, request: Request) -> TraceSchema:
    """Fetch the full per-span breakdown for a single request."""
    store = _get_trace_store(request)
    trace = store.get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"Trace '{trace_id}' not found.")
    return TraceSchema(
        trace_id=trace.trace_id,
        started_at=trace.started_at,
        total_ms=round(trace.total_ms, 3),
        spans=[
            SpanSchema(
                name=s.name,
                start_ms=round(s.start_ms, 3),
                duration_ms=round(s.duration_ms, 3),
                metadata=s.metadata,
            )
            for s in trace.spans
        ],
        metadata=trace.metadata,
    )
