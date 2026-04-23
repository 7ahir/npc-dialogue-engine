"""Tests for the FastAPI endpoints."""

import os

import httpx
import pytest
from fastapi import FastAPI

from src.api.app import create_app
from src.models.dialogue_model import MockDialogueModel
from src.pipeline.context_manager import ContextManager
from src.pipeline.dialogue_pipeline import DialoguePipeline
from src.pipeline.prompt_templates import PromptBuilder
from src.rag.retriever import LoreRetriever
from src.utils.tracing import get_trace_store

os.environ["DIALOGUE_MODEL_MODE"] = "mock"


def _build_test_app() -> FastAPI:
    """Build a test app with pipeline injected directly (no lifespan needed)."""
    app = create_app()

    # Manually set up app state (mirrors what lifespan does)
    prompt_builder = PromptBuilder()
    model = MockDialogueModel(character_loader=prompt_builder.character_loader)
    context_manager = ContextManager()

    trace_store = get_trace_store()
    pipeline = DialoguePipeline(
        model=model,
        retriever=LoreRetriever(),
        prompt_builder=prompt_builder,
        context_manager=context_manager,
        trace_store=trace_store,
    )

    app.state.pipeline = pipeline
    app.state.context_manager = context_manager
    app.state.character_loader = prompt_builder.character_loader
    app.state.trace_store = trace_store

    return app


@pytest.fixture
async def client():
    """Create an async test client with pipeline pre-initialized."""
    app = _build_test_app()
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert isinstance(data["active_sessions"], int)

    @pytest.mark.asyncio
    async def test_health_model_loaded(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/api/v1/health")
        assert resp.json()["model_loaded"] is True


class TestDialogueEndpoint:
    @pytest.mark.asyncio
    async def test_dialogue_success(self, client: httpx.AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/dialogue",
            json={
                "player_message": "Got any swords for sale?",
                "character_id": "blacksmith",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["npc_response"]) > 0
        assert data["model_version"] == "mock-v1"
        assert data["latency_ms"] > 0
        assert "intent" in data
        assert "sentiment" in data
        assert isinstance(data["lore_refs"], list)

    @pytest.mark.asyncio
    async def test_dialogue_invalid_character(self, client: httpx.AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/dialogue",
            json={
                "player_message": "Hello",
                "character_id": "nonexistent_npc",
            },
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_dialogue_missing_message(self, client: httpx.AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/dialogue",
            json={"character_id": "blacksmith"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_dialogue_empty_message(self, client: httpx.AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/dialogue",
            json={"player_message": "", "character_id": "blacksmith"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_dialogue_with_session(self, client: httpx.AsyncClient) -> None:
        session_id = "api-test-session"
        # First message
        resp1 = await client.post(
            "/api/v1/dialogue",
            json={
                "player_message": "Hello smith!",
                "character_id": "blacksmith",
                "session_id": session_id,
            },
        )
        assert resp1.status_code == 200

        # Second message (same session)
        resp2 = await client.post(
            "/api/v1/dialogue",
            json={
                "player_message": "What about armor?",
                "character_id": "blacksmith",
                "session_id": session_id,
            },
        )
        assert resp2.status_code == 200

    @pytest.mark.asyncio
    async def test_dialogue_invalid_character_id_format(self, client: httpx.AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/dialogue",
            json={
                "player_message": "Hello",
                "character_id": "Invalid-ID-123",
            },
        )
        assert resp.status_code == 422


class TestStreamingEndpoint:
    """Lock down the SSE contract: metadata frame first, tokens, done frame last.

    Pipeline-level tests already cover event ordering, but the route layer
    adds EventSourceResponse framing + DIALOGUE_LATENCY observation, so
    both deserve a dedicated HTTP-level test.
    """

    @pytest.mark.asyncio
    async def test_stream_dialogue_emits_metadata_then_tokens_then_done(
        self, client: httpx.AsyncClient
    ) -> None:
        import json as _json

        async with client.stream(
            "POST",
            "/api/v1/dialogue/stream",
            json={
                "player_message": "Got any swords?",
                "character_id": "blacksmith",
                "session_id": "stream-route-test",
            },
        ) as resp:
            assert resp.status_code == 200
            # sse-starlette announces text/event-stream
            assert "text/event-stream" in resp.headers.get("content-type", "")

            events: list[tuple[str, str]] = []
            current_event: str | None = None
            current_data: list[str] = []

            async for raw_line in resp.aiter_lines():
                # SSE frames: "event: <type>" then "data: <payload>" then blank.
                if raw_line.startswith("event:"):
                    current_event = raw_line.split(":", 1)[1].strip()
                elif raw_line.startswith("data:"):
                    current_data.append(raw_line.split(":", 1)[1].lstrip())
                elif raw_line == "" and current_event is not None:
                    events.append((current_event, "\n".join(current_data)))
                    current_event = None
                    current_data = []

        assert events, "no SSE frames received"
        assert events[0][0] == "metadata", f"first frame was {events[0][0]!r}"
        meta = _json.loads(events[0][1])
        assert "intent" in meta and "lore_refs" in meta and "trace_id" in meta
        assert meta["trace_id"], "metadata must include a non-empty trace_id"

        assert events[-1][0] == "done"
        done = _json.loads(events[-1][1])
        assert done.get("trace_id") == meta["trace_id"], "done frame trace_id must match metadata"
        assert isinstance(done.get("latency_ms"), (int, float))

        token_frames = [ev for ev in events[1:-1] if ev[0] == "token"]
        assert token_frames, "no token frames between metadata and done"

    @pytest.mark.asyncio
    async def test_stream_dialogue_invalid_character_returns_404(
        self, client: httpx.AsyncClient
    ) -> None:
        resp = await client.post(
            "/api/v1/dialogue/stream",
            json={"player_message": "Hello", "character_id": "nonexistent"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_stream_trace_is_retrievable_via_traces_endpoint(
        self, client: httpx.AsyncClient
    ) -> None:
        """The trace_id emitted in metadata must resolve at GET /traces/{id}."""
        import json as _json

        trace_id: str | None = None
        async with client.stream(
            "POST",
            "/api/v1/dialogue/stream",
            json={"player_message": "Hi", "character_id": "tavern_keeper"},
        ) as resp:
            async for raw_line in resp.aiter_lines():
                if raw_line.startswith("data:") and trace_id is None:
                    try:
                        payload = _json.loads(raw_line.split(":", 1)[1].lstrip())
                        trace_id = payload.get("trace_id")
                    except _json.JSONDecodeError:
                        pass
                if trace_id:
                    # Drain the rest so the server-side trace gets finalized
                    continue

        assert trace_id, "streaming endpoint did not emit a trace_id"
        resp = await client.get(f"/api/v1/traces/{trace_id}")
        assert resp.status_code == 200, f"trace {trace_id} not retrievable"
        data = resp.json()
        assert data["metadata"].get("stream") is True
        span_names = {s["name"] for s in data["spans"]}
        assert {"intent", "generation"} <= span_names


class TestCharacterEndpoints:
    @pytest.mark.asyncio
    async def test_list_characters(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/api/v1/characters")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 3
        ids = [c["id"] for c in data]
        assert "blacksmith" in ids
        assert "tavern_keeper" in ids
        assert "mysterious_sage" in ids

    @pytest.mark.asyncio
    async def test_list_characters_have_names(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/api/v1/characters")
        for char in resp.json():
            assert "name" in char
            assert "role" in char
            assert len(char["name"]) > 0

    @pytest.mark.asyncio
    async def test_get_character_detail(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/api/v1/characters/blacksmith")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Grenn Ironheart"
        assert len(data["personality_traits"]) > 0
        assert len(data["example_phrases"]) > 0
        assert len(data["description"]) > 0

    @pytest.mark.asyncio
    async def test_get_character_not_found(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/api/v1/characters/nonexistent")
        assert resp.status_code == 404


class TestSessionEndpoints:
    @pytest.mark.asyncio
    async def test_reset_existing_session(self, client: httpx.AsyncClient) -> None:
        # Create a session first
        await client.post(
            "/api/v1/dialogue",
            json={
                "player_message": "Hello",
                "character_id": "blacksmith",
                "session_id": "reset-test",
            },
        )
        # Reset it
        resp = await client.post("/api/v1/sessions/reset-test/reset")
        assert resp.status_code == 200
        assert resp.json()["reset"] is True

    @pytest.mark.asyncio
    async def test_reset_nonexistent_session(self, client: httpx.AsyncClient) -> None:
        resp = await client.post("/api/v1/sessions/doesnt-exist/reset")
        assert resp.status_code == 200
        assert resp.json()["reset"] is False


class TestMetricsEndpoint:
    @pytest.mark.asyncio
    async def test_metrics_returns_prometheus(self, client: httpx.AsyncClient) -> None:
        resp = await client.get("/api/v1/metrics")
        assert resp.status_code == 200
        text = resp.text
        assert "npc_dialogue_requests_total" in text
