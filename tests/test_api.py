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

os.environ["DIALOGUE_MODEL_MODE"] = "mock"


def _build_test_app() -> FastAPI:
    """Build a test app with pipeline injected directly (no lifespan needed)."""
    app = create_app()

    # Manually set up app state (mirrors what lifespan does)
    prompt_builder = PromptBuilder()
    model = MockDialogueModel(character_loader=prompt_builder.character_loader)
    context_manager = ContextManager()

    pipeline = DialoguePipeline(
        model=model,
        retriever=LoreRetriever(),
        prompt_builder=prompt_builder,
        context_manager=context_manager,
    )

    app.state.pipeline = pipeline
    app.state.context_manager = context_manager
    app.state.character_loader = prompt_builder.character_loader

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
