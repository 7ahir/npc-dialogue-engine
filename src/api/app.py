"""FastAPI application factory for the NPC Dialogue Engine."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware import MetricsMiddleware
from src.api.routes import router
from src.models.dialogue_model import create_dialogue_model
from src.models.intent_classifier import IntentClassifier
from src.pipeline.context_manager import ContextManager
from src.pipeline.dialogue_pipeline import DialoguePipeline
from src.pipeline.prompt_templates import PromptBuilder
from src.rag.retriever import LoreRetriever
from src.utils.config import get_config
from src.utils.logging_config import get_logger, setup_logging
from src.utils.tracing import get_trace_store

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: initialize pipeline on startup, cleanup on shutdown."""
    setup_logging()
    config = get_config()

    logger.info("app_starting", model_mode=config.model.base_model)

    # Initialize all pipeline components
    prompt_builder = PromptBuilder()
    model = create_dialogue_model(
        config=config.model,
        character_loader=prompt_builder.character_loader,
    )
    context_manager = ContextManager()

    # Retriever and classifier may fail to initialize (missing index, etc.)
    # The pipeline handles this gracefully via _retrieve_safe / _classify_safe
    retriever = LoreRetriever()
    classifier = IntentClassifier()

    trace_store = get_trace_store()
    pipeline = DialoguePipeline(
        model=model,
        retriever=retriever,
        classifier=classifier,
        prompt_builder=prompt_builder,
        context_manager=context_manager,
        config=config,
        trace_store=trace_store,
    )

    # Store in app state for dependency injection via routes
    app.state.pipeline = pipeline
    app.state.context_manager = context_manager
    app.state.character_loader = prompt_builder.character_loader
    app.state.trace_store = trace_store

    logger.info(
        "app_started",
        model_version=model.model_version,
        characters=prompt_builder.character_loader.list_characters(),
    )

    yield

    logger.info("app_shutdown")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    config = get_config()

    app = FastAPI(
        title="NPC Dialogue Engine",
        description="AI-powered dynamic NPC dialogue for games",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Prometheus + logging middleware
    app.add_middleware(MetricsMiddleware)

    # Routes
    app.include_router(router)

    return app
