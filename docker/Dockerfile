# ─── Stage 1: Builder ───────────────────────────────────────────
# Install dependencies in an isolated layer for caching
FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml .
RUN pip install --no-cache-dir --prefix=/install .

# ─── Stage 2: Runtime ───────────────────────────────────────────
# Minimal runtime image — no build tools, no cache
FROM python:3.11-slim AS runtime

LABEL maintainer="Tahir T <7ahir@users.noreply.github.com>"
LABEL description="NPC Dialogue Engine — AI-powered dynamic NPC conversations"

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
WORKDIR /app
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY data/lore/ /app/data/lore/
COPY scripts/ /app/scripts/

# Create directory for model weights (mounted as volume)
RUN mkdir -p /app/models /app/chroma_data

# Non-root user for security
RUN useradd --create-home --shell /bin/bash npc && \
    chown -R npc:npc /app
USER npc

# Default environment
ENV DIALOGUE_MODEL_MODE=mock \
    LOG_LEVEL=INFO \
    LOG_FORMAT=json \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/api/v1/health'); r.raise_for_status()"

CMD ["uvicorn", "src.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
