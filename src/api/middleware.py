"""Prometheus metrics and structured request logging middleware."""

import time
import uuid

import structlog
from prometheus_client import Counter, Gauge, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# ─── Prometheus Metrics ─────────────────────────────────────────

REQUEST_COUNT = Counter(
    "npc_dialogue_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "npc_dialogue_request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

ACTIVE_SESSIONS = Gauge(
    "npc_dialogue_active_sessions",
    "Number of active dialogue sessions",
)

DIALOGUE_LATENCY = Histogram(
    "npc_dialogue_generation_latency_seconds",
    "Dialogue generation latency (pipeline only)",
    ["character_id"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

logger = structlog.get_logger("api.middleware")


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware that tracks request metrics and logs structured data."""

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])

        # Normalize the path for metric labels (avoid cardinality explosion)
        endpoint = self._normalize_path(request.url.path)

        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            REQUEST_COUNT.labels(
                method=request.method, endpoint=endpoint, status_code="500"
            ).inc()
            raise

        latency = time.perf_counter() - start

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=str(response.status_code),
        ).inc()

        REQUEST_LATENCY.labels(
            method=request.method, endpoint=endpoint
        ).observe(latency)

        logger.info(
            "request_completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=round(latency * 1000, 1),
        )

        response.headers["X-Request-ID"] = request_id
        return response

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Normalize URL path for metric labels.

        Replaces dynamic path segments to prevent metric cardinality explosion.
        /api/v1/characters/blacksmith -> /api/v1/characters/{id}
        /api/v1/sessions/abc123/reset -> /api/v1/sessions/{id}/reset
        """
        parts = path.strip("/").split("/")
        normalized = []
        for i, part in enumerate(parts):
            if i > 0 and parts[i - 1] in ("characters", "sessions"):
                normalized.append("{id}")
            else:
                normalized.append(part)
        return "/" + "/".join(normalized)
