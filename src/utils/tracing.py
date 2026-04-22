"""Per-stage pipeline tracing.

The pipeline already logs total request latency, but a single number hides
where the time actually goes. ``TraceRecorder`` wraps each stage in a span,
records duration + arbitrary metadata, and ``TraceStore`` keeps the last N
completed traces in memory so the API can expose them for inspection.

This is deliberately lightweight (no OpenTelemetry dependency, no exporter
plumbing). The goal is "I can curl /traces/summary and see where time goes,"
not full distributed tracing. If we ever need OTel, swap the recorder
backend without changing call sites.
"""

from __future__ import annotations

import statistics
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from typing import Any
from uuid import uuid4


@dataclass
class Span:
    """A single timed stage within a trace."""

    name: str
    start_ms: float  # ms since trace start
    duration_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start_ms": round(self.start_ms, 3),
            "duration_ms": round(self.duration_ms, 3),
            "metadata": self.metadata,
        }


@dataclass
class Trace:
    """A completed pipeline trace: ordered spans plus summary metadata."""

    trace_id: str
    started_at: str  # ISO 8601 UTC
    total_ms: float
    spans: list[Span] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "started_at": self.started_at,
            "total_ms": round(self.total_ms, 3),
            "spans": [s.to_dict() for s in self.spans],
            "metadata": self.metadata,
        }


class TraceRecorder:
    """Per-request recorder. Use one instance per pipeline invocation.

    Usage::

        recorder = TraceRecorder()
        with recorder.span("intent"):
            ...
        with recorder.span("retrieval", top_k=3):
            ...
        trace = recorder.finish(character_id="blacksmith")
    """

    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or uuid4().hex[:16]
        self._t0 = time.perf_counter()
        self._started_at = datetime.now(UTC).isoformat()
        self._spans: list[Span] = []

    @contextmanager
    def span(self, name: str, **metadata: Any):
        """Time a code block as a span. Metadata is merged with anything
        added later via the returned mutable dict (yielded value)."""
        start_ms = (time.perf_counter() - self._t0) * 1000
        meta: dict[str, Any] = dict(metadata)
        t_start = time.perf_counter()
        try:
            yield meta
        finally:
            duration_ms = (time.perf_counter() - t_start) * 1000
            self._spans.append(
                Span(name=name, start_ms=start_ms, duration_ms=duration_ms, metadata=meta)
            )

    def finish(self, **metadata: Any) -> Trace:
        total_ms = (time.perf_counter() - self._t0) * 1000
        return Trace(
            trace_id=self.trace_id,
            started_at=self._started_at,
            total_ms=total_ms,
            spans=list(self._spans),
            metadata=dict(metadata),
        )


class TraceStore:
    """In-memory ring buffer of recent traces.

    Bounded so a long-running process can't OOM. Thread-safe for the
    add/read mix that the API and pipeline produce.
    """

    DEFAULT_MAX = 200

    def __init__(self, max_traces: int = DEFAULT_MAX) -> None:
        self._buf: deque[Trace] = deque(maxlen=max_traces)
        self._lock = RLock()

    def add(self, trace: Trace) -> None:
        with self._lock:
            self._buf.append(trace)

    def get(self, trace_id: str) -> Trace | None:
        with self._lock:
            for t in reversed(self._buf):
                if t.trace_id == trace_id:
                    return t
        return None

    def list(self, limit: int = 50) -> list[Trace]:
        """Most recent first."""
        with self._lock:
            return list(reversed(list(self._buf)[-limit:]))

    def summary(self) -> dict[str, Any]:
        """Aggregate stats: count, p50/p95 total latency, per-span p50/p95.

        Returns an empty-ish payload when the store has no traces (rather
        than raising), so the endpoint is safe to call on a fresh process.
        """
        with self._lock:
            traces = list(self._buf)

        if not traces:
            return {"count": 0, "total_ms": {}, "spans": {}}

        totals = [t.total_ms for t in traces]
        per_span: dict[str, list[float]] = {}
        for t in traces:
            for s in t.spans:
                per_span.setdefault(s.name, []).append(s.duration_ms)

        return {
            "count": len(traces),
            "total_ms": _percentiles(totals),
            "spans": {name: _percentiles(vals) for name, vals in per_span.items()},
        }

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


def _percentiles(values: list[float]) -> dict[str, float]:
    """p50 / p95 / max / count for a non-empty list."""
    if not values:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    sorted_vals = sorted(values)
    return {
        "count": len(values),
        "p50": round(statistics.median(sorted_vals), 3),
        "p95": round(_quantile(sorted_vals, 0.95), 3),
        "max": round(sorted_vals[-1], 3),
    }


def _quantile(sorted_vals: list[float], q: float) -> float:
    """Nearest-rank percentile. Good enough for ops dashboards; avoids
    pulling numpy just for this."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    # Nearest-rank: idx = ceil(q * N) - 1, clamped
    n = len(sorted_vals)
    rank = q * n
    idx = int(rank) - 1 + (1 if rank % 1 else 0)
    idx = max(0, min(n - 1, idx))
    return sorted_vals[idx]


# ─── Module-level singleton ────────────────────────────────────────

_default_store: TraceStore | None = None


def get_trace_store() -> TraceStore:
    """Return the process-wide default TraceStore, creating it if needed."""
    global _default_store
    if _default_store is None:
        _default_store = TraceStore()
    return _default_store


def reset_trace_store() -> None:
    """Reset the singleton — primarily for tests."""
    global _default_store
    _default_store = None
