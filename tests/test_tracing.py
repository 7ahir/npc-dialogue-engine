"""Tests for the pipeline tracing module."""

import time

import pytest

from src.utils.tracing import (
    Span,
    Trace,
    TraceRecorder,
    TraceStore,
    _percentiles,
    get_trace_store,
    reset_trace_store,
)

# ─── TraceRecorder ────────────────────────────────────────────────


class TestTraceRecorder:
    def test_generates_trace_id(self) -> None:
        r = TraceRecorder()
        assert isinstance(r.trace_id, str)
        assert len(r.trace_id) > 0

    def test_accepts_explicit_trace_id(self) -> None:
        r = TraceRecorder(trace_id="custom-id")
        assert r.trace_id == "custom-id"

    def test_unique_trace_ids(self) -> None:
        ids = {TraceRecorder().trace_id for _ in range(50)}
        assert len(ids) == 50

    def test_finish_with_no_spans(self) -> None:
        trace = TraceRecorder().finish()
        assert isinstance(trace, Trace)
        assert trace.spans == []
        assert trace.total_ms >= 0

    def test_span_records_duration(self) -> None:
        r = TraceRecorder()
        with r.span("work"):
            time.sleep(0.01)
        trace = r.finish()
        assert len(trace.spans) == 1
        assert trace.spans[0].name == "work"
        assert trace.spans[0].duration_ms >= 9  # ~10ms, allow scheduler slop

    def test_span_metadata_initial(self) -> None:
        r = TraceRecorder()
        with r.span("retrieval", top_k=3, strategy="dense"):
            pass
        trace = r.finish()
        assert trace.spans[0].metadata == {"top_k": 3, "strategy": "dense"}

    def test_span_metadata_mutable_inside_block(self) -> None:
        r = TraceRecorder()
        with r.span("intent") as meta:
            meta["confidence"] = 0.87
            meta["label"] = "quest"
        trace = r.finish()
        assert trace.spans[0].metadata["confidence"] == 0.87
        assert trace.spans[0].metadata["label"] == "quest"

    def test_multiple_spans_in_order(self) -> None:
        r = TraceRecorder()
        with r.span("a"):
            pass
        with r.span("b"):
            pass
        with r.span("c"):
            pass
        trace = r.finish()
        assert [s.name for s in trace.spans] == ["a", "b", "c"]

    def test_span_start_ms_is_relative_and_monotonic(self) -> None:
        r = TraceRecorder()
        with r.span("first"):
            time.sleep(0.005)
        with r.span("second"):
            pass
        trace = r.finish()
        assert trace.spans[0].start_ms >= 0
        assert trace.spans[1].start_ms >= trace.spans[0].start_ms

    def test_span_records_even_on_exception(self) -> None:
        r = TraceRecorder()
        with pytest.raises(RuntimeError), r.span("boom"):
            raise RuntimeError("fail")
        trace = r.finish()
        assert len(trace.spans) == 1
        assert trace.spans[0].name == "boom"

    def test_finish_metadata(self) -> None:
        trace = TraceRecorder().finish(character_id="blacksmith", session_id="abc")
        assert trace.metadata == {"character_id": "blacksmith", "session_id": "abc"}

    def test_total_ms_covers_all_spans(self) -> None:
        r = TraceRecorder()
        with r.span("a"):
            time.sleep(0.01)
        with r.span("b"):
            time.sleep(0.01)
        trace = r.finish()
        assert trace.total_ms >= sum(s.duration_ms for s in trace.spans) - 1  # rounding slack


# ─── Trace / Span dict serialization ──────────────────────────────


class TestTraceSerialization:
    def test_span_to_dict(self) -> None:
        s = Span(name="x", start_ms=1.234, duration_ms=5.6789, metadata={"k": "v"})
        d = s.to_dict()
        assert d["name"] == "x"
        assert d["start_ms"] == 1.234
        assert d["duration_ms"] == 5.679
        assert d["metadata"] == {"k": "v"}

    def test_trace_to_dict(self) -> None:
        r = TraceRecorder(trace_id="t1")
        with r.span("intent"):
            pass
        trace = r.finish(character_id="sage")
        d = trace.to_dict()
        assert d["trace_id"] == "t1"
        assert "started_at" in d
        assert d["metadata"] == {"character_id": "sage"}
        assert len(d["spans"]) == 1
        assert d["spans"][0]["name"] == "intent"


# ─── TraceStore ───────────────────────────────────────────────────


class TestTraceStore:
    def test_add_and_get(self) -> None:
        store = TraceStore()
        trace = TraceRecorder(trace_id="abc").finish()
        store.add(trace)
        assert store.get("abc") is trace

    def test_get_missing_returns_none(self) -> None:
        assert TraceStore().get("nope") is None

    def test_list_most_recent_first(self) -> None:
        store = TraceStore()
        for i in range(5):
            store.add(TraceRecorder(trace_id=f"t{i}").finish())
        ids = [t.trace_id for t in store.list()]
        assert ids == ["t4", "t3", "t2", "t1", "t0"]

    def test_list_respects_limit(self) -> None:
        store = TraceStore()
        for i in range(10):
            store.add(TraceRecorder(trace_id=f"t{i}").finish())
        assert len(store.list(limit=3)) == 3

    def test_ring_buffer_evicts_oldest(self) -> None:
        store = TraceStore(max_traces=3)
        for i in range(5):
            store.add(TraceRecorder(trace_id=f"t{i}").finish())
        ids = {t.trace_id for t in store.list(limit=10)}
        assert ids == {"t2", "t3", "t4"}
        assert store.get("t0") is None
        assert store.get("t4") is not None

    def test_len(self) -> None:
        store = TraceStore()
        assert len(store) == 0
        store.add(TraceRecorder().finish())
        assert len(store) == 1

    def test_clear(self) -> None:
        store = TraceStore()
        store.add(TraceRecorder().finish())
        store.clear()
        assert len(store) == 0

    def test_summary_empty(self) -> None:
        s = TraceStore().summary()
        assert s == {"count": 0, "total_ms": {}, "spans": {}}

    def test_summary_aggregates_per_span(self) -> None:
        store = TraceStore()
        # Build 3 traces, each with intent + retrieval spans
        for _ in range(3):
            r = TraceRecorder()
            with r.span("intent"):
                time.sleep(0.005)
            with r.span("retrieval"):
                time.sleep(0.005)
            store.add(r.finish())

        summary = store.summary()
        assert summary["count"] == 3
        assert "p50" in summary["total_ms"]
        assert "intent" in summary["spans"]
        assert "retrieval" in summary["spans"]
        assert summary["spans"]["intent"]["count"] == 3


# ─── Percentiles ──────────────────────────────────────────────────


class TestPercentiles:
    def test_empty(self) -> None:
        assert _percentiles([]) == {"count": 0, "p50": 0.0, "p95": 0.0, "max": 0.0}

    def test_single_value(self) -> None:
        p = _percentiles([42.0])
        assert p["p50"] == 42.0
        assert p["p95"] == 42.0
        assert p["max"] == 42.0
        assert p["count"] == 1

    def test_basic_distribution(self) -> None:
        vals = [float(i) for i in range(1, 101)]  # 1..100
        p = _percentiles(vals)
        assert p["count"] == 100
        assert p["max"] == 100.0
        assert 49 <= p["p50"] <= 51
        assert 94 <= p["p95"] <= 96


# ─── Module singleton ─────────────────────────────────────────────


class TestSingleton:
    def setup_method(self) -> None:
        reset_trace_store()

    def teardown_method(self) -> None:
        reset_trace_store()

    def test_returns_same_instance(self) -> None:
        a = get_trace_store()
        b = get_trace_store()
        assert a is b

    def test_reset_creates_new_instance(self) -> None:
        a = get_trace_store()
        reset_trace_store()
        b = get_trace_store()
        assert a is not b
