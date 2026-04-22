# Architecture

This document covers the *why* behind the system — the design principles that shape the code, the request flow, the trade-offs taken, and where the seams are if you want to swap a piece out.

For *what* the system does and how to run it, see the [README](../README.md). For evaluation methodology, see the [model card](model_card.md).

---

## Design Principles

Five principles drive every decision below. When two of them conflict, the order here is the tie-breaker.

1. **The pipeline is the product.** Game NPCs need consistent behavior, not just plausible text. The intent → RAG → generation pipeline exists because each stage gives the game engine a structured signal (intent label, retrieved lore refs, latency budget) that downstream gameplay code can act on.
2. **Develop CPU-only, deploy GPU.** A `MockDialogueModel` makes the entire pipeline runnable without torch — tests run in seconds, contributors don't need a GPU, and the same code path serves real inference once the LoRA adapter is loaded. The mock is not a stub; it's a deterministic implementation of the same interface.
3. **Graceful degradation over hard failure.** If ChromaDB is missing, the pipeline runs without RAG. If the intent classifier crashes, intent defaults to `social` with confidence 0.0 and the response still goes out. A game can't pause for a stack trace.
4. **Observability is not optional.** Per-stage tracing, structured JSON logs, Prometheus metrics, and a glass-box demo are all in the box from day one. Every commit message is a span you can inspect; every request produces a trace you can pivot to.
5. **Composition over configuration.** Every pipeline stage is constructor-injected. Want to A/B a different retriever? Pass it in. Want to wrap the LLM with a guardrail? Wrap it in `DialogueModel` and inject. No registry, no plugin system, just constructors.

---

## Request Flow

```
                 ┌────────────────────────────────┐
                 │  POST /api/v1/dialogue         │
                 │  { player_message, character } │
                 └────────────────────────────────┘
                                │
                                ▼
                ┌─────────────────────────────────┐
                │       DialoguePipeline          │
                │                                 │
   ┌────────────┼─[span: session]─────────────┐   │
   │ Get / create ContextManager session       │  │
   └────────────┼───────────────────────────────┘  │
                │                                 │
   ┌────────────┼─[span: intent ~5ms]──────────┐  │
   │ DistilBERT zero-shot →                    │  │
   │   intent ∈ {quest, trade, lore, ...}      │  │
   │   confidence, sentiment                   │  │
   └────────────┼───────────────────────────────┘  │
                │                                 │
   ┌────────────┼─[span: retrieval ~10ms]──────┐  │
   │ MiniLM embed(query) → ChromaDB top-K      │  │
   │   chunks above relevance_threshold        │  │
   │   formatted as `lore_context`             │  │
   └────────────┼───────────────────────────────┘  │
                │                                 │
   ┌────────────┼─[span: prompt ~1ms]──────────┐  │
   │ Jinja2 render: persona + lore + history   │  │
   │   + player message → chat messages list   │  │
   └────────────┼───────────────────────────────┘  │
                │                                 │
   ┌────────────┼─[span: generation 200-800ms]─┐  │
   │ Qwen 2.5-3B + LoRA adapter (4-bit NF4)    │  │
   │   OR MockDialogueModel (deterministic)    │  │
   │   OR Tree-of-Thoughts (3 candidates)      │  │
   └────────────┼───────────────────────────────┘  │
                │                                 │
                ▼                                 │
   Update session, finish trace, log, respond ────┘
                │
                ▼
   ┌─────────────────────────────────┐
   │ DialogueResponse                │
   │   npc_response, intent, ...     │
   │   trace_id ──────► /traces/{id} │
   └─────────────────────────────────┘
```

Each `[span: …]` is a real `TraceRecorder.span()` block — the per-stage durations are queryable via `/api/v1/traces/summary`, not just guessed at.

---

## The Mock Model Strategy

The single most useful design decision in the project. `MockDialogueModel` implements `DialogueModel` (`generate`, `generate_stream`, `model_version`), reads the character's `example_phrases` and `personality_traits` from the YAML config, and produces a deterministic response keyed off the player message and intent.

**What it gives you:**

- **Tests run in 40s** for the full 170-test suite, with no model downloads.
- **CI is fast and free** — GitHub Actions doesn't need a GPU runner.
- **Contributors can iterate** on prompts, retrieval, sessions, evaluation, and tracing without a GPU on their laptop.
- **The Gradio demo works on CPU** — you can show it to anyone without spinning up infra.

**What it doesn't give you:** real text quality. The mock is for plumbing, not for evaluation of the LLM itself. The real Qwen + LoRA path is selected by `DIALOGUE_MODEL_MODE=transformers` (or `auto`, which picks transformers if torch is installed).

The dispatch lives in `src/models/dialogue_model.py::create_dialogue_model` — three modes (`mock`, `transformers`, `auto`) make the pivot a single env var.

---

## RAG Tuning Trade-offs

The retrieval defaults are in `RAGConfig`:

| Setting | Default | Why this value |
|---|---|---|
| `embedding_model` | `all-MiniLM-L6-v2` | Small (~80 MB), fast on CPU, strong enough on game-lore prose. Bigger models (e.g. mpnet) cost ~3× embed latency for marginal gain on this corpus. |
| `chunk_size` | 512 chars | Lore documents are mostly paragraph-sized. 512 keeps a chunk to one idea (a faction, a location, a war), which makes retrieval cleaner than chunking by token count. |
| `chunk_overlap` | 64 | Enough to keep a sentence intact across a chunk boundary; small enough not to bloat the index. |
| `top_k` | 3 | Above 3, the prompt fills with weakly-relevant lore that confuses generation. Below 3, sparse coverage on multi-topic queries. |
| `relevance_threshold` | 0.35 | Tuned on the 20-example golden eval set. Lower → more hallucinated grounding; higher → empty `lore_context` for legitimate queries. |

The thresholds are tied to the embedding model's distance distribution. Swap the model and they need re-tuning — re-run `make eval` after any change to `RAG_EMBEDDING_MODEL`.

---

## Latency Breakdown

Numbers from `GET /api/v1/traces/summary` against the mock model on an M1 MacBook (no GPU). The shape — not the absolute values — is what matters; real LLM generation dominates everything else.

| Stage | p50 | p95 | Notes |
|---|---:|---:|---|
| `session` | <1 ms | <1 ms | In-memory dict lookup. |
| `intent` | 4 ms | 8 ms | DistilBERT zero-shot, 7 candidate labels. CPU. |
| `retrieval` | 9 ms | 14 ms | MiniLM embed + Chroma query + format. |
| `prompt` | <1 ms | 2 ms | Jinja2 template render. |
| `generation` (mock) | <1 ms | 1 ms | Lookup + format. |
| `generation` (Qwen 3B, 4-bit, GPU) | ~250 ms | ~600 ms | From earlier exploratory runs; not in CI. |
| **Total (mock)** | ~15 ms | ~25 ms | |
| **Total (real)** | ~270 ms | ~620 ms | Generation is ~95% of wall-clock. |

The first three stages run sequentially on every request even when generation streams. That's a deliberate choice: the game engine wants intent + lore refs *before* the first token arrives so it can stage animations or update quest state — see `process_stream` in `src/pipeline/dialogue_pipeline.py`.

---

## Trade-offs Made

**ChromaDB instead of pgvector / FAISS.** Chroma runs as an embedded process by default with zero ops, which fits the "demo + portfolio" target. For production at game scale you'd want pgvector (joins onto your existing player/quest data) or a managed vector DB.

**LoRA instead of full fine-tune.** A 3B base + a few-MB adapter ships in CI artifacts and loads in seconds. Full fine-tunes would lock the project to one base model and force a full reload to swap characters / personalities.

**In-process trace store instead of OTel.** Per the principles: observability is non-optional, but pulling in OTel + a collector for a single-process demo is overkill. The `TraceRecorder` / `TraceStore` shape is small enough to swap for an OTel exporter without changing the call sites in `dialogue_pipeline.py`.

**Tree of Thoughts as a flag, not a default.** ToT triples generation cost. It's wired through `use_tot=True` for cases where quality matters more than latency (cinematic dialogue, quest-critical exchanges). Off by default.

**Gradio for the demo, not React.** The demo is for showing the pipeline — Gradio's `Blocks` lets the "glass box" panel (intent, sentiment, retrieved lore, latency) sit next to the chatbot in ~150 lines. A React + custom backend would be 10× the code for the same demo value.

**Sync API on top of FastAPI.** The pipeline is CPU/GPU bound, not I/O bound. Wrapping it in `async def` would just move the work onto the event loop without parallelism. SSE streaming is the only place async pays off, and it's used there.

---

## Where to Read Next

- `src/pipeline/dialogue_pipeline.py` — the orchestration is short; read it first.
- `src/models/dialogue_model.py` — `DialogueModel` interface + Mock vs Transformers selection.
- `src/utils/tracing.py` — the `TraceRecorder` / `TraceStore` shape.
- `src/api/routes.py` — what's actually exposed.
- `docs/case-study.md` — the engineering narrative: what was built, what was learned, what's next.
