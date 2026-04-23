# Case Study — NPC Dialogue Engine

A short narrative of what was built, why each engineering choice was made, and what I'd change with more time. Companion to [architecture.md](architecture.md), which covers the *system*; this covers the *project*.

---

## The Problem

Game NPCs are stuck in a tradeoff. Hand-authored dialogue trees are character-consistent and cheap at runtime, but rigid — the player asks something the writer didn't anticipate and the NPC says nothing useful. Drop in a raw LLM and you get the opposite: the NPC will respond to anything, but it forgets it's a blacksmith two turns in, hallucinates lore that contradicts the canon, and breaks character the moment a player tries to jailbreak it.

The interesting engineering problem isn't "make an LLM say words." It's: **can you build a pipeline that gives the game engine structured, character-consistent, lore-grounded dialogue with predictable latency, while staying observable and testable on a laptop?**

That question shaped every decision below.

---

## The Solution

A three-stage pipeline behind a thin FastAPI surface:

1. **Intent classification** (DistilBERT zero-shot, ~5 ms) → the game engine gets `{intent, confidence, sentiment}` *before* the LLM runs, so it can trigger animations, update quest state, or short-circuit hostile inputs.
2. **RAG retrieval** (MiniLM + ChromaDB, ~10 ms) → grounds the response in versioned lore documents instead of LLM training-data trivia.
3. **Generation** (Qwen 2.5-3B + LoRA adapter, 4-bit NF4) → produces the final line, with optional Tree-of-Thoughts for quality-critical exchanges.

Each stage is independently swappable, instrumented with per-stage tracing, and degrades gracefully if a downstream service is missing. The whole thing runs on CPU with a `MockDialogueModel` for development and CI; flipping `DIALOGUE_MODEL_MODE=transformers` switches to the real LLM path with no other code changes.

The full system lives at https://github.com/7ahir/npc-dialogue-engine.

---

## Engineering Signals

The things a reviewer would actually want to see, in one place:

| Signal | Where it shows up |
|---|---|
| **Tests** | 178 fast tests, ~50s. Markers separate fast / slow / integration. CI runs fast tier on every push. |
| **CI** | GitHub Actions: lint (ruff) → test → docker build, on every PR. |
| **Lint** | `ruff check` clean across `src/` and `tests/`. Selected rules: `E, F, I, N, W, UP, B, SIM`. |
| **Containerization** | Multi-stage `Dockerfile` (CPU + GPU variants). `docker-compose` with API + ChromaDB + Prometheus + Grafana. |
| **Observability** | Structured JSON logs (structlog), Prometheus middleware, in-process pipeline tracing (`/api/v1/traces`). |
| **API surface** | FastAPI with Pydantic v2 models, SSE streaming, OpenAPI auto-docs, CORS, health + metrics endpoints. |
| **Eval** | 7 automated metrics + 20-example golden set + 15-example adversarial set. CLI runner produces a JSON report. |
| **Training** | LoRA via PEFT/TRL, 4-bit NF4 quantization, synthetic data generator with 10 scenarios × N characters. |
| **Demo** | Gradio "glass box" UI: chatbot on the left, live pipeline internals (intent, sentiment, lore refs, latency) on the right. |
| **Docs** | README, architecture, case study (this), model card, contributing guide. |
| **Hygiene** | LICENSE, `.env.example`, `CONTRIBUTING.md`, badges, Makefile. |

---

## What I'd Do Differently With More Time / GPU Budget

Honest list. Order is roughly "biggest payoff first."

1. **Actually run the LoRA fine-tune.** The training pipeline is wired and tested against the mock, but I haven't spent a real GPU hour on the Qwen base. The eval numbers in the model card are a baseline against the un-tuned model — proper before/after numbers would be the most valuable single addition. The Colab notebook at [`notebooks/colab_finetune.ipynb`](../notebooks/colab_finetune.ipynb) packages the full train → merge → evaluate loop for a free T4; running it produces `results/eval_report_ft.json` next to the existing mock baseline.
2. **Replace the in-process `TraceStore` with OpenTelemetry.** The current ring-buffer is fine for a demo and avoids OTel overhead in CI, but in production you want spans to flow into Tempo / Honeycomb / your favorite backend. The `TraceRecorder` shape was kept small specifically so this is a swap, not a rewrite.
3. **Move from "intent labels" to "tool calls."** Right now intent classification is a separate model whose only output is a label. With function-calling-capable LLMs (or just structured-output prompting), the same signal could come out of the generation step itself, eliminating one network hop and one model load. Trade-off: harder to A/B the classifier.
4. **Replace ChromaDB with pgvector.** Chroma is great for "embedded, no ops." For production at game scale you want the vector index *next to* the player/quest tables so retrieval can do `WHERE faction_id = ?` joins.
5. **Adversarial robustness eval that goes deeper than 15 examples.** Current adversarial set covers the obvious shapes (prompt injection, jailbreak, off-topic). A proper red-team would generate adversarial inputs *per character* (e.g., things that try to break a specific persona) and track regression over fine-tuning runs.
6. **Latency budget enforcement.** Every stage has a known p95 — the pipeline could enforce a per-request budget and short-circuit (e.g., skip RAG if intent is `social`) when the budget is tight. Hooks are in place via the trace metadata; the policy is the missing piece.

---

## What This Project Was Really About

This started as "build an NPC dialogue system" but it's mostly an exercise in **engineering an ML pipeline that other people can read, run, and trust.** The LLM is the smallest part of the codebase. The bigger part is everything around it: the abstraction that lets you swap models, the tracing that tells you where time goes, the tests that run without a GPU, the eval that catches regressions, the Docker stack that brings it all up locally, the README that gets a reviewer to "I get it" in two minutes.

That ratio — small LLM, large supporting cast — is what production ML actually looks like. Building it this way was the point.
