# NPC Dialogue Engine

[![CI](https://github.com/7ahir/npc-dialogue-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/7ahir/npc-dialogue-engine/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**AI-powered dynamic NPC dialogue system for RPGs** — generates character-consistent, lore-grounded dialogue using a three-stage ML pipeline: intent classification, RAG retrieval, and fine-tuned LLM generation.

Built as a production-ready ML engineering portfolio project demonstrating end-to-end NLP/GenAI system design for game development.

---

## Architecture

```
Player Input
    │
    ▼
┌─────────────────────────────┐
│  Stage 1: Intent Classifier │  DistilBERT zero-shot (~5ms)
│  → intent label + sentiment │
└──────────────┬──────────────┘
               │
    ▼          ▼
┌──────────────────────────────────────────────┐
│  Stage 2: Context Assembly                    │
│  ┌─────────────┐ ┌──────────┐ ┌────────────┐│
│  │ RAG: top-3  │ │ History  │ │ Character  ││
│  │ lore chunks │ │ (5 turns)│ │ persona    ││
│  │ SentenceBERT│ │          │ │ template   ││
│  │ + ChromaDB  │ │          │ │ (Jinja2)   ││
│  └─────────────┘ └──────────┘ └────────────┘│
└──────────────┬───────────────────────────────┘
               │
    ▼
┌─────────────────────────────────────┐
│  Stage 3: Dialogue Generation       │
│  Fine-tuned Qwen 2.5-3B (LoRA)     │
│  + Constrained decoding             │
│  + Optional: Tree of Thoughts (3x)  │
└──────────────┬──────────────────────┘
               │
    ▼
NPC Response + Metadata
(intent, sentiment, lore_refs, latency_ms)
```

## Key Features

- **Character-consistent dialogue** — 3 distinct NPC personas with enforced personality, speech patterns, and knowledge boundaries
- **Lore-grounded responses** — RAG pipeline retrieves relevant world lore using SentenceBERT embeddings + ChromaDB
- **Fast intent classification** — Zero-shot DistilBERT classifier (7 categories, ~5ms) for deterministic game logic routing
- **LoRA fine-tuning** — Parameter-efficient training on Qwen 2.5-3B with 4-bit NF4 quantization (<1% trainable params)
- **Tree of Thoughts** — Research paper implementation generating 3 candidate responses and selecting the best
- **Production API** — FastAPI with async endpoints, SSE streaming, session management, Prometheus metrics
- **Full MLOps** — Docker multi-stage builds, CI/CD, automated evaluation (7 metrics), monitoring stack
- **Glass box demo** — Gradio app showing pipeline internals in real-time (intent, lore chunks, latency breakdown)

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Base LLM | Qwen 2.5-3B-Instruct | Apache 2.0, game-ready latency (<500ms), small enough for consumer GPU |
| Fine-tuning | PEFT/LoRA + TRL SFTTrainer | <1% trainable params, single GPU in hours |
| Quantization | bitsandbytes NF4 (train) | Fits 3B model in 16GB VRAM |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | 384-dim, <5ms per embedding |
| Vector DB | ChromaDB | Zero-config embedded or service mode |
| Intent | typeform/distilbert-base-uncased-mnli | Zero-shot, 5ms, deterministic |
| API | FastAPI | Async, OpenAPI docs, Pydantic v2 |
| Monitoring | Prometheus + Grafana | MLOps production observability |
| Containers | Docker multi-stage + compose | API + ChromaDB + monitoring stack |
| CI | GitHub Actions | Lint + test + Docker build |

## Project Structure

```
npc-dialogue-engine/
├── src/
│   ├── models/          # dialogue_model.py, intent_classifier.py
│   ├── rag/             # embeddings.py, retriever.py, lore_indexer.py
│   ├── pipeline/        # dialogue_pipeline.py, prompt_templates.py, context_manager.py
│   ├── training/        # dataset.py, train_lora.py, data_generation.py
│   ├── evaluation/      # metrics.py (7 metrics), human_eval_app.py (Gradio)
│   ├── api/             # app.py, routes.py, schemas.py, middleware.py
│   └── utils/           # config.py, logging_config.py
├── configs/
│   ├── characters/      # blacksmith.yaml, tavern_keeper.yaml, mysterious_sage.yaml
│   ├── model_config.yaml, rag_config.yaml, eval_config.yaml
├── data/
│   ├── lore/            # 5 world-building docs for RAG
│   ├── eval/            # Golden dialogues + adversarial inputs
│   └── processed/       # Training data (JSONL)
├── docker/              # Dockerfile, Dockerfile.gpu, docker-compose.yml
├── scripts/             # generate_training_data.py, index_lore.py, run_evaluation.py, export_model.py
├── tests/               # 142+ tests (config, characters, RAG, pipeline, API, training, evaluation)
└── .github/workflows/   # CI: lint + test + Docker build
```

## Quick Start

### Local Development (No GPU Required)

```bash
# Clone and install
git clone https://github.com/7ahir/npc-dialogue-engine.git
cd npc-dialogue-engine
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ml]"

# Run tests
make test  # or: pytest tests/ -v -m "not slow"

# Start API server (mock model)
DIALOGUE_MODEL_MODE=mock make serve
# → http://localhost:8000/api/v1/health

# Chat with an NPC
curl -X POST http://localhost:8000/api/v1/dialogue \
  -H 'Content-Type: application/json' \
  -d '{"player_message": "Got any swords?", "character_id": "blacksmith"}'

# Launch Gradio demo
DIALOGUE_MODEL_MODE=mock python src/evaluation/human_eval_app.py
# → http://localhost:7860
```

### Docker (Full Stack)

```bash
make docker-up  # or: docker compose -f docker/docker-compose.yml up -d
# API:        http://localhost:8000
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000 (admin/admin)
```

### Training (GPU Required)

After `pip install -e .`, the project exposes four CLI commands. Each is also runnable as a script (`python scripts/<name>.py …`).

```bash
# Generate synthetic training data
npc-generate-data --output data/processed/train.jsonl

# Index lore documents into ChromaDB
npc-index-lore

# Fine-tune with LoRA
pip install -e ".[ml,gpu,train]"
python src/training/train_lora.py --data-path data/processed/train.jsonl

# Merge adapter into base model for deployment
npc-export --adapter-path models/lora/final
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/dialogue` | Generate NPC response (sync) |
| `POST` | `/api/v1/dialogue/stream` | SSE streaming response |
| `GET` | `/api/v1/characters` | List available NPCs |
| `GET` | `/api/v1/characters/{id}` | Character details |
| `POST` | `/api/v1/sessions/{id}/reset` | Clear conversation history |
| `GET` | `/api/v1/health` | System health check |
| `GET` | `/api/v1/metrics` | Prometheus metrics |

## Evaluation Pipeline (7 Automated Metrics)

| Metric | Target | Method |
|--------|--------|--------|
| Character Consistency | >0.65 | Cosine similarity: response embedding vs persona embedding |
| Lore Accuracy | >0.80 | Semantic similarity vs retrieved lore chunks |
| Response Diversity | Self-BLEU <0.4 | 10 responses to same prompt |
| BERTScore F1 | >0.70 | vs golden reference responses |
| Latency p95 | <800ms | End-to-end timing distribution |
| Safety Rate | >95% | Adversarial input handling (15 test cases) |
| Grounding Rate | tracked | Response references RAG-retrieved info |

```bash
npc-eval --output results/eval_report.json
```

## Characters

| NPC | Role | Personality |
|-----|------|-------------|
| **Grenn Ironheart** | Dwarven Blacksmith | Gruff, honest, proud of craft |
| **Mira Hearthstone** | Tavern Keeper | Warm, perceptive, ex-adventurer |
| **Eldris the Veiled** | Mysterious Sage | Cryptic, ancient, all-knowing |

## Research Paper Implementation

**Tree of Thoughts** (Yao et al., 2023) — for complex dialogue scenarios:
1. Generate 3 candidate responses with different emotional tones
2. Score each against character consistency + lore accuracy
3. Select highest-scoring candidate
4. Only triggered for flagged scenarios (quest decisions, moral dilemmas)

Adds ~2x latency but measurably improves character consistency on complex prompts.

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Qwen 3B vs Mistral 7B | Latency-quality Pareto frontier. 3B fine-tuned on domain data matches 7B base on in-domain tasks |
| RAG vs fine-tuning on lore | Lore changes between game updates. RAG = update knowledge without retraining |
| Separate intent classifier | 5ms deterministic labels for game state machines vs waiting for LLM parse |
| ToT only for complex scenarios | Quality/latency trade-off — exactly like real game production |
| ChromaDB vs FAISS | Portfolio simplicity. At production scale: FAISS with IVF or managed vector service |
| MockDialogueModel | Enables full development and testing on CPU-only machines |

## Testing

```bash
make test           # Fast tests (142+ tests, ~35s)
make test-slow      # Include intent classifier tests (requires model download)
make lint           # Ruff linting + format check
make type-check     # MyPy type checking
```

## License

MIT
