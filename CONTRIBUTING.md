# Contributing

Thanks for your interest in NPC Dialogue Engine. This is primarily a portfolio project, but contributions, issues, and forks are welcome.

## Quick Start

```bash
# Clone and install (CPU-only, mock model — works without a GPU)
git clone https://github.com/7ahir/npc-dialogue-engine.git
cd npc-dialogue-engine
make install-all

# Copy env template
cp .env.example .env

# Run the fast test suite
make test-fast

# Lint
make lint
```

## Development Workflow

1. **Branch off `main`** — `git checkout -b your-feature-name`
2. **Write the test first** — every behavior change needs a test in `tests/`
3. **Use the mock model in dev** — `DIALOGUE_MODEL_MODE=mock` keeps things deterministic and CPU-only
4. **Run the gates locally** — `make test-fast && make lint` should pass before you push
5. **Open a PR** — describe what changed, why, and how it was tested

## Project Layout

```
src/
├── api/          # FastAPI routes, middleware, schemas
├── models/       # Dialogue model abstraction (mock + transformers)
├── pipeline/     # Orchestration: intent → RAG → generation
├── rag/          # ChromaDB retrieval + embeddings
├── evaluation/   # Automated metrics + Gradio glass-box demo
├── training/     # LoRA fine-tuning + synthetic data generation
└── utils/        # Config, logging
tests/            # Unit + integration tests
data/             # Lore docs, eval sets, training data
configs/          # Character YAMLs, prompts
scripts/          # CLI entry points
```

## Code Standards

- **Python 3.11+**, type hints required on public functions
- **Ruff** for linting and formatting (`E, F, I, N, W, UP, B, SIM`)
- **Pytest** for tests, `pytest-asyncio` for async paths
- **Structured logging** via `structlog` — never `print()` in `src/`
- **Pydantic v2** for all data models
- Line length: **100**

## Tests

Three tiers, declared via pytest markers:

| Tier | Command | What |
|------|---------|------|
| Fast | `make test-fast` | Unit tests, no external services. Run on every change. |
| Full | `make test` | Includes slow tests. |
| Integration | `pytest -m integration` | Requires ChromaDB, real models, etc. CI runs these on a schedule. |

CI gate: fast tests + lint must pass on every PR.

## Reporting Issues

- **Bugs** — include the command run, expected vs actual, and OS / Python version
- **Feature requests** — explain the use case before the proposed solution
- **Security** — please email rather than filing a public issue

## License

By contributing, you agree your contributions are licensed under the project's [MIT License](LICENSE).
