#!/usr/bin/env python3
"""Run the NPC dialogue evaluation suite.

Loads the checked-in golden + adversarial datasets from ``data/eval/`` and
runs each example through the full ``DialoguePipeline`` (intent + RAG +
generation), so the metrics that depend on retrieved lore (grounding,
lore_accuracy) actually see lore. Earlier versions of this CLI used
hardcoded prompts and skipped RAG, which made the grounding/lore numbers
vacuous — that's now fixed.

Usage:
    npc-eval                                          # full eval, default datasets
    npc-eval --character blacksmith                   # single character
    npc-eval --limit 5                                # smoke run (5 examples per char)
    npc-eval --output results/eval_report.json
    npc-eval --golden data/eval/custom_golden.jsonl   # override dataset path
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path so the script works whether invoked as
# `python scripts/run_evaluation.py` or via the `npc-eval` console script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import run_evaluation  # noqa: E402
from src.pipeline.dialogue_pipeline import DialoguePipeline  # noqa: E402
from src.pipeline.prompt_templates import PromptBuilder  # noqa: E402
from src.rag.embeddings import EmbeddingService  # noqa: E402
from src.utils.config import get_config  # noqa: E402
from src.utils.logging_config import get_logger, setup_logging  # noqa: E402

DEFAULT_GOLDEN = Path("data/eval/golden_dialogues.jsonl")
DEFAULT_ADVERSARIAL = Path("data/eval/adversarial_inputs.jsonl")


def _load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts. Skips blank lines."""
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{i} invalid JSONL: {exc}") from exc
    return rows


def _filter_examples(
    examples: list[dict],
    character: str | None,
    limit: int | None,
) -> list[dict]:
    if character:
        examples = [ex for ex in examples if ex.get("character") == character]
    if limit:
        # Limit *per character* so multi-character runs stay balanced
        per_char: dict[str, list[dict]] = {}
        for ex in examples:
            per_char.setdefault(ex.get("character", "_"), []).append(ex)
        examples = []
        for cid in per_char:
            examples.extend(per_char[cid][:limit])
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NPC dialogue evaluation")
    parser.add_argument(
        "--character",
        type=str,
        default=None,
        help="Evaluate only examples for this character (default: all)",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=DEFAULT_GOLDEN,
        help=f"Path to golden dialogues JSONL (default: {DEFAULT_GOLDEN})",
    )
    parser.add_argument(
        "--adversarial",
        type=Path,
        default=DEFAULT_ADVERSARIAL,
        help=f"Path to adversarial inputs JSONL (default: {DEFAULT_ADVERSARIAL})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/eval_report.json"),
        help="Output path for JSON report",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap examples per character (for smoke runs)",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Skip RAG retrieval (faster but disables grounding metrics)",
    )
    args = parser.parse_args()

    setup_logging()
    logger = get_logger(__name__)
    config = get_config()

    # ─── Load datasets ──────────────────────────────────────────
    golden = _filter_examples(_load_jsonl(args.golden), args.character, args.limit)
    adversarial = _load_jsonl(args.adversarial)

    if not golden:
        print(f"No golden examples matched (character={args.character!r})")
        sys.exit(1)

    print(f"Loaded {len(golden)} golden + {len(adversarial)} adversarial examples")

    # ─── Build pipeline ─────────────────────────────────────────
    # Share the embedding service across the pipeline (for ToT scoring) and
    # the metrics computation — avoids loading SentenceTransformer twice.
    embed_service = EmbeddingService()
    prompt_builder = PromptBuilder()

    if args.no_rag:
        # Skip retriever construction entirely (avoids needing a populated
        # ChromaDB collection on machines that haven't indexed yet).
        from src.models.dialogue_model import create_dialogue_model

        class _NullRetriever:
            embedding_service = embed_service

            def retrieve(self, query, **_):  # type: ignore[no-untyped-def]
                return []

            def format_context(self, _chunks):  # type: ignore[no-untyped-def]
                return ""

        pipeline = DialoguePipeline(
            model=create_dialogue_model(config.model),
            retriever=_NullRetriever(),  # type: ignore[arg-type]
            prompt_builder=prompt_builder,
            embedding_service=embed_service,
        )
    else:
        pipeline = DialoguePipeline(
            prompt_builder=prompt_builder,
            embedding_service=embed_service,
        )

    # ─── Run pipeline on golden examples ─────────────────────────
    responses: list[str] = []
    references: list[str] = []
    char_ids_list: list[str] = []
    latencies: list[float] = []
    lore_chunks_list: list[list[str]] = []

    for i, ex in enumerate(golden, 1):
        char_id = ex["character"]
        prompt = ex["player_message"]
        ref = ex.get("expected_response", "")

        start = time.perf_counter()
        # Use a unique session_id so history doesn't bleed across examples
        result = pipeline.process(
            player_message=prompt,
            character_id=char_id,
            session_id=f"eval-{i}",
        )
        latency_ms = (time.perf_counter() - start) * 1000

        # For lore_accuracy/grounding we need the actual chunk *texts*, not
        # just the source filenames. Re-run retrieval to grab them — cheap
        # since the embedding cache makes it ~5ms.
        try:
            chunks = pipeline.retriever.retrieve(prompt) if not args.no_rag else []
            chunk_texts = [c.text for c in chunks]
        except Exception:
            chunk_texts = []

        responses.append(result.npc_response)
        references.append(ref)
        char_ids_list.append(char_id)
        latencies.append(latency_ms)
        lore_chunks_list.append(chunk_texts)

        if i % 10 == 0:
            logger.info("eval_progress", processed=i, total=len(golden))

    # ─── Adversarial probe: stays-in-character via the same pipeline ─
    # Round-robin across characters (deterministic, sorted order) so the
    # safety check covers every NPC, not just whichever one was first
    # alphabetically. Previous version had the *comment* but not the
    # behavior — `sorted(…)[0]` always picked the same character.
    adversarial_chars = sorted({ex["character"] for ex in golden})
    _probe_idx = {"i": 0}

    def generate_fn(prompt: str) -> str:
        char_id = adversarial_chars[_probe_idx["i"] % len(adversarial_chars)]
        _probe_idx["i"] += 1
        result = pipeline.process(
            player_message=prompt,
            character_id=char_id,
            session_id=f"adversarial-{char_id}",
        )
        return result.npc_response

    adversarial_inputs = [ex["input"] for ex in adversarial]

    # ─── Run metrics ────────────────────────────────────────────
    report = run_evaluation(
        responses=responses,
        character_ids=char_ids_list,
        latencies_ms=latencies,
        lore_chunks=lore_chunks_list,
        references=references,
        adversarial_inputs=adversarial_inputs,
        generate_fn=generate_fn,
        embed_fn=embed_service.embed,
    )

    # ─── Output ─────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = report.summary()
    out["dataset"] = {
        "golden_path": str(args.golden),
        "adversarial_path": str(args.adversarial),
        "golden_count": len(golden),
        "adversarial_count": len(adversarial),
        "rag_enabled": not args.no_rag,
        "model_version": pipeline.model.model_version,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'=' * 60}")
    print("NPC Dialogue Evaluation Report")
    print(f"{'=' * 60}")
    print(f"Dataset:        {args.golden.name} ({len(golden)} examples)")
    print(f"Adversarial:    {args.adversarial.name} ({len(adversarial)} probes)")
    print(f"Model:          {pipeline.model.model_version}")
    print(f"RAG:            {'enabled' if not args.no_rag else 'disabled'}")
    print(f"Total examples: {report.total_examples}")
    print(f"Overall pass:   {'PASS' if report.overall_pass else 'FAIL'}")
    print("\nMetrics:")
    for m in report.metrics:
        status = "PASS" if m.passed else "FAIL"
        print(f"  [{status}] {m.name}: {m.score:.4f} (threshold: {m.threshold})")
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
