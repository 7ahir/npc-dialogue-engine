#!/usr/bin/env python3
"""Render fine-tuned evaluation artifacts from the mock + FT JSON reports.

This script solves the last mile after a Colab fine-tune run:

1. Load ``results/eval_report.json`` and ``results/eval_report_ft.json``.
2. Render a reviewer-friendly Markdown comparison table.
3. Optionally replace the README placeholder block between
   ``<!-- FT_RESULTS_START -->`` and ``<!-- FT_RESULTS_END -->``.

The point is to turn "copy a table out of a notebook" into a deterministic
artifact generation step. The FT report is the hard evidence; this script makes
that evidence legible in the portfolio without manual editing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_MOCK = Path("results/eval_report.json")
DEFAULT_FT = Path("results/eval_report_ft.json")
DEFAULT_OUTPUT = Path("results/eval_comparison.md")
DEFAULT_README = Path("README.md")

README_START = "<!-- FT_RESULTS_START -->"
README_END = "<!-- FT_RESULTS_END -->"

METRIC_META: dict[str, dict[str, str]] = {
    "character_consistency": {"label": "Character Consistency", "threshold": ">0.65"},
    "lore_accuracy": {"label": "Lore Accuracy", "threshold": ">0.80"},
    "bert_score_f1": {"label": "BERTScore F1", "threshold": ">0.70"},
    "response_diversity": {"label": "Response Diversity", "threshold": "<0.4"},
    "latency_p95": {"label": "Latency p95", "threshold": "<800ms"},
    "safety_rate": {"label": "Safety Rate", "threshold": ">95%"},
    "grounding_rate": {"label": "Grounding Rate", "threshold": "tracked"},
}


def _load_report(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _format_number(value: float) -> str:
    return f"{value:.4f}"


def _format_score(metric_name: str, score: float) -> str:
    if metric_name == "latency_p95":
        return f"{score:.1f}ms"
    if metric_name == "safety_rate":
        pct = score * 100
        return f"{pct:.1f}%".replace(".0%", "%")
    return _format_number(score)


def _format_delta(metric_name: str, mock_score: float, ft_score: float) -> str:
    delta = ft_score - mock_score
    if metric_name == "latency_p95":
        return f"{delta:+.1f}ms"
    if metric_name == "safety_rate":
        return f"{delta * 100:+.1f}pp"
    return f"{delta:+.4f}"


def _metric_label(metric_name: str) -> str:
    return METRIC_META.get(metric_name, {}).get("label", metric_name)


def _metric_threshold(metric_name: str, fallback: float) -> str:
    return METRIC_META.get(metric_name, {}).get("threshold", str(fallback))


def _build_table(mock_report: dict, ft_report: dict) -> str:
    lines = [
        "| Metric | Threshold | Mock | Fine-tuned | Δ |",
        "|---|---|---|---|---|",
    ]
    for metric_name, mock_metric in mock_report["metrics"].items():
        ft_metric = ft_report["metrics"].get(metric_name)
        if ft_metric is None:
            raise KeyError(f"fine-tuned report missing metric: {metric_name}")

        mock_status = "✅" if mock_metric["passed"] else "❌"
        ft_status = "✅" if ft_metric["passed"] else "❌"
        mock_display = f"{_format_score(metric_name, mock_metric['score'])} {mock_status}"
        ft_display = f"{_format_score(metric_name, ft_metric['score'])} {ft_status}"
        lines.append(
            "| "
            + " | ".join(
                [
                    _metric_label(metric_name),
                    _metric_threshold(metric_name, mock_metric["threshold"]),
                    mock_display,
                    ft_display,
                    _format_delta(metric_name, mock_metric["score"], ft_metric["score"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _build_metadata_lines(ft_report: dict) -> list[str]:
    env = ft_report.get("environment", {})
    training = ft_report.get("training", {})

    lines: list[str] = []
    commit_sha = env.get("commit_sha")
    if commit_sha:
        lines.append(f"- Commit: `{commit_sha[:12]}`")
    gpu = env.get("gpu")
    if gpu:
        lines.append(f"- GPU: `{gpu}`")
    adapter_path = env.get("adapter_path")
    if adapter_path:
        lines.append(f"- Model path: `{adapter_path}`")
    train_examples = training.get("train_examples")
    if train_examples is not None:
        lines.append(f"- Training examples: `{train_examples}`")
    train_wallclock = training.get("train_wallclock_min")
    if train_wallclock is not None:
        lines.append(f"- Training wallclock: `{train_wallclock:.1f} min`")
    note = env.get("note")
    if note:
        lines.append(f"- Note: `{note}`")

    return lines


def render_comparison_markdown(mock_report: dict, ft_report: dict) -> str:
    lines = [
        "# Fine-Tuned Evaluation Comparison",
        "",
        "Generated from `results/eval_report.json` and `results/eval_report_ft.json` "
        "using the same evaluation harness and datasets.",
        "",
    ]

    metadata_lines = _build_metadata_lines(ft_report)
    if metadata_lines:
        lines.extend(metadata_lines)
        lines.append("")

    lines.append(_build_table(mock_report, ft_report))
    lines.append("")
    return "\n".join(lines)


def render_readme_block(mock_report: dict, ft_report: dict) -> str:
    lines = [
        "Generated from [`results/eval_report.json`](results/eval_report.json) and "
        "[`results/eval_report_ft.json`](results/eval_report_ft.json) using the same "
        "harness and datasets.",
        "",
    ]
    metadata_lines = _build_metadata_lines(ft_report)
    if metadata_lines:
        lines.extend(metadata_lines)
        lines.append("")
    lines.append(_build_table(mock_report, ft_report))
    return "\n".join(lines)


def update_readme(readme_path: Path, block: str) -> None:
    text = readme_path.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        raise ValueError(
            f"{readme_path} is missing README markers {README_START!r} / {README_END!r}"
        )

    before, rest = text.split(README_START, 1)
    _old_block, after = rest.split(README_END, 1)
    replacement = f"{README_START}\n{block.rstrip()}\n{README_END}"
    readme_path.write_text(before + replacement + after, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render FT eval comparison artifacts")
    parser.add_argument("--mock", type=Path, default=DEFAULT_MOCK, help="Mock baseline JSON report")
    parser.add_argument("--ft", type=Path, default=DEFAULT_FT, help="Fine-tuned JSON report")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the Markdown comparison artifact",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=DEFAULT_README,
        help="README to update when --update-readme is passed",
    )
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="Replace the FT results placeholder block in the README",
    )
    args = parser.parse_args()

    mock_report = _load_report(args.mock)
    ft_report = _load_report(args.ft)

    markdown = render_comparison_markdown(mock_report, ft_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")

    if args.update_readme:
        update_readme(args.readme, render_readme_block(mock_report, ft_report))

    print(markdown)
    print(f"Comparison saved to: {args.output}")
    if args.update_readme:
        print(f"README updated:      {args.readme}")


if __name__ == "__main__":
    main()
