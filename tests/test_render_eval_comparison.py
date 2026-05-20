"""Tests for the FT eval comparison renderer."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import render_eval_comparison  # noqa: E402


def _mock_report() -> dict:
    return {
        "metrics": {
            "character_consistency": {
                "score": 0.3858,
                "threshold": 0.65,
                "passed": False,
            },
            "latency_p95": {
                "score": 660.9319,
                "threshold": 800.0,
                "passed": True,
            },
            "safety_rate": {
                "score": 1.0,
                "threshold": 0.95,
                "passed": True,
            },
        }
    }


def _ft_report() -> dict:
    return {
        "metrics": {
            "character_consistency": {
                "score": 0.7123,
                "threshold": 0.65,
                "passed": True,
            },
            "latency_p95": {
                "score": 742.25,
                "threshold": 800.0,
                "passed": True,
            },
            "safety_rate": {
                "score": 0.98,
                "threshold": 0.95,
                "passed": True,
            },
        },
        "environment": {
            "commit_sha": "abcdef1234567890",
            "gpu": "Tesla T4",
            "adapter_path": "models/merged",
            "note": "seed=42",
        },
        "training": {
            "train_examples": 900,
            "train_wallclock_min": 147.3,
        },
    }


class TestRenderComparisonMarkdown:
    def test_renders_table_with_human_labels_and_units(self):
        markdown = render_eval_comparison.render_comparison_markdown(
            _mock_report(),
            _ft_report(),
        )
        assert "| Character Consistency | >0.65 | 0.3858 ❌ | 0.7123 ✅ | +0.3265 |" in markdown
        assert "| Latency p95 | <800ms | 660.9ms ✅ | 742.2ms ✅ | +81.3ms |" in markdown
        assert "| Safety Rate | >95% | 100% ✅ | 98% ✅ | -2.0pp |" in markdown

    def test_includes_ft_metadata_summary(self):
        markdown = render_eval_comparison.render_comparison_markdown(
            _mock_report(),
            _ft_report(),
        )
        assert "- Commit: `abcdef123456`" in markdown
        assert "- GPU: `Tesla T4`" in markdown
        assert "- Model path: `models/merged`" in markdown
        assert "- Training examples: `900`" in markdown
        assert "- Training wallclock: `147.3 min`" in markdown


class TestUpdateReadme:
    def test_replaces_marker_block_only(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text(
            "\n".join(
                [
                    "# Title",
                    "Intro",
                    render_eval_comparison.README_START,
                    "old block",
                    render_eval_comparison.README_END,
                    "Footer",
                ]
            ),
            encoding="utf-8",
        )

        render_eval_comparison.update_readme(readme, "new block\n| table |")

        text = readme.read_text(encoding="utf-8")
        assert "old block" not in text
        assert "new block" in text
        assert "Footer" in text

    def test_raises_if_markers_missing(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text("# Title only", encoding="utf-8")

        try:
            render_eval_comparison.update_readme(readme, "irrelevant")
        except ValueError as exc:
            assert "missing README markers" in str(exc)
        else:
            raise AssertionError("expected ValueError when markers are absent")
