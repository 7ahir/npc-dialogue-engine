"""HuggingFace Space entry point.

Boots the project's Gradio glass-box UI in mock-model mode so the Space runs
on the free CPU tier without downloading the LLM. The actual UI lives in
``src/evaluation/human_eval_app.py`` in the main repo — this file is just a
thin wrapper that sets the right env vars and delegates.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Mock mode keeps cold start fast and fits CPU-only Spaces
os.environ.setdefault("DIALOGUE_MODEL_MODE", "mock")

# ─── Workaround: gradio_client bool-schema bug ─────────────────────
# Pydantic v2 models with extra="forbid" emit `"additionalProperties": false`,
# which gradio_client's recursive schema walker mishandles (TypeError:
# argument of type 'bool' is not iterable). Patch the two functions to
# treat bool schemas as `Any` before importing the demo.
import gradio_client.utils as _gcu  # noqa: E402

_orig_get_type = _gcu.get_type
_orig_json_to_py = _gcu._json_schema_to_python_type


def _safe_get_type(schema):  # type: ignore[no-untyped-def]
    if isinstance(schema, bool):
        return "Any"
    return _orig_get_type(schema)


def _safe_json_to_py(schema, defs):  # type: ignore[no-untyped-def]
    if isinstance(schema, bool):
        return "Any"
    return _orig_json_to_py(schema, defs)


_gcu.get_type = _safe_get_type
_gcu._json_schema_to_python_type = _safe_json_to_py
# ───────────────────────────────────────────────────────────────────

# Repo layout: <root>/src and <root>/deploy/huggingface-space
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.human_eval_app import create_demo  # noqa: E402

demo = create_demo()

if __name__ == "__main__":
    # HF Spaces injects GRADIO_SERVER_NAME / _PORT — let gradio pick them up.
    demo.launch()
