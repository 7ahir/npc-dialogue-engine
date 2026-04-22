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

# Repo layout: <root>/src and <root>/deploy/huggingface-space
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.human_eval_app import create_demo  # noqa: E402

demo = create_demo()

if __name__ == "__main__":
    demo.launch()
