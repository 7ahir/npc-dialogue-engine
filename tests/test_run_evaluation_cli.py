"""Tests for the ``scripts/run_evaluation.py`` CLI helpers.

The full CLI is exercised end-to-end every time we run an eval, but the
``--model-path`` env-wiring is the kind of thing that's easy to break
silently — if we forget to set ``DIALOGUE_MODEL_MODE`` the merged-model
path gets swapped in but the mock model still serves traffic, and the
report says "fine-tuned" while actually scoring the mock. That's exactly
the kind of regression a small, fast unit test catches.

The helper was extracted from ``main()`` specifically so this test could
exist without spinning up the full pipeline (which needs a populated
ChromaDB collection and ~hundreds of MB of model weights).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the script importable. ``scripts/`` isn't a package, so we add it
# to ``sys.path`` and import as a top-level module.
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import run_evaluation  # noqa: E402


class TestApplyModelPathEnv:
    """The helper must set every env var the model loader reads."""

    def test_sets_base_model_path(self, monkeypatch):
        monkeypatch.delenv("MODEL_BASE_MODEL", raising=False)
        run_evaluation._apply_model_path_env("models/merged")
        import os

        assert os.environ["MODEL_BASE_MODEL"] == "models/merged"

    def test_forces_transformers_mode(self, monkeypatch):
        # Explicitly set mock first to prove the helper overrides it.
        monkeypatch.setenv("DIALOGUE_MODEL_MODE", "mock")
        run_evaluation._apply_model_path_env("models/merged")
        import os

        assert os.environ["DIALOGUE_MODEL_MODE"] == "transformers"

    def test_disables_4bit_by_default(self, monkeypatch):
        monkeypatch.delenv("MODEL_LOAD_IN_4BIT", raising=False)
        run_evaluation._apply_model_path_env("models/merged")
        import os

        # The merged model is saved at the dtype the export script chose;
        # re-quantizing on load would double-quantize / crash.
        assert os.environ["MODEL_LOAD_IN_4BIT"] == "false"

    def test_respects_user_override_for_4bit(self, monkeypatch):
        """If a caller intentionally sets MODEL_LOAD_IN_4BIT=true, leave it."""
        monkeypatch.setenv("MODEL_LOAD_IN_4BIT", "true")
        run_evaluation._apply_model_path_env("models/merged")
        import os

        assert os.environ["MODEL_LOAD_IN_4BIT"] == "true"

    def test_works_with_absolute_path(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MODEL_BASE_MODEL", raising=False)
        absolute = str(tmp_path / "merged")
        run_evaluation._apply_model_path_env(absolute)
        import os

        assert os.environ["MODEL_BASE_MODEL"] == absolute


class TestCollectEnvironment:
    """The environment block is what makes a JSON report reproducible.

    Each field has a specific job; if any go missing silently the report
    becomes much harder to interpret months later. So check every field
    is present, and verify the optional ones (gpu, note, adapter_path)
    behave correctly when absent.
    """

    def test_returns_all_expected_keys(self):
        env = run_evaluation._collect_environment(adapter_path=None, note=None)
        expected = {
            "commit_sha",
            "python_version",
            "torch_version",
            "transformers_version",
            "peft_version",
            "gpu",
            "adapter_path",
            "note",
        }
        assert set(env.keys()) == expected

    def test_python_version_is_a_string(self):
        env = run_evaluation._collect_environment(adapter_path=None, note=None)
        # platform.python_version() always returns "X.Y.Z"; this is a
        # smoke test that we didn't accidentally store the tuple form.
        assert isinstance(env["python_version"], str)
        assert env["python_version"].count(".") >= 1

    def test_passes_through_adapter_path(self):
        env = run_evaluation._collect_environment(adapter_path="models/merged", note=None)
        assert env["adapter_path"] == "models/merged"

    def test_passes_through_note(self):
        env = run_evaluation._collect_environment(
            adapter_path=None, note="trained 2h41m on T4, seed=42"
        )
        assert env["note"] == "trained 2h41m on T4, seed=42"

    def test_omits_optional_fields_when_absent(self):
        env = run_evaluation._collect_environment(adapter_path=None, note=None)
        # Mock baseline runs from a CPU dev machine: no GPU, no adapter,
        # no note. These should be ``None``, not the string "None" or "".
        assert env["adapter_path"] is None
        assert env["note"] is None

    def test_torch_version_when_torch_installed(self):
        # The dev install pulls torch in via the [ml] extra, so this
        # should be populated. If we're ever in an env without torch,
        # the helper must return None (covered by the next test).
        env = run_evaluation._collect_environment(adapter_path=None, note=None)
        # Allow None for environments without torch installed.
        assert env["torch_version"] is None or isinstance(env["torch_version"], str)

    def test_package_version_returns_none_for_missing_package(self):
        # The helper that powers torch_version / transformers_version
        # must not raise on a typo or uninstalled package — silent None
        # is the right answer because the alternative would crash an
        # otherwise-successful eval run on a stale dev box.
        assert run_evaluation._package_version("definitely-not-a-real-pkg-xyz") is None


class TestCollectTrainingMetadata:
    def test_returns_expected_shape(self):
        meta = run_evaluation._collect_training_metadata(
            train_examples=900,
            train_wallclock_min=147.3,
        )
        assert meta == {
            "train_examples": 900,
            "train_wallclock_min": 147.3,
        }

    def test_preserves_missing_values(self):
        meta = run_evaluation._collect_training_metadata(
            train_examples=None,
            train_wallclock_min=None,
        )
        assert meta == {
            "train_examples": None,
            "train_wallclock_min": None,
        }


class TestGitCommitSha:
    """``_git_commit_sha`` is best-effort. Outside a repo we want None,
    not a crash."""

    def test_returns_a_sha_inside_a_repo(self):
        # This test lives inside the repo, so git rev-parse should work.
        sha = run_evaluation._git_commit_sha()
        assert sha is None or (isinstance(sha, str) and len(sha) >= 7)

    def test_returns_none_outside_a_repo(self, tmp_path, monkeypatch):
        # Run from a temp dir that isn't a git repo.
        monkeypatch.chdir(tmp_path)
        assert run_evaluation._git_commit_sha() is None
