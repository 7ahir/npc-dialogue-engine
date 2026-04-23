"""Smoke test for the LoRA merge script.

End-to-end verification that ``scripts/export_model.py`` actually produces
a loadable, generatable ``transformers`` model directory. Uses
``hf-internal-testing/tiny-random-LlamaForCausalLM`` (a few MB) as the
base so the test runs on CPU in a reasonable time.

Marked ``slow`` because it downloads model weights on first run; default
``make test`` skips it. CI runs the slow suite separately.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

TINY_BASE = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.mark.slow
def test_merge_adapter_roundtrip(tmp_path: Path) -> None:
    """Train a no-op adapter on the tiny base, merge it, reload, generate."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from scripts.export_model import merge_adapter

    # ── 1. Build a tiny adapter on top of the tiny base ────────────
    base = AutoModelForCausalLM.from_pretrained(TINY_BASE, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(TINY_BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base, lora_cfg)

    adapter_dir = tmp_path / "adapter"
    peft_model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # Sanity: PEFT only writes the adapter, not full base weights
    assert (adapter_dir / "adapter_config.json").exists()
    assert (adapter_dir / "adapter_model.safetensors").exists()

    # ── 2. Run the script under test ──────────────────────────────
    output_dir = tmp_path / "merged"
    merge_adapter(
        adapter_path=adapter_dir,
        output_path=output_dir,
        dtype="float32",
        device_map=None,  # pure CPU
    )

    # ── 3. Verify the merged directory is a real model ────────────
    assert (output_dir / "config.json").exists(), "merged dir missing config.json"
    weight_files = list(output_dir.glob("*.safetensors"))
    assert weight_files, "merged dir has no safetensors weights"
    assert (output_dir / "tokenizer_config.json").exists()

    # ── 4. Load + generate from the merged model ──────────────────
    merged = AutoModelForCausalLM.from_pretrained(str(output_dir), torch_dtype=torch.float32)
    merged_tok = AutoTokenizer.from_pretrained(str(output_dir))
    if merged_tok.pad_token is None:
        merged_tok.pad_token = merged_tok.eos_token

    inputs = merged_tok("Hello", return_tensors="pt")
    out = merged.generate(**inputs, max_new_tokens=4, do_sample=False)
    assert out.shape[1] > inputs["input_ids"].shape[1], "generate produced no new tokens"


@pytest.mark.slow
def test_merge_adapter_rejects_non_adapter_dir(tmp_path: Path) -> None:
    """A directory without adapter_config.json should raise a clear error."""
    from scripts.export_model import merge_adapter

    bogus = tmp_path / "not-an-adapter"
    bogus.mkdir()
    (bogus / "README.md").write_text("definitely not a peft adapter")

    with pytest.raises(FileNotFoundError, match="adapter_config.json"):
        merge_adapter(
            adapter_path=bogus,
            output_path=tmp_path / "out",
            dtype="float32",
            device_map=None,
        )
