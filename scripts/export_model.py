#!/usr/bin/env python3
"""Export a LoRA adapter as a merged, deployment-ready model.

The previous version of this script called
``AutoModelForCausalLM.from_pretrained(adapter_path)``, which only works
on a *full* model directory. PEFT adapter directories contain just the
adapter deltas (``adapter_config.json`` + ``adapter_model.safetensors``)
and the base model name — they cannot be loaded as a standalone model.

The right pattern is to use PEFT's ``AutoPeftModelForCausalLM``: it reads
``adapter_config.json``, downloads the base model by name (or finds it in
the local HF cache), wraps it with the adapter, and exposes
``merge_and_unload()`` to fold the adapter weights into the base. The
result is a plain ``transformers`` model directory that any HF-compatible
runtime (TGI, vLLM, llama.cpp converters, GPTQ quantizers) can consume.

Usage:
    python scripts/export_model.py --adapter-path models/lora/final
    python scripts/export_model.py --adapter-path models/lora/final \\
        --output models/exported --dtype bfloat16
    python scripts/export_model.py --adapter-path models/lora/final --quantize gptq
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _resolve_dtype(dtype_str: str):  # type: ignore[no-untyped-def]
    """Map --dtype CLI string to a torch dtype."""
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_str]


def merge_adapter(
    adapter_path: Path,
    output_path: Path,
    dtype: str = "float16",
    device_map: str | None = "auto",
) -> Path:
    """Merge a LoRA adapter into its base model and save the result.

    Args:
        adapter_path: Directory containing ``adapter_config.json`` and the
            adapter weights (typically the output of a PEFT training run).
        output_path: Where to write the merged ``transformers`` model.
        dtype: Torch dtype for loading + saving (``float16``, ``bfloat16``,
            ``float32``). Defaults to ``float16`` to halve disk size.
        device_map: Passed to ``from_pretrained``. Use ``None`` for pure
            CPU (the smoke test does this with a tiny model).

    Returns:
        The output path.
    """
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    from src.utils.logging_config import get_logger, setup_logging

    setup_logging()
    logger = get_logger(__name__)

    adapter_path = adapter_path.resolve()
    output_path = output_path.resolve()

    # Sanity-check: the adapter_config.json must exist, otherwise PEFT
    # falls back to confusing error messages downstream.
    adapter_config_path = adapter_path / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(
            f"No adapter_config.json at {adapter_config_path}. "
            f"Is {adapter_path} actually a PEFT adapter directory?"
        )

    with open(adapter_config_path) as f:
        adapter_cfg = json.load(f)
    base_model = adapter_cfg.get("base_model_name_or_path", "<unknown>")
    logger.info(
        "loading_adapter",
        adapter_path=str(adapter_path),
        base_model=base_model,
    )

    torch_dtype = _resolve_dtype(dtype)

    load_kwargs: dict = {"torch_dtype": torch_dtype}
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    # AutoPeftModelForCausalLM = "load the base model named in
    # adapter_config.json AND apply the adapter on top". This is the
    # one-liner that the previous AutoModelForCausalLM call was missing.
    model = AutoPeftModelForCausalLM.from_pretrained(str(adapter_path), **load_kwargs)

    logger.info("merging_adapter")
    merged = model.merge_and_unload()

    # Tokenizer: prefer the adapter dir (training may have added special
    # tokens), fall back to the base model.
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    except Exception:
        logger.info("tokenizer_fallback_to_base", base_model=base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model)

    output_path.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_path), safe_serialization=True)
    tokenizer.save_pretrained(str(output_path))

    logger.info(
        "model_exported",
        output=str(output_path),
        base_model=base_model,
        dtype=dtype,
    )
    print(f"\nMerged model saved to: {output_path}")
    print(f"  Base model: {base_model}")
    print(f"  Dtype:      {dtype}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a LoRA model for deployment")
    parser.add_argument(
        "--adapter-path",
        type=Path,
        required=True,
        help="Path to PEFT adapter directory (must contain adapter_config.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/exported"),
        help="Output path for merged model (default: models/exported)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Save dtype (default: float16)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU loading (skip device_map='auto'). Small/test models only.",
    )
    parser.add_argument(
        "--quantize",
        choices=["none", "gptq"],
        default="none",
        help="Optional post-merge quantization (gptq requires auto-gptq)",
    )
    args = parser.parse_args()

    if not args.adapter_path.exists():
        print(f"Adapter path not found: {args.adapter_path}")
        sys.exit(1)

    merge_adapter(
        adapter_path=args.adapter_path,
        output_path=args.output,
        dtype=args.dtype,
        device_map=None if args.cpu else "auto",
    )

    if args.quantize == "gptq":
        print("\nGPTQ quantization requires auto-gptq. Install with: pip install auto-gptq")
        print(f"Then run: python -m auto_gptq.quantize --model_dir {args.output}")


if __name__ == "__main__":
    main()
