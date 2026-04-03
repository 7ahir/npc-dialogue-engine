#!/usr/bin/env python3
"""Export LoRA adapter weights for deployment.

Merges the LoRA adapter into the base model and optionally exports
to GPTQ quantized format for production inference.

Usage:
    python scripts/export_model.py --adapter-path models/lora/final
    python scripts/export_model.py --adapter-path models/lora/final --quantize gptq
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def merge_adapter(adapter_path: Path, output_path: Path) -> None:
    """Merge LoRA adapter into base model and save."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.utils.logging_config import get_logger, setup_logging

    setup_logging()
    logger = get_logger(__name__)

    logger.info("loading_adapter", path=str(adapter_path))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))

    # Load base model from adapter config
    model = AutoModelForCausalLM.from_pretrained(
        str(adapter_path),
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Merge and unload
    logger.info("merging_adapter")
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()

    # Save merged model
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    logger.info("model_exported", output=str(output_path))
    print(f"\nMerged model saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LoRA model for deployment")
    parser.add_argument(
        "--adapter-path",
        type=Path,
        required=True,
        help="Path to LoRA adapter weights",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/exported"),
        help="Output path for merged model",
    )
    parser.add_argument(
        "--quantize",
        choices=["none", "gptq"],
        default="none",
        help="Quantization format for export",
    )
    args = parser.parse_args()

    if not args.adapter_path.exists():
        print(f"Adapter path not found: {args.adapter_path}")
        sys.exit(1)

    merge_adapter(args.adapter_path, args.output)

    if args.quantize == "gptq":
        print("\nGPTQ quantization requires auto-gptq. Install with: pip install auto-gptq")
        print("Then run: python -m auto_gptq.quantize --model_dir", args.output)


if __name__ == "__main__":
    main()
