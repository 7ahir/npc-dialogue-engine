#!/usr/bin/env python3
"""LoRA fine-tuning script for NPC dialogue generation.

Fine-tunes a base model (Qwen 2.5-3B) using LoRA adapters on
character-specific dialogue data. Uses 4-bit NF4 quantization
during training to fit in consumer GPU VRAM (16GB).

Usage:
    python src/training/train_lora.py [--config configs/model_config.yaml]
    python src/training/train_lora.py --data-path data/processed/train.jsonl

Requires: pip install ".[ml,gpu,train]"
"""

import argparse
from pathlib import Path

import yaml


def train(
    config_path: Path = Path("configs/model_config.yaml"),
    data_path: Path = Path("data/processed/train.jsonl"),
) -> Path:
    """Run LoRA fine-tuning and return the path to saved adapter weights.

    This function is separated from main() so it can be called
    programmatically from notebooks or scripts.
    """
    # Import heavy deps only when training (not at module level)
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    from src.training.dataset import create_train_val_split, load_dialogue_dataset
    from src.utils.logging_config import get_logger, setup_logging

    setup_logging()
    logger = get_logger(__name__)

    # ─── Load config ────────────────────────────────────────────
    with open(config_path) as f:
        config = yaml.safe_load(f)

    base_model = config["base_model"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]
    quant_cfg = config["quantization"]
    output_dir = Path(train_cfg["output_dir"])

    logger.info(
        "training_config",
        base_model=base_model,
        lora_r=lora_cfg["r"],
        epochs=train_cfg["num_epochs"],
        data_path=str(data_path),
    )

    # ─── Load data ──────────────────────────────────────────────
    examples = load_dialogue_dataset(data_path)
    train_data, val_data = create_train_val_split(examples)

    logger.info("data_split", train=len(train_data), val=len(val_data))

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    # ─── Load tokenizer ────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ─── Quantization config ───────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
    )

    # ─── Load base model ───────────────────────────────────────
    logger.info("loading_base_model", model=base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False  # Required for gradient checkpointing

    # ─── LoRA config ───────────────────────────────────────────
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        "lora_applied",
        trainable_params=trainable,
        total_params=total,
        percent=round(trainable / total * 100, 2),
    )

    # ─── Training arguments ────────────────────────────────────
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        bf16=train_cfg["bf16"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=train_cfg["save_steps"],
        report_to="none",  # Set to "wandb" for experiment tracking
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        max_length=train_cfg["max_seq_length"],
        eos_token=tokenizer.eos_token,
    )

    # ─── Trainer ───────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    logger.info("training_started")
    trainer.train()

    # ─── Save adapter weights ──────────────────────────────────
    final_path = output_dir / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    logger.info("training_complete", adapter_path=str(final_path))
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune NPC dialogue model with LoRA")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model_config.yaml"),
        help="Path to model config YAML",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/train.jsonl"),
        help="Path to training data JSONL",
    )
    args = parser.parse_args()

    final_path = train(config_path=args.config, data_path=args.data_path)
    print(f"\nLoRA adapter saved to: {final_path}")


if __name__ == "__main__":
    main()
