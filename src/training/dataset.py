"""Training dataset for NPC dialogue fine-tuning.

Loads dialogue examples from JSONL files and formats them as chat-template
conversations for supervised fine-tuning with TRL's SFTTrainer.

Expected JSONL format:
{
  "character": "blacksmith",
  "scenario": "quest_item_request",
  "conversation": [
    {"role": "player", "content": "I need a sword..."},
    {"role": "npc", "content": "Bring me three Moonstone Ingots..."}
  ],
  "metadata": {"tone": "gruff_helpful", "intent": "quest_request", "lore_refs": [...]}
}
"""

import json
from pathlib import Path

import yaml

from src.utils.config import get_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _load_character_persona(character_id: str) -> str:
    """Load character persona description for system prompt."""
    config = get_config()
    path = config.characters_dir / f"{character_id}.yaml"
    if not path.exists():
        return f"You are an NPC named {character_id}."

    with open(path) as f:
        data = yaml.safe_load(f)

    return (
        f"You are {data['name']}, {data['role']}.\n\n"
        f"{data['description']}\n\n"
        f"Speak in character at all times. Keep responses concise."
    )


def load_dialogue_dataset(data_path: Path) -> list[dict]:
    """Load and format dialogue examples for SFT training.

    Converts raw JSONL into chat-template format:
    [
        {"role": "system", "content": "<persona>"},
        {"role": "user", "content": "<player message>"},
        {"role": "assistant", "content": "<npc response>"},
        ...
    ]

    Args:
        data_path: Path to a JSONL file with dialogue examples.

    Returns:
        List of dicts, each with a "messages" key containing the
        formatted conversation.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    examples = []
    persona_cache: dict[str, str] = {}

    with open(data_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("skipping_invalid_line", line=line_num, error=str(e))
                continue

            character_id = entry.get("character", "unknown")
            conversation = entry.get("conversation", [])

            if len(conversation) < 2:
                logger.warning("skipping_short_conversation", line=line_num)
                continue

            # Cache persona lookups
            if character_id not in persona_cache:
                persona_cache[character_id] = _load_character_persona(character_id)

            # Build chat messages
            messages: list[dict[str, str]] = [
                {"role": "system", "content": persona_cache[character_id]}
            ]

            for turn in conversation:
                role = turn["role"]
                content = turn["content"]
                # Map "player" -> "user", "npc" -> "assistant"
                chat_role = "user" if role == "player" else "assistant"
                messages.append({"role": chat_role, "content": content})

            examples.append({"messages": messages})

    logger.info("dataset_loaded", path=str(data_path), examples=len(examples))
    return examples


def create_train_val_split(
    examples: list[dict],
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split examples into train and validation sets.

    Uses a deterministic shuffle for reproducibility.
    """
    import random

    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)

    val_size = max(1, int(len(shuffled) * val_fraction))
    return shuffled[val_size:], shuffled[:val_size]
