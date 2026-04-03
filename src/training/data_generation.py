"""Synthetic training data generation for NPC dialogue.

Generates multi-turn dialogue examples by defining character × scenario
combinations and producing structured JSONL output. Designed to work
with any LLM API (Claude, GPT-4, or local models) via a simple
generation function interface.

When no LLM API is available, falls back to template-based generation
using character example phrases and scenario templates.
"""

import json
import random
from pathlib import Path

import yaml

from src.utils.config import get_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# ─── Scenario Definitions ──────────────────────────────────────

SCENARIOS = [
    {
        "id": "greeting",
        "description": "Player greets the NPC for the first time",
        "player_messages": [
            "Hello there!",
            "Hey, are you open?",
            "Good day to you.",
            "Excuse me, can I talk to you?",
        ],
    },
    {
        "id": "quest_request",
        "description": "Player asks about available quests or tasks",
        "player_messages": [
            "Do you have any work for me?",
            "I'm looking for a quest. Anything you need done?",
            "I'm an adventurer. Got any tasks?",
            "Know of anything that needs doing around here?",
        ],
    },
    {
        "id": "trade_inquiry",
        "description": "Player asks about buying or selling items",
        "player_messages": [
            "What do you have for sale?",
            "I need to buy some supplies.",
            "How much for your best item?",
            "Can I see your wares?",
        ],
    },
    {
        "id": "lore_question",
        "description": "Player asks about world history or local knowledge",
        "player_messages": [
            "What can you tell me about this place?",
            "What happened here long ago?",
            "Tell me about the history of this region.",
            "What do you know about the old legends?",
        ],
    },
    {
        "id": "threat_response",
        "description": "Player threatens or intimidates the NPC",
        "player_messages": [
            "Give me what I want or else.",
            "I could burn this place down, you know.",
            "You're going to regret making me angry.",
            "Don't test me. I'm dangerous.",
        ],
    },
    {
        "id": "personal_question",
        "description": "Player asks about the NPC's personal life or background",
        "player_messages": [
            "How did you end up here?",
            "What's your story?",
            "Do you have a family?",
            "How long have you been doing this?",
        ],
    },
    {
        "id": "farewell",
        "description": "Player says goodbye",
        "player_messages": [
            "I should get going. Farewell.",
            "Thanks for your help. See you around.",
            "I'll be back later. Goodbye.",
            "Take care of yourself.",
        ],
    },
    {
        "id": "request_help",
        "description": "Player asks for specific help or information",
        "player_messages": [
            "I need your expertise with something.",
            "Can you help me with a problem?",
            "I've got something I need identified.",
            "There's something strange going on. What do you think?",
        ],
    },
    {
        "id": "rumor_gossip",
        "description": "Player asks about rumors or recent events",
        "player_messages": [
            "Heard any rumors lately?",
            "What's the word around town?",
            "Anything strange happening recently?",
            "People seem worried. What's going on?",
        ],
    },
    {
        "id": "return_visit",
        "description": "Player returns after a previous conversation",
        "player_messages": [
            "I'm back. Did you find what I asked about?",
            "Remember me? I was here earlier.",
            "I've done what you asked. Here's the result.",
            "Any updates since we last spoke?",
        ],
    },
]


def _load_all_characters() -> dict[str, dict]:
    """Load all character configs."""
    config = get_config()
    characters = {}
    for path in config.characters_dir.glob("*.yaml"):
        with open(path) as f:
            data = yaml.safe_load(f)
            characters[data["id"]] = data
    return characters


def _generate_template_response(character: dict, scenario: dict) -> str:
    """Generate a plausible NPC response using templates and example phrases.

    Simple heuristic — picks relevant example phrases and adapts them
    to the scenario. For production training data, replace this with
    an LLM API call.
    """
    phrases = character.get("example_phrases", [])
    if not phrases:
        return "..."

    # Pick a phrase, optionally with scenario context
    phrase = random.choice(phrases)

    # For some scenarios, add contextual flavor
    scenario_id = scenario["id"]
    name = character["name"]

    if scenario_id == "greeting":
        greetings = [
            f"*nods* {phrase}",
            phrase,
            f"Welcome. {phrase}",
        ]
        return random.choice(greetings)
    elif scenario_id == "farewell":
        farewells = [
            f"Safe travels. {phrase}",
            f"Until next time. {phrase}",
            phrase,
        ]
        return random.choice(farewells)
    elif scenario_id == "threat_response":
        # Characters should respond to threats in character
        responses = [
            f"*{character['personality_traits'][0].lower()}* {phrase}",
            phrase,
        ]
        return random.choice(responses)

    return phrase


def generate_training_data(
    output_path: Path,
    num_examples_per_combo: int = 4,
    seed: int = 42,
) -> int:
    """Generate synthetic training data as JSONL.

    Creates dialogue examples for every character × scenario combination.

    Args:
        output_path: Where to write the JSONL file.
        num_examples_per_combo: How many examples per character×scenario pair.
        seed: Random seed for reproducibility.

    Returns:
        Total number of examples generated.
    """
    random.seed(seed)
    characters = _load_all_characters()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(output_path, "w") as f:
        for char_id, char_data in characters.items():
            for scenario in SCENARIOS:
                for i in range(num_examples_per_combo):
                    player_msg = random.choice(scenario["player_messages"])
                    npc_response = _generate_template_response(char_data, scenario)

                    example = {
                        "character": char_id,
                        "scenario": scenario["id"],
                        "conversation": [
                            {"role": "player", "content": player_msg},
                            {"role": "npc", "content": npc_response},
                        ],
                        "metadata": {
                            "tone": char_data["personality_traits"][0].lower()
                            if char_data.get("personality_traits")
                            else "neutral",
                            "intent": scenario["id"],
                            "generated": True,
                        },
                    }
                    f.write(json.dumps(example) + "\n")
                    total += 1

    logger.info(
        "training_data_generated",
        output_path=str(output_path),
        total_examples=total,
        characters=len(characters),
        scenarios=len(SCENARIOS),
    )
    return total


# ─── Script Entry Point ────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic NPC training data")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/train.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--examples-per-combo",
        type=int,
        default=4,
        help="Examples per character×scenario pair",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from src.utils.logging_config import setup_logging

    setup_logging()
    total = generate_training_data(args.output, args.examples_per_combo, args.seed)
    print(f"\nGenerated {total} training examples → {args.output}")
