"""Jinja2 prompt templates for NPC dialogue generation.

Constructs structured prompts that combine character persona, lore context,
conversation history, and player input. The template architecture separates
concerns cleanly:
  - Character persona → system prompt (who the NPC is)
  - Lore context → system prompt (what the NPC knows)
  - Conversation history → context (what has been said)
  - Player input → user message (what to respond to)
"""

from pathlib import Path

import yaml
from jinja2 import BaseLoader, Environment

from src.utils.config import get_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# ─── System Prompt Template ────────────────────────────────────────
# Injected as the system message. Defines character identity and available knowledge.
SYSTEM_TEMPLATE = """\
You are {{ name }}, {{ role }}.

{{ description }}

## Personality
{% for trait in personality_traits %}- {{ trait }}
{% endfor %}

## How You Speak
{% for pattern in speech_patterns %}- {{ pattern }}
{% endfor %}

## What You Know
{% for boundary in knowledge_boundaries %}- {{ boundary }}
{% endfor %}

## Emotional State
Respond with emotion appropriate to the player's tone and the situation.
{% for emotion, behavior in emotional_range.items() %}- When {{ emotion }}: {{ behavior }}
{% endfor %}

{% if lore_context %}
## Relevant World Knowledge
{{ lore_context }}
{% endif %}

{% if conversation_history %}
## Recent Conversation
{{ conversation_history }}
{% endif %}

## Rules
- Stay in character at all times. Never break the fourth wall.
- Keep responses concise (2-4 sentences for casual exchanges, longer for lore/quests).
- If asked about something outside your knowledge, redirect to the appropriate person.
- React to the player's tone — be warmer to friendly players, guarded with hostile ones.
- Never invent facts that contradict the world knowledge above.
- Do not use modern slang, references, or out-of-world concepts.\
"""

# ─── Tree of Thoughts Candidate Template ───────────────────────────
# Used for complex dialogue scenarios where we generate multiple candidates
# and select the best one based on character consistency and lore accuracy.
TOT_CANDIDATE_TEMPLATE = """\
Generate {{ num_candidates }} different response options for the NPC, \
each with a distinct emotional tone:

{% for tone in tones %}\
Option {{ loop.index }} ({{ tone }}):
{% endfor %}

For each option, stay fully in character as {{ name }} and ensure \
consistency with the world knowledge provided. Each option should be \
a complete, standalone response.\
"""


class CharacterLoader:
    """Loads and caches character persona data from YAML configs."""

    def __init__(self, characters_dir: Path | None = None) -> None:
        config = get_config()
        self._dir = characters_dir or config.characters_dir
        self._cache: dict[str, dict] = {}

    def load(self, character_id: str) -> dict:
        """Load a character config by ID."""
        if character_id in self._cache:
            return self._cache[character_id]

        path = self._dir / f"{character_id}.yaml"
        if not path.exists():
            raise ValueError(
                f"Character '{character_id}' not found at {path}. "
                f"Available: {[p.stem for p in self._dir.glob('*.yaml')]}"
            )

        with open(path) as f:
            data = yaml.safe_load(f)

        self._cache[character_id] = data
        return data

    def list_characters(self) -> list[str]:
        """Return all available character IDs."""
        return sorted(p.stem for p in self._dir.glob("*.yaml"))


class PromptBuilder:
    """Builds structured prompts for the dialogue generation model."""

    def __init__(self, character_loader: CharacterLoader | None = None) -> None:
        self.character_loader = character_loader or CharacterLoader()
        self._env = Environment(loader=BaseLoader())
        self._system_template = self._env.from_string(SYSTEM_TEMPLATE)
        self._tot_template = self._env.from_string(TOT_CANDIDATE_TEMPLATE)

    def build_system_prompt(
        self,
        character_id: str,
        lore_context: str = "",
        conversation_history: str = "",
    ) -> str:
        """Build the system prompt with character persona and context."""
        character = self.character_loader.load(character_id)

        # Convert emotional_range from YAML list-of-dicts to flat dict
        emotional_range = character.get("emotional_range", {})
        if isinstance(emotional_range, list):
            parsed = {}
            for item in emotional_range:
                if isinstance(item, dict):
                    for key, val in item.items():
                        parsed[key.lower()] = val
                elif isinstance(item, str) and ":" in item:
                    key, val = item.split(":", 1)
                    parsed[key.strip().lower()] = val.strip()
            emotional_range = parsed

        return self._system_template.render(
            name=character["name"],
            role=character["role"],
            description=character["description"],
            personality_traits=character["personality_traits"],
            speech_patterns=character["speech_patterns"],
            knowledge_boundaries=character["knowledge_boundaries"],
            emotional_range=emotional_range,
            lore_context=lore_context,
            conversation_history=conversation_history,
        )

    def build_chat_messages(
        self,
        character_id: str,
        player_message: str,
        lore_context: str = "",
        conversation_history: str = "",
    ) -> list[dict[str, str]]:
        """Build chat messages in the format expected by transformers.

        Returns a list of {"role": ..., "content": ...} dicts suitable
        for the model's chat template.
        """
        system_prompt = self.build_system_prompt(
            character_id=character_id,
            lore_context=lore_context,
            conversation_history=conversation_history,
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": player_message},
        ]

    def build_tot_prompt(
        self,
        character_id: str,
        num_candidates: int = 3,
        tones: list[str] | None = None,
    ) -> str:
        """Build Tree of Thoughts candidate generation prompt."""
        character = self.character_loader.load(character_id)
        tones = tones or ["helpful and warm", "cautious and guarded", "mysterious and cryptic"]

        return self._tot_template.render(
            name=character["name"],
            num_candidates=num_candidates,
            tones=tones,
        )
