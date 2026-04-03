"""Dialogue generation model abstraction.

Provides a clean interface for dialogue generation with two implementations:
- MockDialogueModel: character-consistent responses without loading any LLM.
  Uses example phrases from character configs. Ideal for local dev and testing.
- TransformersDialogueModel: real HF model with optional LoRA + 4-bit quantization.
  Requires GPU and significant VRAM.

A factory function picks the right implementation based on environment.
"""

import os
import random
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator

from src.pipeline.prompt_templates import CharacterLoader
from src.utils.config import ModelConfig, get_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class DialogueModel(ABC):
    """Abstract base for dialogue generation models."""

    @abstractmethod
    def generate(self, messages: list[dict[str, str]]) -> str:
        """Generate a complete NPC response from chat messages.

        Args:
            messages: List of {"role": "system"|"user", "content": "..."} dicts.
                      System message contains character persona + context.
                      User message contains the player's input.

        Returns:
            The NPC's response string.
        """
        ...

    @abstractmethod
    def generate_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Generate NPC response token-by-token for streaming.

        Yields individual tokens/words as they are generated.
        """
        ...

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Return a version identifier for the model."""
        ...


class MockDialogueModel(DialogueModel):
    """Character-consistent mock responses using example phrases.

    Extracts the character name from the system prompt, looks up their
    example phrases, and returns a contextually selected response.
    No LLM loading required — runs on any machine.
    """

    def __init__(self, character_loader: CharacterLoader | None = None) -> None:
        self._loader = character_loader or CharacterLoader()
        self._characters: dict[str, dict] = {}
        self._load_all_characters()

    def _load_all_characters(self) -> None:
        """Pre-load all character data for name matching."""
        for char_id in self._loader.list_characters():
            data = self._loader.load(char_id)
            self._characters[data["name"].lower()] = data

    def _extract_character_name(self, system_prompt: str) -> str | None:
        """Extract character name from 'You are {name}, {role}.' pattern."""
        match = re.match(r"You are (.+?),", system_prompt)
        return match.group(1) if match else None

    def _find_character(self, messages: list[dict[str, str]]) -> dict | None:
        """Find the character data from the system message."""
        for msg in messages:
            if msg["role"] == "system":
                name = self._extract_character_name(msg["content"])
                if name and name.lower() in self._characters:
                    return self._characters[name.lower()]
        return None

    def _select_response(self, character: dict, player_message: str) -> str:
        """Select a contextually appropriate example phrase.

        Simple heuristic: if the player message contains keywords that
        match a phrase, prefer that phrase. Otherwise pick randomly.
        """
        phrases = character.get("example_phrases", [])
        if not phrases:
            return "..."

        player_words = set(player_message.lower().split())

        # Score each phrase by keyword overlap with player message
        scored = []
        for phrase in phrases:
            phrase_words = set(phrase.lower().split())
            overlap = len(player_words & phrase_words)
            scored.append((overlap, phrase))

        # If any phrase has overlap, pick the best match
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored[0][0] > 0:
            return scored[0][1]

        # No keyword overlap — pick randomly
        return random.choice(phrases)

    def generate(self, messages: list[dict[str, str]]) -> str:
        character = self._find_character(messages)
        if not character:
            return "The NPC stares at you blankly."

        player_message = ""
        for msg in messages:
            if msg["role"] == "user":
                player_message = msg["content"]
                break

        return self._select_response(character, player_message)

    def generate_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Yield the response word-by-word, simulating streaming."""
        response = self.generate(messages)
        words = response.split()
        for i, word in enumerate(words):
            yield word if i == 0 else f" {word}"
            time.sleep(0.02)  # Simulate generation latency

    @property
    def model_version(self) -> str:
        return "mock-v1"


class TransformersDialogueModel(DialogueModel):
    """Real dialogue generation using a HuggingFace transformer model.

    Supports:
    - Base model loading (Qwen, Mistral, etc.)
    - LoRA adapter merging via PEFT
    - 4-bit NF4 quantization via bitsandbytes
    - Configurable generation parameters

    Requires GPU with sufficient VRAM (6GB+ for 3B model in 4-bit).
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self._config = config or get_config().model
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        """Lazy-load model and tokenizer."""
        # Import here to avoid import errors on machines without GPU
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("loading_dialogue_model", model=self._config.base_model)

        model_kwargs: dict = {}

        # 4-bit quantization
        if self._config.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError:
                logger.warning("bitsandbytes_not_available", msg="Skipping 4-bit quantization")

        # Device mapping
        if self._config.device == "auto":
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = self._config.device

        self._tokenizer = AutoTokenizer.from_pretrained(self._config.base_model)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._config.base_model,
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        )

        # Apply LoRA adapter if specified
        if self._config.lora_path and self._config.lora_path.exists():
            from peft import PeftModel

            logger.info("loading_lora_adapter", path=str(self._config.lora_path))
            self._model = PeftModel.from_pretrained(self._model, str(self._config.lora_path))

        logger.info("dialogue_model_loaded", model=self._config.base_model)

    @property
    def _loaded_model(self):  # type: ignore[no-untyped-def]
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def _loaded_tokenizer(self):  # type: ignore[no-untyped-def]
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    def generate(self, messages: list[dict[str, str]]) -> str:
        import torch

        tokenizer = self._loaded_tokenizer
        model = self._loaded_model

        # Apply chat template
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self._config.max_new_tokens,
                temperature=self._config.temperature,
                top_p=self._config.top_p,
                repetition_penalty=self._config.repetition_penalty,
                do_sample=True,
            )

        # Decode only the generated tokens (skip the input)
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    def generate_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        from threading import Thread

        from transformers import TextIteratorStreamer

        tokenizer = self._loaded_tokenizer
        model = self._loaded_model

        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        generate_kwargs = {
            **inputs,
            "max_new_tokens": self._config.max_new_tokens,
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
            "repetition_penalty": self._config.repetition_penalty,
            "do_sample": True,
            "streamer": streamer,
        }

        thread = Thread(target=model.generate, kwargs=generate_kwargs)
        thread.start()

        for token in streamer:
            if token:
                yield token

        thread.join()

    @property
    def model_version(self) -> str:
        version = self._config.base_model
        if self._config.lora_path:
            version += f"+lora:{self._config.lora_path.name}"
        if self._config.load_in_4bit:
            version += "+4bit"
        return version


def create_dialogue_model(
    config: ModelConfig | None = None,
    character_loader: CharacterLoader | None = None,
) -> DialogueModel:
    """Factory: create the appropriate dialogue model based on environment.

    Checks DIALOGUE_MODEL_MODE env var:
    - "mock": always use MockDialogueModel (no LLM loading)
    - "transformers": always use TransformersDialogueModel (requires GPU)
    - "auto" (default): use transformers if GPU available, else mock
    """
    mode = os.environ.get("DIALOGUE_MODEL_MODE", "auto").lower()

    if mode == "mock":
        logger.info("using_mock_model", reason="DIALOGUE_MODEL_MODE=mock")
        return MockDialogueModel(character_loader=character_loader)

    if mode == "transformers":
        logger.info("using_transformers_model", reason="DIALOGUE_MODEL_MODE=transformers")
        return TransformersDialogueModel(config=config)

    # Auto-detect
    try:
        import torch

        if torch.cuda.is_available():
            logger.info("using_transformers_model", reason="GPU detected")
            return TransformersDialogueModel(config=config)
    except ImportError:
        pass

    logger.info("using_mock_model", reason="No GPU available")
    return MockDialogueModel(character_loader=character_loader)
