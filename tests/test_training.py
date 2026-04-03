"""Tests for training data generation and dataset loading."""

import json
import tempfile
from pathlib import Path

import pytest

from src.training.data_generation import (
    SCENARIOS,
    _generate_template_response,
    generate_training_data,
)
from src.training.dataset import (
    load_dialogue_dataset,
    create_train_val_split,
    _load_character_persona,
)


# ─── Scenario Definitions ────────────────────────────────────────


class TestScenarios:
    def test_scenario_count(self):
        assert len(SCENARIOS) == 10

    def test_scenario_structure(self):
        for scenario in SCENARIOS:
            assert "id" in scenario
            assert "description" in scenario
            assert "player_messages" in scenario
            assert len(scenario["player_messages"]) >= 3

    def test_unique_ids(self):
        ids = [s["id"] for s in SCENARIOS]
        assert len(ids) == len(set(ids)), "Duplicate scenario IDs found"

    def test_expected_scenarios(self):
        ids = {s["id"] for s in SCENARIOS}
        expected = {
            "greeting", "quest_request", "trade_inquiry", "lore_question",
            "threat_response", "personal_question", "farewell",
            "request_help", "rumor_gossip", "return_visit",
        }
        assert ids == expected


# ─── Template Response Generation ─────────────────────────────────


class TestTemplateResponse:
    @pytest.fixture
    def sample_character(self):
        return {
            "id": "test_npc",
            "name": "Test NPC",
            "role": "Test Role",
            "personality_traits": ["Gruff", "Honest"],
            "example_phrases": [
                "Fine steel, that is.",
                "I don't have time for nonsense.",
                "Bring me the materials and I'll craft it.",
            ],
        }

    def test_returns_string(self, sample_character):
        scenario = SCENARIOS[0]  # greeting
        response = _generate_template_response(sample_character, scenario)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_no_phrases_returns_ellipsis(self):
        char = {"id": "empty", "name": "Empty", "personality_traits": []}
        response = _generate_template_response(char, SCENARIOS[0])
        assert response == "..."

    def test_greeting_scenario(self, sample_character):
        greeting = next(s for s in SCENARIOS if s["id"] == "greeting")
        response = _generate_template_response(sample_character, greeting)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_farewell_scenario(self, sample_character):
        farewell = next(s for s in SCENARIOS if s["id"] == "farewell")
        response = _generate_template_response(sample_character, farewell)
        assert isinstance(response, str)

    def test_threat_scenario(self, sample_character):
        threat = next(s for s in SCENARIOS if s["id"] == "threat_response")
        response = _generate_template_response(sample_character, threat)
        assert isinstance(response, str)


# ─── Training Data Generation ─────────────────────────────────────


class TestGenerateTrainingData:
    def test_generates_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "train.jsonl"
            total = generate_training_data(output, num_examples_per_combo=1, seed=42)
            assert total > 0
            assert output.exists()

            # Verify JSONL format
            with open(output) as f:
                lines = f.readlines()
            assert len(lines) == total

            # Verify each line is valid JSON
            for line in lines:
                data = json.loads(line)
                assert "character" in data
                assert "scenario" in data
                assert "conversation" in data
                assert "metadata" in data

    def test_conversation_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "train.jsonl"
            generate_training_data(output, num_examples_per_combo=1, seed=42)

            with open(output) as f:
                data = json.loads(f.readline())

            conv = data["conversation"]
            assert len(conv) == 2
            assert conv[0]["role"] == "player"
            assert conv[1]["role"] == "npc"
            assert len(conv[0]["content"]) > 0
            assert len(conv[1]["content"]) > 0

    def test_metadata_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "train.jsonl"
            generate_training_data(output, num_examples_per_combo=1, seed=42)

            with open(output) as f:
                data = json.loads(f.readline())

            meta = data["metadata"]
            assert "tone" in meta
            assert "intent" in meta
            assert "generated" in meta
            assert meta["generated"] is True

    def test_reproducible_with_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out1 = Path(tmp) / "run1.jsonl"
            out2 = Path(tmp) / "run2.jsonl"

            generate_training_data(out1, num_examples_per_combo=2, seed=123)
            generate_training_data(out2, num_examples_per_combo=2, seed=123)

            assert out1.read_text() == out2.read_text()

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "sub" / "dir" / "train.jsonl"
            total = generate_training_data(output, num_examples_per_combo=1, seed=42)
            assert total > 0
            assert output.exists()


# ─── Dataset Loading ──────────────────────────────────────────────


class TestLoadDialogueDataset:
    @pytest.fixture
    def sample_jsonl(self, tmp_path):
        data = [
            {
                "character": "blacksmith",
                "scenario": "greeting",
                "conversation": [
                    {"role": "player", "content": "Hello!"},
                    {"role": "npc", "content": "Welcome to my forge."},
                ],
                "metadata": {"tone": "gruff", "intent": "greeting"},
            },
            {
                "character": "tavern_keeper",
                "scenario": "trade_inquiry",
                "conversation": [
                    {"role": "player", "content": "What's for sale?"},
                    {"role": "npc", "content": "Fresh ale and stew!"},
                ],
                "metadata": {"tone": "warm", "intent": "trade_inquiry"},
            },
        ]
        path = tmp_path / "test.jsonl"
        with open(path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")
        return path

    def test_loads_examples(self, sample_jsonl):
        examples = load_dialogue_dataset(sample_jsonl)
        assert len(examples) == 2

    def test_chat_format(self, sample_jsonl):
        examples = load_dialogue_dataset(sample_jsonl)
        messages = examples[0]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"

    def test_system_prompt_contains_character(self, sample_jsonl):
        examples = load_dialogue_dataset(sample_jsonl)
        system_msg = examples[0]["messages"][0]["content"]
        # Should contain character info (from persona loader)
        assert len(system_msg) > 0

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dialogue_dataset(tmp_path / "nonexistent.jsonl")

    def test_skips_invalid_json(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        with open(path, "w") as f:
            f.write('{"valid": true, "character": "blacksmith", "conversation": [{"role": "player", "content": "Hi"}, {"role": "npc", "content": "Hello"}]}\n')
            f.write("not json\n")
            f.write('{"character": "blacksmith", "conversation": [{"role": "player", "content": "Bye"}, {"role": "npc", "content": "Farewell"}]}\n')

        examples = load_dialogue_dataset(path)
        assert len(examples) == 2  # Skipped the invalid line

    def test_skips_short_conversations(self, tmp_path):
        path = tmp_path / "short.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({
                "character": "blacksmith",
                "conversation": [{"role": "player", "content": "Hi"}],
            }) + "\n")

        examples = load_dialogue_dataset(path)
        assert len(examples) == 0


# ─── Train/Val Split ──────────────────────────────────────────────


class TestTrainValSplit:
    def test_split_sizes(self):
        examples = [{"messages": []} for _ in range(100)]
        train, val = create_train_val_split(examples, val_fraction=0.1)
        assert len(val) == 10
        assert len(train) == 90

    def test_no_overlap(self):
        examples = [{"messages": [{"id": i}]} for i in range(50)]
        train, val = create_train_val_split(examples)
        train_ids = {e["messages"][0]["id"] for e in train}
        val_ids = {e["messages"][0]["id"] for e in val}
        assert train_ids.isdisjoint(val_ids)

    def test_reproducible(self):
        examples = [{"messages": [{"id": i}]} for i in range(50)]
        train1, val1 = create_train_val_split(examples, seed=42)
        train2, val2 = create_train_val_split(examples, seed=42)
        assert train1 == train2
        assert val1 == val2

    def test_minimum_val_size(self):
        examples = [{"messages": []} for _ in range(5)]
        train, val = create_train_val_split(examples, val_fraction=0.1)
        assert len(val) >= 1  # At least 1 val example
