---
title: NPC Dialogue Engine
emoji: 🎮
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Glass-box NPC dialogue pipeline — intent → RAG → generation
---

# NPC Dialogue Engine — Live Demo

Interactive glass-box demo of the [NPC Dialogue Engine](https://github.com/7ahir/npc-dialogue-engine).

Chat with a dwarven blacksmith, tavern keeper, or mysterious sage and watch the three-stage ML pipeline run in real time:

1. **Intent classification** (DistilBERT zero-shot) — what is the player trying to do?
2. **RAG retrieval** (SentenceBERT + ChromaDB) — which lore chunks are relevant?
3. **Generation** (mocked here for CPU; LoRA-tuned Qwen 2.5-3B in the full repo) — the NPC responds in character.

The right-hand panel exposes per-stage latency, retrieved chunks, and intent confidence so you can see *why* the pipeline produced what it did.

This Space runs in **mock-model mode** (no GPU). The full architecture, training pipeline, evaluation suite, and FastAPI service live in the GitHub repo.
