# Model Card: NPC Dialogue Engine

> **Status:** Design + training-recipe card. The fine-tune itself has not been
> run — current eval numbers come from the `MockDialogueModel` baseline
> ([`results/eval_report.json`](../results/eval_report.json)). The card below
> documents the recipe that would be used when the GPU-backed run lands
> (tracked as a Tier-2 follow-up in [`case-study.md`](case-study.md)).

## Model Details

- **Model type:** Causal language model fine-tuned for NPC dialogue generation
- **Base model:** Qwen/Qwen2.5-3B-Instruct
- **Fine-tuning method:** LoRA (Low-Rank Adaptation) with PEFT
- **Quantization:** NF4 (4-bit) during training via bitsandbytes, optional GPTQ export for inference
- **Parameters:** ~3B total, ~2.4M trainable (<0.1%)
- **License:** Apache 2.0 (base model), MIT (fine-tuning code)
- **Language:** English

## Intended Use

Generate character-consistent, lore-grounded dialogue for RPG NPCs. Target behavior:
- Stay in character across turns within a session
- Reference game world lore when RAG injects relevant chunks
- Handle adversarial inputs without leaking the system prompt
- Support 3 NPC personas today (blacksmith, tavern keeper, sage); extensible via YAML configs

**Not intended for:** general-purpose chat, factual Q&A, real-world advice.

## Training Data

The checked-in generator (`src/training/data_generation.py`) emits **single-turn
player→NPC exchanges** — not multi-turn conversations. Default run produces:

- **3 characters × 10 scenarios × 4 examples = 120 exchanges**
- One `player` message + one `npc` response per example (not a conversation tree)
- Scenarios: greeting, quest_request, trade_inquiry, lore_question, threat_response,
  personal_question, farewell, request_help, rumor_gossip, return_visit
- Templated responses: the generator fills character example_phrases into
  scenario templates. This is a deliberately small, reproducible seed — not a
  training corpus.

**What a real fine-tune run would use on top of the seed:**

1. `--examples 30` for ~900 seed exchanges
2. Augmentation pass: paraphrase each with the base Qwen model to 3× (≈2,700 total)
3. Multi-turn chaining: stitch compatible scenarios into 2-4 turn conversations
4. Human review + filter pass before training

The 120-example seed is enough to smoke-test the pipeline; it is **not** enough
to move the character-consistency metric meaningfully. The `case-study.md`
Tier-2 section is where the real recipe gets run.

## Training Procedure (planned)

- **LoRA config:** r=16, alpha=32, target=q_proj/k_proj/v_proj/o_proj, dropout=0.05
- **Quantization:** load_in_4bit=True, NF4, double quantization
- **Optimizer:** AdamW via HuggingFace Trainer
- **Learning rate:** 2e-4 with cosine scheduler, warmup ratio 0.03
- **Batch size:** 4 per device, gradient accumulation 4 (effective 16)
- **Epochs:** 3
- **Max sequence length:** 512 tokens
- **Hardware:** Single GPU (16GB VRAM minimum)

## Evaluation

Current numbers are **mock-baseline** from `results/eval_report.json`
(20 golden examples, 15 adversarial probes, full pipeline with RAG):

| Metric | Target | Mock baseline | Description |
|---|---|---|---|
| Character Consistency | >0.65 | 0.39 ❌ | Cosine(response_embedding, persona_embedding) |
| Lore Accuracy | >0.80 | 0.32 ❌ | Cosine(response, retrieved_chunks) |
| Response Diversity | Self-BLEU <0.4 | 0.75 ❌ | Across 20 golden responses |
| BERTScore F1 | >0.70 | 0.25 ❌ | Against golden reference responses |
| Latency p95 | <800ms | 661ms ✅ | End-to-end; median 273ms, max 3.8s (first-request warmup) |
| Safety Rate | >95% | 100% ✅ | 15 adversarial probes; mock can't follow malicious instructions either |
| Grounding Rate | tracked | 0.00 | Mock ignores RAG context by design |

The four failing metrics are the ones a real fine-tune is expected to fix.
Latency + safety pass by pipeline architecture, not by model quality — see
[`failure-modes.md`](failure-modes.md) for what that distinction costs.

## Limitations

- Fine-tuned on a small, synthetic dataset — may not generalize to all dialogue scenarios
- Character consistency depends on quality of persona prompts
- No multilingual support (English only)
- Lore grounding depends on RAG retrieval quality; may hallucinate if lore DB is incomplete
- 3B parameter model has inherent quality ceiling vs larger models

## Ethical Considerations

- NPCs are fictional characters in a fantasy RPG setting
- Safety filters detect and prevent character-breaking on adversarial inputs
- No real personal data in training set
- Model outputs should be reviewed before use in shipped game content

## Environmental Impact

- Training: ~1-3 hours on single consumer GPU (estimated 0.5-1.5 kWh)
- Inference: ~200-800ms per response on GPU; mock mode for CPU development
- 4-bit quantization reduces memory footprint by ~75% vs full precision
