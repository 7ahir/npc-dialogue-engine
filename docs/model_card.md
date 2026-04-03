# Model Card: NPC Dialogue Engine

## Model Details

- **Model type:** Causal language model fine-tuned for NPC dialogue generation
- **Base model:** Qwen/Qwen2.5-3B-Instruct
- **Fine-tuning method:** LoRA (Low-Rank Adaptation) with PEFT
- **Quantization:** NF4 (4-bit) during training via bitsandbytes, optional GPTQ export for inference
- **Parameters:** ~3B total, ~2.4M trainable (<0.1%)
- **License:** Apache 2.0 (base model), MIT (fine-tuning code)
- **Language:** English

## Intended Use

Generate character-consistent, lore-grounded dialogue for RPG NPCs. The model is designed to:
- Stay in character across multi-turn conversations
- Reference game world lore when relevant (via RAG context)
- Handle adversarial inputs by staying in character
- Support 3+ distinct NPC personas (blacksmith, tavern keeper, sage)

**Not intended for:** General-purpose chat, factual Q&A, real-world advice.

## Training Data

- **Synthetic dialogues:** ~2,000-3,000 examples generated via template-based expansion
- **Format:** Multi-turn conversations across 10 scenarios × 3 characters
- **Scenarios:** greeting, quest_request, trade_inquiry, lore_question, threat_response, personal_question, farewell, request_help, rumor_gossip, return_visit
- **Augmentation:** Character persona injected as system prompt; lore context from RAG

## Training Procedure

- **LoRA config:** r=16, alpha=32, target=q_proj/k_proj/v_proj/o_proj, dropout=0.05
- **Quantization:** load_in_4bit=True, NF4, double quantization
- **Optimizer:** AdamW via HuggingFace Trainer
- **Learning rate:** 2e-4 with cosine scheduler, warmup ratio 0.03
- **Batch size:** 4 per device, gradient accumulation 4 (effective 16)
- **Epochs:** 3
- **Max sequence length:** 512 tokens
- **Hardware:** Single GPU (16GB VRAM minimum)

## Evaluation

| Metric | Target | Description |
|--------|--------|-------------|
| Character Consistency | >0.65 | Cosine similarity between response and persona embeddings |
| Lore Accuracy | >0.80 | Semantic similarity against retrieved lore chunks |
| Response Diversity | Self-BLEU <0.4 | Diversity across 10 responses to same prompt |
| BERTScore F1 | >0.70 | Against golden reference responses |
| Latency p95 | <800ms | End-to-end including retrieval |
| Safety Rate | >95% | Stays in character on adversarial inputs |
| Grounding Rate | tracked | References RAG-retrieved information |

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
