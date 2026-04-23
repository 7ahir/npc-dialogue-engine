# Evaluation Reports

Two reports live (or will live) here. They share the same evaluation harness (`scripts/run_evaluation.py`) and the same datasets (`data/eval/golden_dialogues.jsonl` + `data/eval/adversarial_inputs.jsonl`) — only the model under test differs.

| File | Model | Status | How produced |
|---|---|---|---|
| `eval_report.json` | `MockDialogueModel` (mock-v1) | ✅ committed | `npc-eval --output results/eval_report.json` |
| `eval_report_ft.json` | Qwen 2.5-3B + LoRA (merged) | ⏳ pending | [`notebooks/colab_finetune.ipynb`](../notebooks/colab_finetune.ipynb), then `npc-eval --model-path models/merged --output results/eval_report_ft.json` |

The mock report is the **lower bound**: every quality metric fails because the mock model is a deterministic phrase-picker, not an instruction-follower. It exists to prove the pipeline is correctly wired (intent + RAG + generation + 7 metrics all run end-to-end without a GPU) and to set the bar the fine-tune has to clear.

The fine-tuned report is the **real-model evidence**. Once `colab_finetune.ipynb` produces it, drop it next to this README and update the README's "Fine-tuned model results" table from cell 9's markdown output.

## Why two reports, not one

A single overwriting "current" report would lose the baseline. Keeping both side by side makes the before/after delta a `git diff` away, and lets reviewers verify the harness didn't change between runs (same datasets, same metric thresholds, same code path — only the generator swapped).

## Reproducing

```bash
# Mock baseline (no GPU, ~30s)
DIALOGUE_MODEL_MODE=mock npc-eval --output results/eval_report.json

# Fine-tuned (T4 GPU, after running the Colab notebook)
npc-eval --model-path models/merged --output results/eval_report_ft.json
```

The `--model-path` flag flips the pipeline into transformers mode and points it at a local merged-model directory (the output of `npc-export`).
