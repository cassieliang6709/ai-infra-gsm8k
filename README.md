# GSM8K post-training study · Qwen2.5-0.5B

SFT variants, CoT distillation and GRPO on one RTX 4090, with every number traceable to a raw
evaluation file in `results/`.

**Findings**

1. **Plain LoRA SFT teaches the model to never stop.** 525 of its 851 misses run past 2,000 tokens.
   The base model's `<|im_end|>` row was never trained (norm 0.301 vs. a 0.461 median, identical to
   `<|im_start|>`), and LoRA on the linear layers cannot touch it: after a finished answer the model
   gives `<|im_end|>` probability 0.0003. Making `embed_tokens` / `lm_head` trainable raises that to
   0.9998 and cuts runaway misses to 34.
2. **CoT distillation is the largest single gain.** SFT on 4,426 answer-verified Qwen2.5-7B-Instruct
   traces lifts 5-shot accuracy from 34.0% to 47.9%, and arithmetic errors fall from 21% to 8% of misses.
3. **GRPO lifts zero-shot accuracy from 34.7% to 52.2%** in 116 steps and nearly eliminates runaway outputs
   (1 of 1,319). Adding a format bonus to the reward (51.1%) did not help.
4. **Math-only training leaves general knowledge roughly unchanged**: MMLU stays within 0.6 pt of
   the base model; C-Eval moves by up to −1.8 pt after GRPO.

## Results

### GSM8K test (1,319), 5-shot, chat template, greedy — `lm-evaluation-harness` + vLLM

| Model | strict-match | flexible-extract |
|---|---|---|
| Base, raw-text prompt | 34.0% | 34.9% |
| Base, chat template | 29.3% | 29.6% |
| SFT: LoRA | 35.5% | 27.9% |
| SFT: full fine-tune | 33.1% | 33.1% |
| SFT: LoRA + trainable embed/lm_head | 34.0% | 34.0% |
| SFT: CoT distillation (7B teacher) | 47.9% | 47.9% |
| GRPO, accuracy reward (step 80) | 45.6% | 45.6% |
| GRPO, accuracy + format reward (step 116) | 44.9% | 44.9% |

Plain LoRA's strict/flexible gap is the runaway problem: strict-match takes the first `#### n`, while
flexible-extract takes the last number in the output, which is text generated after the answer. Regenerate the table with `python analysis/summarize.py`.

### GSM8K test (1,319), zero-shot, greedy — verl validation during GRPO

Both runs start from the LoRA + embed/lm_head SFT model and share every hyperparameter except the reward.

| Step | 0 | 20 | 40 | 60 | 80 | 100 | 116 |
|---|---|---|---|---|---|---|---|
| Accuracy reward | 34.7% | 40.3% | 45.3% | 47.5% | 48.7% | 49.2% | **52.2%** |
| Accuracy + format reward | 34.5% | 38.1% | 42.7% | 46.9% | 47.9% | 50.3% | **51.1%** |

Source: `results/logs/grpo_*_val_curve.log`. The accuracy-reward run was resumed from step 80,
so its log has two step-80 entries (48.7% before, 48.8% after the restart).

### Where the misses come from (5-shot runs above)

| model | wrong | arithmetic | reasoning | format | runaway |
|---|---|---|---|---|---|
| base_chat | 932/1319 | 193 (21%) | 655 (70%) | 0 (0%) | 84 (9%) |
| sft_lora | 851/1319 | 73 (9%) | 253 (30%) | 0 (0%) | 525 (62%) |
| sft_full | 883/1319 | 164 (19%) | 665 (75%) | 0 (0%) | 54 (6%) |
| sft_lora_mts | 871/1319 | 184 (21%) | 653 (75%) | 0 (0%) | 34 (4%) |
| sft_cot | 687/1319 | 53 (8%) | 613 (89%) | 0 (0%) | 21 (3%) |
| grpo_acc_step80 | 718/1319 | 172 (24%) | 545 (76%) | 0 (0%) | 1 (0%) |
| grpo_composite_step116 | 727/1319 | 163 (22%) | 564 (78%) | 0 (0%) | 0 (0%) |

Heuristic rules, applied in order, in `analysis/error_analysis.py` (runaway = ≥ 2,000 tokens).

### Why plain LoRA never stops — `analysis/eos_probe.py`

| Model | `<|im_end|>` lm_head norm | P(`<|im_end|>`) after answer | rank |
|---|---|---|---|
| Base | 0.301 (median token 0.461) | 0.0000 | 141,238 |
| SFT: LoRA | 0.301 (unchanged) | 0.0003 | 356 |
| SFT: LoRA + embed/lm_head | 0.306 (untied from embeddings) | 0.9998 | 1 |
| SFT: full fine-tune (lr 1e-5) | 0.302 | 0.13 | 1 |

Median over 50 training problems; full output in `results/logs/eos_probe.log`.

### General knowledge — MMLU / C-Eval valid, 5-shot, log-likelihood

| Model | MMLU | C-Eval (valid) |
|---|---|---|
| Base | 47.6% | 54.3% |
| SFT: full | 47.7% | 55.0% |
| SFT: LoRA + embed/lm_head | 47.0% | 53.6% |
| GRPO, accuracy reward (step 80) | 47.0% | 52.5% |

## Caveats

- One seed per configuration; differences of about 1 pt are within run-to-run noise.
- Zero-shot (verl) and 5-shot (lm-eval) numbers use different prompts and should not be compared
  with each other.
- The GSM8K test split doubles as GRPO's validation set for monitoring only. No checkpoint was
  selected on it; the final step is reported.
- The accuracy-reward GRPO checkpoint at step 116 was deleted by a cleanup bug before lm-eval ran,
  so its 5-shot row is step 80. Its zero-shot curve covers all 116 steps.
- CoT traces are rejection-sampled: 4,426 of 7,473 (59.2%) teacher solutions had the right final
  answer on a `#### n` last line; mean length 206 tokens.

## Reproduce

Hardware: one RTX 4090 (24 GB). SFT takes about 4 minutes per run, GRPO about 100 s per step
(116 steps ≈ 3.3 h), and a GSM8K eval about 2 minutes.

```bash
python scripts/prepare_data.py            # HF openai/gsm8k -> data/ (SFT jsonl + verl parquet)
bash scripts/setup.sh                     # ms-swift, vLLM, lm-eval, base model
bash scripts/s0_install_verl.sh           # verl in a separate venv (transformers pin conflict)

bash scripts/sft.sh                       # LoRA
bash scripts/sft_full.sh                  # full fine-tune
bash scripts/sft_lora_mts.sh              # LoRA + embed_tokens/lm_head
python scripts/gen_cot.py <Qwen2.5-7B-Instruct path> data/gsm8k_cot_train.jsonl
bash scripts/sft_cot.sh                   # CoT distillation
swift export --adapters <checkpoint> --merge_lora true

MODE=full bash scripts/grpo.sh            # accuracy reward
MODE=full bash scripts/grpo_composite.sh  # accuracy + format reward
python -m verl.model_merger merge --backend fsdp --local_dir <step>/actor --target_dir <out>

bash scripts/eval.sh <model> <name> chat  # GSM8K 5-shot
bash scripts/eval_knowledge.sh            # MMLU + C-Eval
python analysis/eos_probe.py <model> ...
python analysis/error_analysis.py name=<lm-eval output dir> ...
python analysis/summarize.py
```

Stack: PyTorch, Hugging Face Transformers, ms-swift (SFT), verl (GRPO, FSDP + vLLM rollout),
vLLM, lm-evaluation-harness.

## Layout

| Path | Contents |
|---|---|
| `scripts/` | Data prep, training, CoT generation, reward function, evaluation |
| `analysis/` | Stop-token probe, error classifier, results-table generator |
| `results/lm_eval/` | Raw lm-eval result JSON for every row above |
| `results/logs/` | GRPO validation curves, stop-token probe output, CoT filter stats |
| `results/error_categories.md` | Error classifier output |
| `tests/` | Unit tests for the composite reward |
