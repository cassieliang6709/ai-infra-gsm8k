"""Print the README results tables from the raw lm-eval outputs in results/lm_eval/."""
import json
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "results" / "lm_eval"
load = lambda n: json.load(open(D / f"{n}.json"))["results"]

GSM8K = [
    ("base_plain", "Base, raw-text prompt"),
    ("base_chat", "Base, chat template"),
    ("sft_lora_chat", "SFT: LoRA"),
    ("sft_full_chat", "SFT: full fine-tune"),
    ("sft_lora_mts_chat", "SFT: LoRA + trainable embed/lm_head"),
    ("sft_cot_chat", "SFT: CoT distillation (7B teacher)"),
    ("grpo_acc_step80_chat", "GRPO, accuracy reward (step 80)"),
    ("grpo_composite_step116_chat", "GRPO, accuracy + format reward (step 116)"),
]
print("| Model | strict-match | flexible-extract |\n|---|---|---|")
for key, label in GSM8K:
    r = load(key)["gsm8k"]
    print(f"| {label} | {r['exact_match,strict-match']:.1%} | {r['exact_match,flexible-extract']:.1%} |")

print("\n| Model | MMLU | C-Eval (valid) |\n|---|---|---|")
for key, label in [("knowledge_base", "Base"), ("knowledge_sft_full", "SFT: full"),
                   ("knowledge_sft_lora_mts", "SFT: LoRA + embed/lm_head"),
                   ("knowledge_grpo_acc_step80", "GRPO, accuracy reward (step 80)")]:
    r = load(key)
    print(f"| {label} | {r['mmlu']['acc,none']:.1%} | {r['ceval-valid']['acc,none']:.1%} |")
