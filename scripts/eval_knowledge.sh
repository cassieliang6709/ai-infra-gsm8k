#!/bin/bash
# General-knowledge check: MMLU (English) + C-Eval valid (Chinese), 5-shot, to see whether math-only SFT hurts other skills.
# Multiple choice is scored by option log-likelihood, so no chat template (same setting for every model).
export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
cd /root/autodl-tmp/llm-lab
B=/root/autodl-tmp/models/Qwen2.5-0.5B
F=$(ls -d /root/autodl-tmp/output/sft-full-gsm8k/*/checkpoint-458)
M=$(ls -d /root/autodl-tmp/output/sft-lora-mts/*/checkpoint-458-merged)
for a in "$B base" "$F sft_full" "$M sft_lora_mts"; do
  set -- $a; S=$(date +%s)
  lm_eval --model vllm \
    --model_args pretrained=$1,dtype=bfloat16,gpu_memory_utilization=0.8 \
    --tasks mmlu,ceval-valid --num_fewshot 5 --batch_size auto \
    --output_path results/knowledge_$2 2>&1 | tee logs/eval_knowledge_$2.log
  echo "knowledge_$2 $(( $(date +%s)-S ))s rc=${PIPESTATUS[0]}" >> logs/timing.log
done
echo "KNOWLEDGE DONE" >> logs/timing.log
