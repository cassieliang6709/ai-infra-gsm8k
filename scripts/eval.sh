#!/bin/bash
# Usage: bash scripts/eval.sh <model path> <result name> [chat|plain]
#   chat  = Qwen chat template, few-shot as multi-turn (required after SFT)
#   plain = raw text prompt (sanity check against the HF-backend baseline, 35.1%)
# GSM8K 5-shot, vLLM backend, greedy, max_gen_toks 2048.
set -eo pipefail
export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/llm-lab
export HF_ENDPOINT=https://hf-mirror.com
MODEL=$1; NAME=$2; MODE=${3:-chat}
EXTRA=""
[ "$MODE" = "chat" ] && EXTRA="--apply_chat_template --fewshot_as_multiturn"
lm_eval --model vllm \
  --model_args pretrained=$MODEL,dtype=bfloat16,gpu_memory_utilization=0.8 \
  --tasks gsm8k --num_fewshot 5 $EXTRA \
  --gen_kwargs max_gen_toks=2048 --batch_size auto \
  --output_path results/$NAME --log_samples \
  2>&1 | tee logs/eval_$NAME.log
