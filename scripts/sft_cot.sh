#!/bin/bash
# CoT-distillation SFT on Qwen2.5-7B-Instruct traces (gen_cot.py). Same as sft_lora_mts.sh except
# max_length 2048 and batch 4 x grad-accum 4 (effective 16; batch 16 OOMs at 2048 on 24 GB).
set -eo pipefail
export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/llm-lab
swift sft \
  --model /root/autodl-tmp/models/Qwen2.5-0.5B \
  --tuner_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --modules_to_save embed_tokens lm_head \
  --dataset data/gsm8k_cot_train.jsonl \
  --split_dataset_ratio 0.02 \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --max_length 2048 \
  --logging_steps 10 \
  --eval_steps 100 \
  --save_steps 100 \
  --save_total_limit 2 \
  --output_dir /root/autodl-tmp/output/sft-cot-mts \
  2>&1 | tee logs/sft_cot.log
