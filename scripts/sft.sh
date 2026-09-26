#!/bin/bash
# LoRA SFT (rank 16, all linear layers): Qwen2.5-0.5B base, GSM8K train (7,473), 1 epoch
set -eo pipefail
export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/llm-lab
swift sft \
  --model /root/autodl-tmp/models/Qwen2.5-0.5B \
  --tuner_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --dataset data/gsm8k_train.jsonl \
  --split_dataset_ratio 0.02 \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 16 \
  --learning_rate 1e-4 \
  --max_length 1024 \
  --logging_steps 10 \
  --eval_steps 100 \
  --save_steps 100 \
  --save_total_limit 2 \
  --output_dir /root/autodl-tmp/output/sft-gsm8k \
  2>&1 | tee logs/sft.log
