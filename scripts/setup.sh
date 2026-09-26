#!/bin/bash
# One-time setup on the GPU box (AutoDL, RTX 4090).
set -e
export PATH=/root/miniconda3/bin:$PATH

pip install -U "ms-swift" "vllm" "lm-eval[vllm]"

# Base model from ModelScope to the data disk; training and eval use local paths
modelscope download --model Qwen/Qwen2.5-0.5B --local_dir /root/autodl-tmp/models/Qwen2.5-0.5B
echo "setup done"
