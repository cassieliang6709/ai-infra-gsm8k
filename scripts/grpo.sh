#!/bin/bash
# GRPO on Qwen2.5-0.5B (after LoRA + modules_to_save SFT, merged) | single RTX 4090 | FSDP + vLLM rollout
# Adapted from verl examples/grpo_trainer/run_qwen3_4b_fsdp.sh (commit 6093e00)
# Usage: MODE=smoke bash scripts/grpo.sh   (2-step wiring check)
#        MODE=full  bash scripts/grpo.sh   (2 epochs, 116 steps)
set -xeo pipefail
source /root/autodl-tmp/envverl/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/llm-lab
MODEL_PATH=${MODEL_PATH:-$(ls -d /root/autodl-tmp/output/sft-lora-mts/*/checkpoint-458-merged | head -1)}
MODE=${MODE:-smoke}
if [ "$MODE" = smoke ]; then
  TRAIN_BATCH_SIZE=16; MINI=8; ROLLOUT_N=4; STEPS="trainer.total_training_steps=2"; SAVE_FREQ=-1; TEST_FREQ=-1; VAL_BEFORE=False; EXP=smoke
else
  TRAIN_BATCH_SIZE=128; MINI=64; ROLLOUT_N=8; STEPS="trainer.total_epochs=2"; SAVE_FREQ=20; TEST_FREQ=20; VAL_BEFORE=True; EXP=grpo_sft_mts
fi
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files=data/verl/train.parquet \
  data.val_files=data/verl/test.parquet \
  data.train_batch_size=$TRAIN_BATCH_SIZE \
  data.max_prompt_length=512 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.dataloader_num_workers=2 \
  actor_rollout_ref.model.path=$MODEL_PATH \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  ++actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=6144 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.n=$ROLLOUT_N \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.project_name=gsm8k_grpo \
  trainer.experiment_name=$EXP \
  trainer.default_local_dir=/root/autodl-tmp/output/verl/$EXP \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=$TEST_FREQ \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.val_before_train=$VAL_BEFORE \
  $STEPS \
  ray_kwargs.ray_init.runtime_env.py_executable=null \
  "$@"
