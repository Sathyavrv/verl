#!/bin/bash
set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen3_4b_instruct_grpo.sh <n_gpus_per_node> <save_path> [other_configs...]"
    exit 1
fi

n_gpus_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

# Project root (two levels up from this script)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"

# Remap SFT-style override to PPO keys to avoid Hydra errors
# Accepts either 'model.partial_pretrain=...' or '+model.partial_pretrain=...'
EXTRA_ARGS=()
REMAPPED_MODEL_PATH=""
for arg in "$@"; do
    case "$arg" in
        model.partial_pretrain=*|+model.partial_pretrain=*)
            REMAPPED_MODEL_PATH="${arg#*=}"
            ;;
        *)
            EXTRA_ARGS+=("$arg")
            ;;
    esac
done

# Optional env fallback
if [ -z "$REMAPPED_MODEL_PATH" ] && [ -n "${MODEL_PARTIAL_PRETRAIN:-}" ]; then
    REMAPPED_MODEL_PATH="$MODEL_PARTIAL_PRETRAIN"
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/kaggle/working/data/gsm8k/train.parquet \
    data.val_files=/kaggle/working/data/gsm8k/test.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path="/kaggle/working/models/Qwen3-4B-Instruct-2507" \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name=gsm8k-grpo \
    trainer.experiment_name=gsm8k-grpo-qwen3-4b-instruct-2507 \
    trainer.device=cuda \
    trainer.default_local_dir=$save_path \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    custom_reward_function.path="$PROJECT_DIR/recipe/reward/gsm8k_answer_tag.py" \
    custom_reward_function.name=compute_score \
    +custom_reward_function.reward_kwargs.fallback_to_default=True \
    ${REMAPPED_MODEL_PATH:+actor_rollout_ref.model.path="$REMAPPED_MODEL_PATH"} \
    ${REMAPPED_MODEL_PATH:+critic.model.path="$REMAPPED_MODEL_PATH"} \
    "${EXTRA_ARGS[@]}"


