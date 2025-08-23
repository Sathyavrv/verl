# Tested on small single-GPU setup. Uses HF rollout and disables FSDP wrap policy to support Gemma 3 270M IT.
set -x
if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen3_4b_instruct_grpo.sh <n_gpus_per_node> <save_path> [other_configs...]"
    exit 1
fi

n_gpus_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/kaggle/working/data/gsm8k/train.parquet \
    data.val_files=/kaggle/working/data/gsm8k/train.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/kaggle/input/gemma-3/transformers/gemma-3-270m-it/1 \
    actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=2e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    +actor_rollout_ref.actor.fsdp_config.wrap_policy.disable=True \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.ref.fsdp_config.wrap_policy.disable=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=console\
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='gemma3_270m_it_grpo' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=5 $@


