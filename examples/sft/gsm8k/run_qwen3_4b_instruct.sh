#!/bin/bash
set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen3_4b_instruct.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

# Project root (three levels up from this script)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['answer'] \
    data.system_prompt_path="$PROJECT_DIR/data/gsm8k/reasoning_instruction.txt" \
    data.max_length=2048 \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain="$HOME/models/Qwen3-4B-Instruct-2507" \
    model.trust_remote_code=true \
    model.lora_rank=64 \
    model.lora_alpha=32 \
    model.target_modules=all-linear \
    model.strategy=fsdp \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-qwen3-4b-instruct-2507 \
    trainer.logger=console \
    optim.lr=1e-4 \
    trainer.total_epochs=1 \
    trainer.device=cuda \
    trainer.self_training.enable=true \
    trainer.self_training.numeric_match_only=true \
    trainer.self_training.mismatch_retry_enable=true \
    trainer.self_training.max_retries=1 \
    trainer.self_training.verbose_logs=true \
    trainer.preview_assistant_only=false \
    trainer.preview_num_tokens=0 \
    trainer.preview_every_n_steps=1 \
    $@