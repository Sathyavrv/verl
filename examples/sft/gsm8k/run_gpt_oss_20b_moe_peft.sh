#!/bin/bash
set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_gpt_oss_20b_moe_peft.sh <nproc_per_node> <save_path> [other_configs...]"
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
    data.response_key=reward_model \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['ground_truth'] \
    data.system_prompt_path="$PROJECT_DIR/data/gsm8k/reasoning_instruction.txt" \
    data.max_length=2048 \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain="$HOME/models/gpt-oss-20b" \
    model.trust_remote_code=true \
    model.lora_rank=32 \
    model.lora_alpha=16 \
    model.target_modules=[q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj] \
    model.strategy=fsdp \
    model.fsdp_config.model_dtype=bf16 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-gpt-oss-20b-moe-peft \
    trainer.logger=console \
    trainer.total_epochs=1 $@ \
    use_remove_padding=true \
    ulysses_sequence_parallel_size=1 \
    trainer.device=cuda


