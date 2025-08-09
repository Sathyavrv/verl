# Tested with 2 & 4 GPUs

set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen_05_peft.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

# Project root (three levels up from this script)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
HF_ATTENTION_BACKEND=${HF_ATTENTION_BACKEND:-sdpa} \
PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=reward_model \
    optim.lr=1e-4 \
    data.prompt_dict_keys=[question] \
    data.response_dict_keys=[ground_truth] \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=Qwen/Qwen2.5-0.5B-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-qwen-2.5-0.5b-instruct \
    trainer.logger=console \
    trainer.total_epochs=1 $@ \
    model.lora_rank=32\
    model.lora_alpha=16 \
    model.target_modules=all-linear \
    data.system_prompt_path="$PROJECT_DIR/data/gsm8k/reasoning_instruction.txt"

    # Or you can do this:
    # model.target_modules=[q_proj,v_proj] \
