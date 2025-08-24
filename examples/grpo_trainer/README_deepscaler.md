# DeepScaleR Dataset GRPO Training

This directory contains the training script for the DeepScaleR dataset using GRPO (Group Relative Policy Optimization).

## Dataset Structure

The DeepScaleR dataset should be preprocessed into a parquet format with the following columns:

- **problem**: The math problem text (prompt)
- **answer**: The ground truth answer (target for reward computation)
- **solution**: The detailed solution (optional, for reference)

### Dataset Statistics
- **Size**: ~40K training examples
- **Format**: JSON converted to Parquet
- **Source**: [agentica-org/DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset)

## Reward Function

The reward function (`recipe/reward/deepscaler_answer_tag.py`) is designed to:

1. **Extract answers** from `<answer>...</answer>` tags in model responses
2. **Handle various answer formats**:
   - Regular numbers (e.g., 18, -2, 1,234, 3.14)
   - Fractions (e.g., -2/3, 1/2)
3. **Score responses** based on exact match with ground truth
4. **Fall back** to default scoring if no answer tag is found

## Training Script

### Usage

```bash
./run_deepscaler_qwen3_4b_instruct_grpo.sh <n_gpus_per_node> <save_path> [other_configs...]
```

### Example

```bash
./run_deepscaler_qwen3_4b_instruct_grpo.sh 4 /path/to/save/directory
```

### Key Configuration

- **Model**: Qwen3-4B-Instruct-2507
- **Algorithm**: GRPO (Group Relative Policy Optimization)
- **LoRA**: Rank 64, Alpha 32
- **Batch Size**: 64 (train), 32 (PPO mini-batch)
- **Learning Rate**: 2e-5
- **Training Steps**: 50 (configurable)
- **Rollout**: 5 samples per prompt

## Data Preparation

### Converting from HuggingFace Dataset

```python
from datasets import load_dataset
import pandas as pd

# Load the dataset
dataset = load_dataset("agentica-org/DeepScaleR-Preview-Dataset")

# Convert to parquet
train_df = dataset["train"].to_pandas()
train_df.to_parquet("data/deepscaler/train.parquet", index=False)
```

### Expected Directory Structure

```
data/
└── deepscaler/
    └── train.parquet
```

## Model Response Format

The model should be instructed to provide answers in the following format:

```
<answer>final_answer</answer>
```

For example:
- `<answer>18</answer>`
- `<answer>-2/3</answer>`
- `<answer>3.14</answer>`

## Customization

### Modifying Reward Function

Edit `recipe/reward/deepscaler_answer_tag.py` to:
- Change answer extraction patterns
- Modify scoring logic
- Add additional answer format support

### Training Parameters

Key parameters can be modified in the script:
- `data.train_batch_size`: Training batch size
- `actor_rollout_ref.actor.optim.lr`: Learning rate
- `trainer.total_training_steps`: Number of training steps
- `actor_rollout_ref.rollout.n`: Number of rollouts per prompt

## Troubleshooting

### Common Issues

1. **Dataset not found**: Ensure the parquet file exists at the specified path
2. **Memory issues**: Reduce batch sizes or enable gradient checkpointing
3. **Reward computation errors**: Check that the reward function path is correct

### Debugging

- Enable verbose logging with `trainer.logger=console`
- Check reward function output in logs
- Verify dataset format matches expected structure

## Performance Notes

- **GPU Memory**: ~6GB per GPU with current settings
- **Training Time**: ~2-3 hours for 50 steps on 4x A100
- **Scaling**: Linear scaling with number of GPUs
