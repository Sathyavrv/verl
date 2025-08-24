# DeepScaleR Training Setup - Complete Package

This document summarizes all the files created for training on the DeepScaleR dataset using GRPO.

## 📁 Files Created

### 1. **Reward Function**
- **File**: `recipe/reward/deepscaler_answer_tag.py`
- **Purpose**: Custom reward function that extracts answers from `<answer>...</answer>` tags
- **Features**: 
  - Handles numeric and fractional answers
  - Falls back to default scoring if needed
  - Dataset-specific for DeepScaleR

### 2. **Training Script**
- **File**: `examples/grpo_trainer/run_deepscaler_qwen3_4b_instruct_grpo.sh`
- **Purpose**: Main training script for DeepScaleR dataset
- **Model**: Qwen3-4B-Instruct-2507
- **Algorithm**: GRPO (Group Relative Policy Optimization)

### 3. **Data Preprocessing Script**
- **File**: `examples/grpo_trainer/prepare_deepscaler_data.py`
- **Purpose**: Downloads and converts HuggingFace dataset to parquet format
- **Usage**: `python prepare_deepscaler_data.py --output-dir data/deepscaler`

### 4. **Test Script**
- **File**: `examples/grpo_trainer/test_deepscaler_reward.py`
- **Purpose**: Tests the reward function with various scenarios
- **Usage**: `python test_deepscaler_reward.py`

### 5. **Documentation**
- **File**: `examples/grpo_trainer/README_deepscaler.md`
- **Purpose**: Comprehensive documentation for setup and usage

### 6. **Sample Data**
- **File**: `examples/grpo_trainer/sample_deepscaler_data.json`
- **Purpose**: Example of expected dataset structure

## 🚀 Quick Start

### Step 1: Prepare Data
```bash
cd examples/grpo_trainer
python prepare_deepscaler_data.py
```

### Step 2: Test Reward Function
```bash
python test_deepscaler_reward.py
```

### Step 3: Start Training
```bash
./run_deepscaler_qwen3_4b_instruct_grpo.sh 4 /path/to/save/directory
```

## 🔧 Key Features

- **Answer Tag Extraction**: Uses `<answer>...</answer>` tags for precise answer identification
- **Multiple Answer Formats**: Supports integers, decimals, fractions, and negative numbers
- **Fallback Support**: Gracefully handles cases without answer tags
- **Dataset Specific**: Tailored for DeepScaleR dataset structure
- **Comprehensive Testing**: Includes test suite for validation

## 📊 Dataset Information

- **Source**: [agentica-org/DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset)
- **Size**: ~40K training examples
- **Format**: Math problems with structured answers
- **Columns**: problem, answer, solution

## 🎯 Training Configuration

- **Model**: Qwen3-4B-Instruct-2507
- **LoRA**: Rank 64, Alpha 32
- **Batch Size**: 64 (train), 32 (PPO)
- **Learning Rate**: 2e-5
- **Rollouts**: 5 per prompt
- **Training Steps**: 50 (configurable)

## 📝 Expected Model Output Format

The model should be instructed to provide answers in this format:
```
<answer>final_answer</answer>
```

Examples:
- `<answer>18</answer>`
- `<answer>-2/3</answer>`
- `<answer>3.14</answer>`

## 🔍 Troubleshooting

1. **Dataset Issues**: Check parquet file exists and has correct structure
2. **Memory Issues**: Reduce batch sizes or enable gradient checkpointing
3. **Reward Errors**: Verify reward function path and dataset source name
4. **Import Errors**: Ensure project root is in Python path

## 📚 Additional Resources

- **VERL Documentation**: See main project docs for advanced configuration
- **GRPO Algorithm**: Group Relative Policy Optimization details
- **LoRA Training**: Low-Rank Adaptation for efficient fine-tuning

---

**Status**: ✅ Complete and Ready for Use
**Last Updated**: 2025
**Compatibility**: VERL Framework, PyTorch, CUDA
