#!/usr/bin/env python3
"""
Data preprocessing script for DeepScaleR dataset.

This script downloads the DeepScaleR dataset from HuggingFace and converts it
to the parquet format required by the VERL training framework.
"""

import os
import argparse
from pathlib import Path
from datasets import load_dataset
import pandas as pd


def download_and_convert_deepscaler(output_dir: str = "data/deepscaler"):
    """
    Download DeepScaleR dataset and convert to parquet format.
    
    Args:
        output_dir: Directory to save the converted parquet file
    """
    print("Loading DeepScaleR dataset from HuggingFace...")
    
    try:
        # Load the dataset
        dataset = load_dataset("agentica-org/DeepScaleR-Preview-Dataset")
        print(f"Dataset loaded successfully: {dataset}")
        
        # Get training split
        train_data = dataset["train"]
        print(f"Training examples: {len(train_data)}")
        
        # Convert to pandas DataFrame
        train_df = train_data.to_pandas()
        print(f"DataFrame shape: {train_df.shape}")
        print(f"Columns: {list(train_df.columns)}")
        
        # Display sample data
        print("\nSample data:")
        print(train_df.head())
        
        # Convert to VERL format
        print("\nConverting to VERL format...")
        verl_data = []
        
        for idx, row in train_df.iterrows():
            # Create the prompt in chat template format
            prompt_content = f"""{row['problem']}\n\n### **INTERNAL REASONING**

WRITE STEP-BY-STEP 

FINALLY PROVIDE USER ANSWER BETWEEN <answer>...</answer>

**Step 1: Parse the Question**

* What is the question asking?
* What format or depth of answer is expected?
* What is the user’s supplied answer (if any)?

**Step 2: Retrieve Relevant Knowledge**

* Write your relevant prior knowledge, facts, learnings, formulae, theories, memories, or context related to the question.

* Write you used that relevant knowledge. 

**Step 3: Generate Multiple Candidate Answer Pathways**

* Outline at least two plausible reasoning paths that could answer the question.
* Briefly summarize each pathway.

**Step 4: Contradiction Audit**

* Compare the user’s supplied answer with prior knowledge.
* Identify any conflicts or contradictions.
* Analyze root causes of contradictions (e.g., differences in definitions, outdated info, missing premises).

**Step 5: Trace Logical Steps**

* For the chosen pathway, explicitly link facts, inferences, and sub-conclusions.
* Show why each step logically follows from the question or evidence.

**Step 6: Explore Alternative Explanations**

* Note other plausible interpretations or answers and why they might be considered.

**Step 7: Decision and Motivation**

* Select the best answer or course of action.
* Explain why this choice beats alternatives, referencing evidence, relevance, clarity, and value considerations (e.g., politeness, safety).
* If adopting the user’s answer, specify what insight led to this change.
* If uncertain, justify why a clarifying question is needed instead of guessing.

**Step 8: Confidence and Error Monitoring**

* Rate confidence as High, Medium, or Low.
* Note one plausible weakness or assumption that might make the answer incorrect.

**Step 9: Reflect on Learning**

* Briefly reflect on what new knowledge, patterns, or reasoning strategies were gained from this example.
* Suggest how future similar questions might be answered better.

---

### **FINAL USER REPLY (DELIVER THIS TO THE USER)**

* Provide a concise, clear, and helpful answer or clarifying question in plain language.
* Avoid jargon, internal tags, or technical explanations.

---"""
            
            # Create VERL format data
            verl_row = {
                "data_source": "agentica-org/DeepScaleR-Preview-Dataset",
                "prompt": [{"role": "user", "content": prompt_content}],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": str(row['answer'])
                },
                "extra_info": {
                    "split": "train",
                    "index": idx,
                    "original_problem": row['problem'],
                    "original_solution": row['solution'] if 'solution' in row else ""
                }
            }
            verl_data.append(verl_row)
        
        # Convert to DataFrame
        verl_df = pd.DataFrame(verl_data)
        print(f"Converted DataFrame shape: {verl_df.shape}")
        print(f"Converted columns: {list(verl_df.columns)}")
        
        # Display sample converted data
        print("\nSample converted data:")
        print(verl_df.head())
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save train split
        output_file = output_path / "train.parquet"
        verl_df.to_parquet(output_file, index=False)
        print(f"\nTrain data saved to: {output_file}")

        # Create a small validation split (10 rows or fewer if dataset is smaller)
        val_rows = min(10, len(verl_df))
        val_df = verl_df.sample(n=val_rows, random_state=42).copy()
        # mark split as val in extra_info
        if "extra_info" in val_df.columns:
            val_df.loc[:, "extra_info"] = val_df["extra_info"].apply(
                lambda d: {**d, "split": "val"} if isinstance(d, dict) else d
            )
        val_file = output_path / "val.parquet"
        val_df.to_parquet(val_file, index=False)
        print(f"Validation data (n={val_rows}) saved to: {val_file}")
        
        # Verify the saved file
        if output_file.exists():
            file_size = output_file.stat().st_size / (1024 * 1024)  # MB
            print(f"File size: {file_size:.2f} MB")
            
            # Load and verify
            loaded_df = pd.read_parquet(output_file)
            print(f"Verified loaded train data shape: {loaded_df.shape}")

            # Verify val file
            loaded_val_df = pd.read_parquet(val_file)
            print(f"Verified loaded val data shape: {loaded_val_df.shape}")
            
            # Check that required columns exist
            required_columns = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
            missing_columns = [col for col in required_columns if col not in loaded_df.columns]
            if missing_columns:
                print(f"Warning: Missing required columns: {missing_columns}")
                return False
            else:
                print("✅ All required columns present")
            
            return True
        else:
            print("Error: File was not created successfully")
            return False
            
    except Exception as e:
        print(f"Error processing dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Prepare DeepScaleR dataset for training")
    parser.add_argument(
        "--output-dir", 
        default="data/deepscaler",
        help="Output directory for the parquet file (default: data/deepscaler)"
    )
    
    args = parser.parse_args()
    
    print("DeepScaleR Dataset Preparation")
    print("=" * 40)
    
    success = download_and_convert_deepscaler(args.output_dir)
    
    if success:
        print("\n✅ Dataset preparation completed successfully!")
        print(f"📁 Data saved to: {args.output_dir}/train.parquet")
        print("\nNext steps:")
        print("1. Update the training script data paths if needed")
        print("2. Run the training script:")
        print(f"   ./run_deepscaler_qwen3_4b_instruct_grpo.sh <n_gpus> <save_path>")
    else:
        print("\n❌ Dataset preparation failed!")
        print("Please check the error messages above.")


if __name__ == "__main__":
    main()
