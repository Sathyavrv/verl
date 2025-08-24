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
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save to parquet
        output_file = output_path / "train.parquet"
        train_df.to_parquet(output_file, index=False)
        print(f"\nData saved to: {output_file}")
        
        # Verify the saved file
        if output_file.exists():
            file_size = output_file.stat().st_size / (1024 * 1024)  # MB
            print(f"File size: {file_size:.2f} MB")
            
            # Load and verify
            loaded_df = pd.read_parquet(output_file)
            print(f"Verified loaded data shape: {loaded_df.shape}")
            
            return True
        else:
            print("Error: File was not created successfully")
            return False
            
    except Exception as e:
        print(f"Error processing dataset: {e}")
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
