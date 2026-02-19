"""
Download and prepare CL-Bench dataset from HuggingFace.

Usage:
    python download_data.py

Output:
    - clbench_tasks.parquet (local dev file)
    - Prints instructions for /orwd_data upload
"""

from datasets import load_dataset
import pandas as pd
from pathlib import Path


def main():
    print("Downloading CL-Bench dataset from HuggingFace...")
    print("Note: HuggingFace calls the split 'train' but these are evaluation tasks")
    print()

    # Load dataset from HuggingFace (split is named "train" but contains evaluation tasks)
    dataset = load_dataset("tencent/CL-bench", split="train")

    print(f"✓ Loaded {len(dataset)} tasks")

    # Convert to DataFrame
    df = pd.DataFrame(dataset)

    # Validate structure
    required_columns = ["messages", "rubrics", "metadata"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"✓ Validated dataset structure")
    print(f"  - Columns: {list(df.columns)}")
    print(f"  - Sample rubric count: {len(df['rubrics'].iloc[0])}")

    # Handle NaN values and clean up (important for JSON serialization)
    if '__index_level_0__' in df.columns:
        df = df.drop('__index_level_0__', axis=1)
        print(f"✓ Dropped __index_level_0__ column")

    # Replace NaN with empty strings to avoid JSON serialization errors
    df = df.fillna('')
    print(f"✓ Replaced NaN values with empty strings")

    # Save to parquet
    output_path = Path(__file__).parent / "clbench_tasks.parquet"
    df.to_parquet(output_path, index=False)

    print(f"\n✓ Saved to {output_path}")
    print(f"✓ File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Print upload instructions
    print("\n" + "="*60)
    print("NEXT STEPS FOR DEPLOYMENT:")
    print("="*60)
    print("1. Create namespace 'EnvCommons/clbench' on openreward.ai")
    print("2. Upload clbench_tasks.parquet to: /orwd_data/clbench/")
    print("3. Deploy environment using GitHub repo: EnvCommons/clbench")
    print("="*60)
    print("\nFor local testing, the file is ready to use!")


if __name__ == "__main__":
    main()
