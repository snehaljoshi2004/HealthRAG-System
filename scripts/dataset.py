# dataset.py

from datasets import load_dataset
import pandas as pd
import os


def download_ragcare_qa():
    print("Downloading RAGCare-QA dataset...")

    # Create directory if it doesn't exist
    os.makedirs("data/evaluation", exist_ok=True)

    # Load dataset from HuggingFace
    dataset = load_dataset("ChatMED-Project/RAGCare-QA")

    # Convert to pandas
    df = dataset["train"].to_pandas()

    print(f"Dataset loaded: {len(df)} questions")
    print(f"Columns: {df.columns.tolist()}")

    # Normalize complexity values
    df["Complexity"] = df["Complexity"].str.lower()

    # Save full dataset
    full_path = "data/evaluation/ragcare_qa_full.json"
    df.to_json(full_path, orient="records", indent=2)

    print(f"Saved full dataset to {full_path}")

    # Create balanced evaluation dataset
    sample_size = min(100, len(df))

    sampled_df = (
        df.groupby("Complexity", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), sample_size // 3), random_state=42))
        .reset_index(drop=True)
    )

    # Save golden dataset
    golden_path = "data/evaluation/golden_qa_pairs.json"
    sampled_df.to_json(golden_path, orient="records", indent=2)

    print(f"\nCreated golden dataset with {len(sampled_df)} questions")

    print(f"  - Basic: {len(sampled_df[sampled_df['Complexity'] == 'basic'])}")
    print(f"  - Intermediate: {len(sampled_df[sampled_df['Complexity'] == 'intermediate'])}")
    print(f"  - Advanced: {len(sampled_df[sampled_df['Complexity'] == 'advanced'])}")

    # Show one sample
    print("\nSample question:")

    sample = sampled_df.iloc[0]

    print(f"Question: {sample['Question']}")
    print(f"Answer: {sample['Answer']}")
    print(f"Complexity: {sample['Complexity']}")
    print(f"Specialty: {sample['Type']}")

    if "Context" in sampled_df.columns:
        print(f"Context: {sample['Context'][:200]}...")


if __name__ == "__main__":
    download_ragcare_qa()