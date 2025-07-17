# File: scripts/prepare_data.py

from datasets import load_dataset
import os

def main():
    """
    Downloads and caches the SST-2 dataset from the Hugging Face Hub.
    """
    dataset_name = "glue"
    subset_name = "sst2"
    
    # Define a cache directory within your work folder for better management
    cache_dir = os.path.join(os.environ.get("WORK", "."), "hf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Downloading dataset '{dataset_name}' subset '{subset_name}'...")
    print(f"Using cache directory: {cache_dir}")

    # This command downloads the data to the specified cache directory
    load_dataset(dataset_name, subset_name, cache_dir=cache_dir)
    print("======================================================")
    print("Dataset downloaded and cached successfully.")
    print("======================================================")


if __name__ == "__main__":
    main()