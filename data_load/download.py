"""
Download a Hugging Face dataset and save as parquet.

Usage:
    python data_load/download.py openai/gsm8k
    python data_load/download.py openai/gsm8k --config main
    python data_load/download.py openai/gsm8k --save_dir ~/data/gsm8k
    python data_load/download.py openai/gsm8k --splits train test
"""

import argparse
import os

import datasets


def main():
    parser = argparse.ArgumentParser(description="Download a HF dataset and save as parquet")
    parser.add_argument("dataset", help="Hugging Face dataset path (e.g. openai/gsm8k)")
    parser.add_argument("--config", default=None, help="Dataset config/subset name (e.g. 'main' for gsm8k)")
    parser.add_argument("--save_dir", default=None, help="Save directory. Defaults to ~/data/<dataset_name>")
    parser.add_argument("--splits", nargs="+", default=None, help="Splits to download (e.g. train test). Downloads all if omitted.")
    args = parser.parse_args()

    # Determine save directory
    if args.save_dir is not None:
        save_dir = os.path.expanduser(args.save_dir)
    else:
        dataset_name = args.dataset.split("/")[-1]
        save_dir = os.path.expanduser(f"~/data/{dataset_name}")

    os.makedirs(save_dir, exist_ok=True)

    # Download
    print(f"Downloading: {args.dataset}" + (f" (config={args.config})" if args.config else ""))
    ds = datasets.load_dataset(args.dataset, args.config)

    # Determine which splits to save
    available_splits = list(ds.keys())
    if args.splits is not None:
        splits = args.splits
        for s in splits:
            if s not in available_splits:
                print(f"Warning: split '{s}' not found. Available: {available_splits}")
    else:
        splits = available_splits

    # Save each split
    for split in splits:
        if split not in ds:
            continue
        out_path = os.path.join(save_dir, f"{split}.parquet")
        ds[split].to_parquet(out_path)
        print(f"  {split}: {len(ds[split])} rows -> {out_path}")

    print(f"Done. Saved to {save_dir}")


if __name__ == "__main__":
    main()
