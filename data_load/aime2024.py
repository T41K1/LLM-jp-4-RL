"""
Preprocess AIME 2024 (Maxwell-Jia/AIME_2024) into verl parquet format.

Schema per row:
    data_source: str
    prompt:       list[{"role": "user", "content": str}]
    ability:      "math"
    reward_model: {"style": "rule", "ground_truth": str}   # e.g. "33"
    extra_info:   {"split", "index", "id", "solution"}

Usage:
    uv run python data_load/aime2024.py --save_dir data/AIME2024
"""

import argparse
import os

import datasets

DATA_SOURCE = "Maxwell-Jia/AIME_2024"

#verlでの実装準拠
INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


def build_row(example, idx, split):
    problem = example["Problem"]
    answer = example["Answer"]
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": problem + " " + INSTRUCTION}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(answer)},
        "extra_info": {
            "split": split,
            "index": idx,
            "id": example["ID"],
            "solution": example["Solution"],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="data/AIME2024")
    parser.add_argument("--out_name", default="test.parquet",
                        help="Output filename (val用途なのでtest.parquet既定)")
    args = parser.parse_args()

    save_dir = os.path.expanduser(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading {DATA_SOURCE}...")
    ds = datasets.load_dataset(DATA_SOURCE)
    split = "train"  # AIME_2024 provides only 'train' (all 30 problems)

    formatted = ds[split].map(
        lambda ex, idx: build_row(ex, idx, split),
        with_indices=True,
        remove_columns=ds[split].column_names,
    )

    out_path = os.path.join(save_dir, args.out_name)
    formatted.to_parquet(out_path)
    print(f"  {len(formatted)} rows -> {out_path}")


if __name__ == "__main__":
    main()
