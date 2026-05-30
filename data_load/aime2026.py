"""
Preprocess AIME 2026 (MathArena/aime_2026) into verl parquet format.

Schema per row:
    data_source: str
    prompt:       list[{"role": "user", "content": str}]
    ability:      "math"
    reward_model: {"style": "rule", "ground_truth": str}   # e.g. "70"
    extra_info:   {"split", "index", "problem_idx"}

Usage:
    uv run python data_load/aime2026.py --save_dir data/AIME2026
"""

import argparse
import os

import datasets

DATA_SOURCE = "MathArena/aime_2026"

# verlでの実装準拠
INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


def build_row(example, idx, split):
    problem = example["problem"]
    answer = example["answer"]
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": problem + " " + INSTRUCTION}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(answer)},
        "extra_info": {
            "split": split,
            "index": idx,
            "id": str(example["problem_idx"]),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="data/AIME2026")
    parser.add_argument(
        "--out_name",
        default="test.parquet",
        help="Output filename (val用途なのでtest.parquet既定)",
    )
    args = parser.parse_args()

    save_dir = os.path.expanduser(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading {DATA_SOURCE}...")
    ds = datasets.load_dataset(DATA_SOURCE)
    split = "train"  # MathArena/aime_2026 provides only 'train' (all 30 problems)

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
