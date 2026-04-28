"""
Filter allenai/Dolci-Think-RL-7B for math-only rows and save as verl parquet.

Source dataset:
    allenai/Dolci-Think-RL-7B  (train split, ~102k rows, 4 GB)
    - Mixes math / coding / instruction-following / general chat
    - Filter key: row["dataset"] is a list like ["math"], ["general-quality"], ...

Output schema (verl準拠):
    data_source: str
    prompt:       list[{"role": "user", "content": str}]
    ability:      "math"
    reward_model: {"style": "rule", "ground_truth": str}
    extra_info:   {split, index, custom_id, original_dataset, passrate}

Usage:
    uv run python data_load/dolci_think_rl_7b_math.py
    uv run python data_load/dolci_think_rl_7b_math.py --save_dir data/Dolci-Think-RL-7B-math
"""

import argparse
import os

import datasets

DATA_SOURCE = "allenai/Dolci-Think-RL-7B"

# prompt フィールド先頭に付いている "user: " プレフィックスを取り除く用
USER_PREFIX = "user: "


def is_math(example) -> bool:
    ds = example.get("dataset")
    if ds is None:
        return False
    if isinstance(ds, str):
        return ds == "math"
    # list[str] 形式
    return "math" in ds


def build_row(example, idx, split):
    prompt_str = example["prompt"]
    if prompt_str.startswith(USER_PREFIX):
        prompt_str = prompt_str[len(USER_PREFIX):]

    # ground_truth は list[str] で格納されている。数学問題は単一解答なので先頭を採用
    gt = example["ground_truth"]
    if isinstance(gt, list):
        gt = gt[0] if gt else ""

    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": prompt_str}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(gt)},
        "extra_info": {
            "split": split,
            "index": idx,
            "custom_id": str(example.get("custom_id") or ""),
            "original_dataset": str(example.get("original_dataset") or ""),
            "passrate": float(example["passrate"]) if example.get("passrate") is not None else -1.0,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="data/Dolci-Think-RL-7B-math")
    parser.add_argument("--out_name", default="train.parquet",
                        help="Output filename (train用なので train.parquet 既定)")
    parser.add_argument("--num_proc", type=int, default=8,
                        help="filter/map の並列プロセス数")
    args = parser.parse_args()

    save_dir = os.path.expanduser(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading {DATA_SOURCE}...")
    ds = datasets.load_dataset(DATA_SOURCE)
    split = "train"
    print(f"  original: {len(ds[split])} rows")

    # math だけ抽出
    math_ds = ds[split].filter(is_math, num_proc=args.num_proc)
    print(f"  after math filter: {len(math_ds)} rows")

    # verl 形式に整形 + 巨大な input_ids / outputs カラムは drop
    formatted = math_ds.map(
        lambda ex, idx: build_row(ex, idx, split),
        with_indices=True,
        remove_columns=math_ds.column_names,
        num_proc=args.num_proc,
    )

    out_path = os.path.join(save_dir, args.out_name)
    formatted.to_parquet(out_path)
    print(f"  {len(formatted)} rows -> {out_path}")


if __name__ == "__main__":
    main()
