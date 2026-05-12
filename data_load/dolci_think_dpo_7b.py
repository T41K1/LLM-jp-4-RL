#!/usr/bin/env python3
"""Prepare allenai/Dolci-Think-DPO-7B for NeMo-RL DPO."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset


DATA_SOURCE = "allenai/Dolci-Think-DPO-7B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert allenai/Dolci-Think-DPO-7B to NeMo-RL "
            "BinaryPreferenceDataset JSONL."
        )
    )
    parser.add_argument("--dataset", default=DATA_SOURCE)
    parser.add_argument("--split", default="train")
    parser.add_argument("--save-dir", type=Path, default=Path("data/Dolci-Think-DPO-7B/nemo-rl"))
    parser.add_argument("--train-name", default="train.jsonl")
    parser.add_argument("--val-name", default="val.jsonl")
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-val-examples", type=int, default=None)
    parser.add_argument("--max-prompt-chars", type=int, default=50000)
    parser.add_argument("--max-response-chars", type=int, default=80000)
    parser.add_argument("--dataset-source", action="append", default=None)
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--max-stats-examples", type=int, default=None)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def as_builtin(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def content_from_message(value: Any) -> str:
    value = as_builtin(value)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "content" in value:
            return str(value["content"])
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "\n".join(content_from_message(item) for item in value)
    return str(value)


def first_role_content(messages: Any, role: str) -> str:
    messages = as_builtin(messages)
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == role:
            return content_from_message(message).strip()
    return ""


def last_role_content(messages: Any, role: str) -> str:
    messages = as_builtin(messages)
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == role:
            return content_from_message(message).strip()
    return ""


def assistant_response(messages: Any) -> str:
    response = last_role_content(messages, "assistant")
    if response:
        return response

    messages = as_builtin(messages)
    if isinstance(messages, list) and messages:
        return content_from_message(messages[-1]).strip()
    return content_from_message(messages).strip()


def prompt_text(example: dict[str, Any]) -> str:
    prompt = str(example.get("prompt") or "").strip()
    if prompt:
        return prompt

    for key in ("chosen", "rejected"):
        prompt = first_role_content(example.get(key), "user")
        if prompt:
            return prompt
    return ""


def stable_val_assignment(example_id: str, val_ratio: float) -> bool:
    if val_ratio <= 0:
        return False
    digest = hashlib.sha1(example_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:12], 16) / float(16**12)
    return bucket < val_ratio


def format_example(example: dict[str, Any]) -> dict[str, str] | None:
    prompt = prompt_text(example)
    chosen = assistant_response(example.get("chosen"))
    rejected = assistant_response(example.get("rejected"))

    if not prompt or not chosen or not rejected:
        return None
    if chosen == rejected:
        return None

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def passes_filters(
    record: dict[str, str],
    max_prompt_chars: int | None,
    max_response_chars: int | None,
) -> bool:
    if max_prompt_chars is not None and len(record["prompt"]) > max_prompt_chars:
        return False
    if max_response_chars is not None:
        if len(record["chosen"]) > max_response_chars:
            return False
        if len(record["rejected"]) > max_response_chars:
            return False
    return True


def ensure_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def write_record(handle, record: dict[str, str]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = math.ceil((q / 100.0) * len(ordered)) - 1
    index = min(max(index, 0), len(ordered) - 1)
    return ordered[index]


def print_length_stats(
    lengths: dict[str, list[int]],
    counts: dict[str, int],
    max_prompt_chars: int | None,
    max_response_chars: int | None,
) -> None:
    print("stats:")
    for key, value in counts.items():
        print(f"  {key}: {value}")

    for key in ("prompt", "chosen", "rejected", "max_response", "total"):
        values = lengths[key]
        print(f"{key}_chars:")
        if not values:
            print("  count: 0")
            continue
        print(f"  count: {len(values)}")
        print(f"  min: {min(values)}")
        print(f"  p50: {percentile(values, 50)}")
        print(f"  p90: {percentile(values, 90)}")
        print(f"  p95: {percentile(values, 95)}")
        print(f"  p99: {percentile(values, 99)}")
        print(f"  p99.5: {percentile(values, 99.5)}")
        print(f"  p99.9: {percentile(values, 99.9)}")
        print(f"  max: {max(values)}")

    if max_prompt_chars is not None:
        over_prompt = sum(length > max_prompt_chars for length in lengths["prompt"])
        print(f"over_max_prompt_chars({max_prompt_chars}): {over_prompt}")
    if max_response_chars is not None:
        over_response = sum(length > max_response_chars for length in lengths["max_response"])
        print(f"over_max_response_chars({max_response_chars}): {over_response}")
    sys.stdout.flush()
    sys.stderr.flush()


def collect_length_stats(args: argparse.Namespace) -> None:
    allowed_sources = set(args.dataset_source or [])
    ds = load_dataset(args.dataset, split=args.split, streaming=args.streaming)
    counts = {
        "seen": 0,
        "valid_pairs": 0,
        "skipped_source": 0,
        "skipped_empty_or_equal": 0,
    }
    lengths: dict[str, list[int]] = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "max_response": [],
        "total": [],
    }

    for example in ds:
        counts["seen"] += 1
        if allowed_sources and example.get("dataset_source") not in allowed_sources:
            counts["skipped_source"] += 1
            continue

        record = format_example(example)
        if record is None:
            counts["skipped_empty_or_equal"] += 1
            continue

        counts["valid_pairs"] += 1
        prompt_len = len(record["prompt"])
        chosen_len = len(record["chosen"])
        rejected_len = len(record["rejected"])
        lengths["prompt"].append(prompt_len)
        lengths["chosen"].append(chosen_len)
        lengths["rejected"].append(rejected_len)
        lengths["max_response"].append(max(chosen_len, rejected_len))
        lengths["total"].append(prompt_len + chosen_len + rejected_len)

        if args.max_stats_examples is not None and counts["valid_pairs"] >= args.max_stats_examples:
            break

    print_length_stats(lengths, counts, args.max_prompt_chars, args.max_response_chars)


def main() -> None:
    args = parse_args()

    if args.stats_only:
        collect_length_stats(args)
        # datasets streaming can leave native cleanup hooks that trip Python
        # finalization on some cluster images after all output has been written.
        os._exit(0)

    train_path = args.save_dir / args.train_name
    val_path = args.save_dir / args.val_name
    ensure_writable(train_path, args.overwrite)
    ensure_writable(val_path, args.overwrite)

    allowed_sources = set(args.dataset_source or [])
    ds = load_dataset(args.dataset, split=args.split, streaming=args.streaming)

    counts = {
        "seen": 0,
        "train": 0,
        "val": 0,
        "skipped_source": 0,
        "skipped_empty_or_equal": 0,
        "skipped_length": 0,
        "skipped_train_cap": 0,
        "skipped_val_cap": 0,
    }

    with train_path.open("w") as train_f, val_path.open("w") as val_f:
        for idx, example in enumerate(ds):
            counts["seen"] += 1

            if allowed_sources and example.get("dataset_source") not in allowed_sources:
                counts["skipped_source"] += 1
                continue

            record = format_example(example)
            if record is None:
                counts["skipped_empty_or_equal"] += 1
                continue
            if not passes_filters(record, args.max_prompt_chars, args.max_response_chars):
                counts["skipped_length"] += 1
                continue

            example_id = str(example.get("id") or idx)
            is_val = stable_val_assignment(example_id, args.val_ratio)
            if is_val:
                if args.max_val_examples is not None and counts["val"] >= args.max_val_examples:
                    counts["skipped_val_cap"] += 1
                    continue
                write_record(val_f, record)
                counts["val"] += 1
            else:
                if args.max_train_examples is not None and counts["train"] >= args.max_train_examples:
                    counts["skipped_train_cap"] += 1
                    continue
                write_record(train_f, record)
                counts["train"] += 1

            train_done = args.max_train_examples is not None and counts["train"] >= args.max_train_examples
            val_done = args.max_val_examples is not None and counts["val"] >= args.max_val_examples
            if train_done and val_done:
                break

    print(f"wrote train: {counts['train']} -> {train_path}")
    print(f"wrote val:   {counts['val']} -> {val_path}")
    print("stats:")
    for key, value in counts.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
