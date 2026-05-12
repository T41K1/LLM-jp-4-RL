#!/usr/bin/env python3
"""Create NeMo-RL BinaryPreferenceDataset JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROMPT_CANDIDATES = ("prompt", "input", "context", "question")
CHOSEN_CANDIDATES = ("chosen", "chosen_response", "winner", "accepted")
REJECTED_CANDIDATES = ("rejected", "rejected_response", "loser", "rejected_output")
RESPONSE_CANDIDATES = ("output", "response", "completion", "generated_text")
SCORE_CANDIDATES = ("score", "reward", "acc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert pairwise preference data, or scored generation logs, into "
            "NeMo-RL BinaryPreferenceDataset JSONL."
        )
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--input-format", choices=("auto", "parquet", "jsonl", "json"), default="auto")
    parser.add_argument("--prompt-key", default=None)
    parser.add_argument("--chosen-key", default=None)
    parser.add_argument("--rejected-key", default=None)
    parser.add_argument("--group-key", default=None)
    parser.add_argument("--response-key", default=None)
    parser.add_argument("--score-key", default=None)
    parser.add_argument("--min-score-gap", type=float, default=0.0)
    parser.add_argument("--max-prompt-chars", type=int, default=None)
    parser.add_argument("--max-response-chars", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--allow-equal-score", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_table(path: Path, input_format: str) -> pd.DataFrame:
    if input_format == "auto":
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            input_format = "parquet"
        elif suffix == ".jsonl":
            input_format = "jsonl"
        elif suffix == ".json":
            input_format = "json"
        else:
            raise ValueError(f"Could not infer input format from suffix: {path}")

    if input_format == "parquet":
        return pd.read_parquet(path)
    if input_format == "jsonl":
        return pd.read_json(path, lines=True)
    if input_format == "json":
        with path.open() as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("data", data.get("records", data))
        return pd.DataFrame(data)
    raise ValueError(f"Unsupported input format: {input_format}")


def first_existing(columns: Iterable[str], candidates: Iterable[str], label: str) -> str:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    raise ValueError(f"Could not find {label} column. Tried: {', '.join(candidates)}")


def as_builtin(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def stringify_message_value(value: Any) -> str:
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
        if all(isinstance(item, dict) and "content" in item for item in value):
            if len(value) == 1:
                return str(value[0]["content"])
            return "\n".join(f"{item.get('role', 'message')}: {item['content']}" for item in value)
        return "\n".join(stringify_message_value(item) for item in value)
    return str(value)


def score_value(value: Any) -> float:
    value = as_builtin(value)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value)


def within_length_limits(
    prompt: str,
    chosen: str,
    rejected: str,
    max_prompt_chars: int | None,
    max_response_chars: int | None,
) -> bool:
    if max_prompt_chars is not None and len(prompt) > max_prompt_chars:
        return False
    if max_response_chars is not None:
        return len(chosen) <= max_response_chars and len(rejected) <= max_response_chars
    return True


def build_from_pair_columns(
    df: pd.DataFrame,
    prompt_key: str,
    chosen_key: str,
    rejected_key: str,
    max_prompt_chars: int | None,
    max_response_chars: int | None,
    max_examples: int | None,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for row in df.itertuples(index=False):
        item = row._asdict()
        prompt = stringify_message_value(item[prompt_key]).strip()
        chosen = stringify_message_value(item[chosen_key]).strip()
        rejected = stringify_message_value(item[rejected_key]).strip()
        if not prompt or not chosen or not rejected or chosen == rejected:
            continue
        if not within_length_limits(prompt, chosen, rejected, max_prompt_chars, max_response_chars):
            continue
        records.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        if max_examples is not None and len(records) >= max_examples:
            break
    return records


def build_from_scored_generations(
    df: pd.DataFrame,
    group_key: str,
    response_key: str,
    score_key: str,
    min_score_gap: float,
    allow_equal_score: bool,
    max_prompt_chars: int | None,
    max_response_chars: int | None,
    max_examples: int | None,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    df = df.copy()
    df["_dpo_group"] = df[group_key].map(stringify_message_value)
    for _, group in df.groupby("_dpo_group", sort=False):
        if len(group) < 2:
            continue

        group = group.copy()
        group["_dpo_score"] = group[score_key].map(score_value)
        group = group.sort_values("_dpo_score", ascending=False)
        best = group.iloc[0].to_dict()
        worst = group.iloc[-1].to_dict()

        score_gap = float(best["_dpo_score"]) - float(worst["_dpo_score"])
        if score_gap < min_score_gap:
            continue
        if score_gap == 0 and not allow_equal_score:
            continue

        prompt = stringify_message_value(best["_dpo_group"]).strip()
        chosen = stringify_message_value(best[response_key]).strip()
        rejected = stringify_message_value(worst[response_key]).strip()
        if not prompt or not chosen or not rejected or chosen == rejected:
            continue
        if not within_length_limits(prompt, chosen, rejected, max_prompt_chars, max_response_chars):
            continue

        records.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        if max_examples is not None and len(records) >= max_examples:
            break
    return records


def write_jsonl(records: list[dict[str, str]], output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    df = read_table(args.input_path, args.input_format)

    columns = tuple(df.columns)
    prompt_key = args.prompt_key or first_existing(columns, PROMPT_CANDIDATES, "prompt")
    chosen_key = args.chosen_key
    rejected_key = args.rejected_key

    if chosen_key is None:
        chosen_key = next((column for column in CHOSEN_CANDIDATES if column in columns), None)
    if rejected_key is None:
        rejected_key = next((column for column in REJECTED_CANDIDATES if column in columns), None)

    if chosen_key is not None and rejected_key is not None:
        records = build_from_pair_columns(
            df,
            prompt_key,
            chosen_key,
            rejected_key,
            args.max_prompt_chars,
            args.max_response_chars,
            args.max_examples,
        )
        mode = "pair-columns"
    else:
        group_key = args.group_key or prompt_key
        response_key = args.response_key or first_existing(columns, RESPONSE_CANDIDATES, "response")
        score_key = args.score_key or first_existing(columns, SCORE_CANDIDATES, "score")
        records = build_from_scored_generations(
            df,
            group_key,
            response_key,
            score_key,
            args.min_score_gap,
            args.allow_equal_score,
            args.max_prompt_chars,
            args.max_response_chars,
            args.max_examples,
        )
        mode = "scored-generations"

    if not records:
        raise RuntimeError("No DPO pairs were produced.")

    write_jsonl(records, args.output_path, args.overwrite)
    print(f"Wrote {len(records)} DPO pairs to {args.output_path} ({mode})")


if __name__ == "__main__":
    main()
