#!/usr/bin/env bash
#PBS -P gcg51557
#PBS -v RTYPE=rt_HF
#PBS -q R9920261000
#PBS -N 0316_llmjp4-eval-AIME24-25-8b
#PBS -l select=1
#PBS -l walltime=500:00:00
#PBS -j oe
#PBS -o logs/
#PBS -e logs/


# Merge a verl FSDP checkpoint into a Hugging Face checkpoint.
#
# Usage examples:
#   bash scripts/merge_verl_fsdp_to_hf.sh \
#     checkpoints/0316_llm-jp-4-rl/EXP/global_step_100
#
#   bash scripts/merge_verl_fsdp_to_hf.sh --force \
#     checkpoints/0316_llm-jp-4-rl/EXP/global_step_100/actor
#
# Default output name:
#   <run-directory-name>-<global-step-directory-name>
#   e.g. llmjp4-grpo-val-global_step_360
#
# Environment:
#   INPUT_PATH=/path/to/global_step_100      Input path for qsub -v.
#   HF_CHECKPOINT_ROOT=hf_checkpoint         Output root directory.
#   OUTPUT_NAME=name                         Optional output directory name for qsub -v.
#   FORCE=false                              Overwrite output when true.
#   TRUST_REMOTE_CODE=true                   Pass --trust-remote-code to verl merger.
#   USE_CPU_INITIALIZATION=false             Pass --use_cpu_initialization to verl merger.
#   DRY_RUN=false                            Print inferred paths and command without merging.
#   SOURCE_MODEL_PATH=/path/to/base_model    Optional source for custom tokenizer/model .py files.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/merge_verl_fsdp_to_hf.sh [options] <verl-checkpoint-path>

Input path can be one of:
  - actor directory:       .../global_step_100/actor
  - global step directory: .../global_step_100
  - run directory:         .../experiment_name  (latest global_step_* is used)

Options:
  -o, --output NAME        Output directory name under HF_CHECKPOINT_ROOT.
                           Default: <run-directory-name>-<global-step-directory-name>
  --force                  Remove an existing output directory before merging.
  --trust-remote-code      Pass --trust-remote-code to verl model_merger. Default.
  --no-trust-remote-code   Do not pass --trust-remote-code.
  --use-cpu-initialization Pass --use_cpu_initialization to verl model_merger.
  --dry-run                Print inferred paths and command without merging.
  -h, --help               Show this help.

Environment:
  INPUT_PATH               Input checkpoint path. Useful with qsub -v.
  HF_CHECKPOINT_ROOT       Output root. Default: hf_checkpoint
  OUTPUT_NAME              Optional output directory name. Useful with qsub -v.
  FORCE                    Overwrite output when true. Default: false
  TRUST_REMOTE_CODE        Trust custom HF code when true. Default: true
  USE_CPU_INITIALIZATION   Use CPU initialization when true. Default: false
  DRY_RUN                  Print command without merging when true. Default: false
  SOURCE_MODEL_PATH        Optional source for custom tokenizer/model .py files.
EOF
}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

bool_true() {
    case "${1:-}" in
        true|True|TRUE|1|yes|Yes|YES) return 0 ;;
        *) return 1 ;;
    esac
}

INPUT_PATH="${INPUT_PATH:-${LOCAL_DIR:-}}"
OUTPUT_NAME="${OUTPUT_NAME:-}"
HF_CHECKPOINT_ROOT="${HF_CHECKPOINT_ROOT:-hf_checkpoint}"
FORCE="${FORCE:-false}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
USE_CPU_INITIALIZATION="${USE_CPU_INITIALIZATION:-false}"
DRY_RUN="${DRY_RUN:-false}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            OUTPUT_NAME="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --trust-remote-code)
            TRUST_REMOTE_CODE=true
            shift
            ;;
        --no-trust-remote-code)
            TRUST_REMOTE_CODE=false
            shift
            ;;
        --use-cpu-initialization)
            USE_CPU_INITIALIZATION=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            die "Unknown option: $1"
            ;;
        *)
            if [[ -z "${INPUT_PATH}" ]]; then
                INPUT_PATH="$1"
            elif [[ -z "${OUTPUT_NAME}" ]]; then
                OUTPUT_NAME="$1"
            else
                die "Unexpected positional argument: $1"
            fi
            shift
            ;;
    esac
done

[[ -n "${INPUT_PATH}" ]] || {
    usage >&2
    exit 2
}

cd "${PBS_O_WORKDIR:-$(pwd)}"
export PATH="${HOME}/.local/bin:${PATH}"

INPUT_PATH="${INPUT_PATH%/}"
[[ -d "${INPUT_PATH}" ]] || die "Input path is not a directory: ${INPUT_PATH}"

ACTOR_DIR=""
GLOBAL_STEP_DIR=""

if [[ -f "${INPUT_PATH}/fsdp_config.json" ]]; then
    ACTOR_DIR="${INPUT_PATH}"
    GLOBAL_STEP_DIR="$(dirname "${ACTOR_DIR}")"
elif [[ -d "${INPUT_PATH}/actor" && -f "${INPUT_PATH}/actor/fsdp_config.json" ]]; then
    GLOBAL_STEP_DIR="${INPUT_PATH}"
    ACTOR_DIR="${GLOBAL_STEP_DIR}/actor"
elif compgen -G "${INPUT_PATH}/global_step_*" >/dev/null; then
    GLOBAL_STEP_DIR="$(find "${INPUT_PATH}" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)"
    ACTOR_DIR="${GLOBAL_STEP_DIR}/actor"
else
    die "Could not infer verl FSDP checkpoint layout from: ${INPUT_PATH}"
fi

RUN_DIR="$(dirname "${GLOBAL_STEP_DIR}")"
RUN_NAME="$(basename "${RUN_DIR}")"
STEP_NAME="$(basename "${GLOBAL_STEP_DIR}")"

[[ -d "${ACTOR_DIR}" ]] || die "actor directory not found: ${ACTOR_DIR}"
[[ -f "${ACTOR_DIR}/fsdp_config.json" ]] || die "fsdp_config.json not found: ${ACTOR_DIR}/fsdp_config.json"
[[ -d "${ACTOR_DIR}/huggingface" ]] || die "huggingface config/tokenizer directory not found: ${ACTOR_DIR}/huggingface"
compgen -G "${ACTOR_DIR}/model_world_size_*_rank_0.pt" >/dev/null || die "rank 0 model shard not found under: ${ACTOR_DIR}"

if [[ -z "${OUTPUT_NAME}" ]]; then
    OUTPUT_NAME="${RUN_NAME}-${STEP_NAME}"
fi

mkdir -p "${HF_CHECKPOINT_ROOT}"
FINAL_DIR="${HF_CHECKPOINT_ROOT%/}/${OUTPUT_NAME}"

if [[ -e "${FINAL_DIR}" ]]; then
    if bool_true "${FORCE}"; then
        rm -rf "${FINAL_DIR}"
    else
        die "Output already exists. Use --force to overwrite: ${FINAL_DIR}"
    fi
fi

MERGE_ARGS=(
    -m verl.model_merger
    merge
    --backend fsdp
    --local_dir "${ACTOR_DIR}"
    --target_dir "${FINAL_DIR}"
)

if bool_true "${TRUST_REMOTE_CODE}"; then
    MERGE_ARGS+=(--trust-remote-code)
fi

if bool_true "${USE_CPU_INITIALIZATION}"; then
    MERGE_ARGS+=(--use_cpu_initialization)
fi

echo "[INFO] Actor checkpoint: ${ACTOR_DIR}"
echo "[INFO] Output: ${FINAL_DIR}"
echo "[INFO] Trust remote code: ${TRUST_REMOTE_CODE}"

printf '[INFO] Command: uv run python'
printf ' %q' "${MERGE_ARGS[@]}"
printf '\n'

if bool_true "${DRY_RUN}"; then
    echo "[INFO] Dry run complete."
    exit 0
fi

# --- マージ前ステージング ---
# verl.model_merger は --trust-remote-code 時、入力 actor/huggingface から tokenizer を
# 実際にロードする (hf_model_config_path = local_dir/huggingface, CLI 上書き不可)。
# llm-jp-4 系は tokenizer が独自実装で、FSDP checkpoint には .py が同梱されないため、
# ここで SOURCE_MODEL_PATH から auto_map 参照の .py を入力側へ補完しておく。
# これをしないと merger が tokenizer ロードで失敗する。
if bool_true "${TRUST_REMOTE_CODE}"; then
    STAGE_SOURCE="${SOURCE_MODEL_PATH:-${BASE_MODEL_PATH:-}}"
    if [[ -n "${STAGE_SOURCE}" && -d "${STAGE_SOURCE}" ]]; then
        STAGE_DST="${ACTOR_DIR}/huggingface" STAGE_SRC="${STAGE_SOURCE}" uv run python - <<'PY'
import json
import os
import shutil
from pathlib import Path

dst = Path(os.environ["STAGE_DST"])
src = Path(os.environ["STAGE_SRC"])


def auto_map_modules(cfg_path):
    if not cfg_path.exists():
        return
    data = json.load(cfg_path.open(encoding="utf-8"))
    for value in data.get("auto_map", {}).values():
        vals = value if isinstance(value, list) else [value]
        for v in vals:
            if isinstance(v, str):
                module = v.rsplit(".", 1)[0].split("--")[-1]
                if module:
                    yield Path(*module.split(".")).with_suffix(".py")


needed = set(auto_map_modules(dst / "config.json"))
needed.update(auto_map_modules(dst / "tokenizer_config.json"))

for rel in sorted(needed):
    if (dst / rel).exists():
        continue
    cand = src / rel
    if not cand.exists():
        print(f"[WARN] custom code referenced by auto_map not found in source: {cand}")
        continue
    (dst / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cand, dst / rel)
    print(f"[INFO] Staged custom code into input: {cand} -> {dst / rel}")
    # 同ディレクトリの兄弟 .py も合わせて配置
    for sib in cand.parent.glob("*.py"):
        if not (dst / sib.name).exists():
            shutil.copy2(sib, dst / sib.name)
            print(f"[INFO] Staged sibling custom code into input: {sib} -> {dst / sib.name}")
PY
    else
        echo "[WARN] SOURCE_MODEL_PATH not usable for staging (='${STAGE_SOURCE}'); merger may fail on custom tokenizer"
    fi
fi

uv run python "${MERGE_ARGS[@]}"

if [[ -d "${ACTOR_DIR}/huggingface" ]]; then
    echo "[INFO] Copying tokenizer/config assets from: ${ACTOR_DIR}/huggingface"
    cp -an "${ACTOR_DIR}/huggingface/." "${FINAL_DIR}/"
fi

export HF_FINAL_DIR="${FINAL_DIR}"
export HF_ACTOR_DIR="${ACTOR_DIR}"
export SOURCE_MODEL_PATH="${SOURCE_MODEL_PATH:-${BASE_MODEL_PATH:-}}"
uv run python - <<'PY'
import json
import os
import shutil
from pathlib import Path

dst = Path(os.environ["HF_FINAL_DIR"])
actor_hf = Path(os.environ["HF_ACTOR_DIR"]) / "huggingface"

roots = []
for raw in (os.environ.get("SOURCE_MODEL_PATH"), str(actor_hf), "model", "hf_checkpoint"):
    if raw:
        path = Path(raw)
        if path.exists():
            roots.append(path)


def iter_auto_map_refs(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    auto_map = data.get("auto_map", {})
    values = []
    for value in auto_map.values():
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(v for v in value if isinstance(v, str))
    for value in values:
        module = value.rsplit(".", 1)[0].split("--")[-1]
        if module:
            yield Path(*module.split(".")).with_suffix(".py")


needed = set(iter_auto_map_refs(dst / "config.json") or [])
needed.update(iter_auto_map_refs(dst / "tokenizer_config.json") or [])

for rel_path in sorted(needed):
    target = dst / rel_path
    if target.exists():
        continue

    matches = []
    for root in roots:
        direct = root / rel_path
        if direct.exists():
            matches = [direct]
            break

    unique_matches = []
    seen = set()
    for match in matches:
        resolved = match.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_matches.append(match)

    if len(unique_matches) == 0:
        raise FileNotFoundError(
            f"Custom code file referenced by auto_map was not found: {rel_path}. "
            "Set SOURCE_MODEL_PATH to the original HF model directory."
        )
    if len(unique_matches) > 1:
        print(f"[WARN] Multiple candidates for {rel_path}; using {unique_matches[0]}")

    source = unique_matches[0]
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(f"[INFO] Copied custom code file: {source} -> {target}")

    for sibling in source.parent.glob("*.py"):
        sibling_target = target.parent / sibling.name
        if not sibling_target.exists():
            shutil.copy2(sibling, sibling_target)
            print(f"[INFO] Copied sibling custom code file: {sibling} -> {sibling_target}")
PY

echo "[INFO] HF checkpoint ready: ${FINAL_DIR}"
