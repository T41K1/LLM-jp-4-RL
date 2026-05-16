#!/usr/bin/env bash
#PBS -P gcg51557
#PBS -v RTYPE=rt_HF
#PBS -q R9920261000
#PBS -N 0316_llmjp4-convert-hf
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o logs/
#PBS -e logs/




# Convert a NeMo-RL Torch DCP policy checkpoint to a Hugging Face checkpoint.
#
# Usage examples:
#   bash scripts/convert_nemo_rl_dcp_to_hf.sh \
#     checkpoints/nemo-rl-dpo/llm-jp-4-8b-nemo-rl-dolci-think-dpo-7b-multinode-seq12288-tp1-20260513/step_1154
#
#   bash scripts/convert_nemo_rl_dcp_to_hf.sh --force -o dpo-step1154 \
#     checkpoints/nemo-rl-dpo/EXP/step_1154/policy/weights
#
# Default output name:
#   <run-directory-name>-<step-directory-name>
#   e.g. llm-jp-4-8b-nemo-rl-dolci-think-dpo-7b-multinode-seq12288-tp1-20260513-step_1154
#
# Environment:
#   HF_CHECKPOINT_ROOT=hf_checkpoint   Output root directory.
#   NEMO_RL_DIR=deps/nemo-rl           NeMo-RL checkout.
#   MAX_SHARD_SIZE=5GB                 HF shard size when saving safetensors.
#   KEEP_INTERMEDIATE=false            Keep temporary pytorch_model.bin export.
#   INPUT_PATH=/path/to/step_1154      Input checkpoint path for qsub -v.
#   OUTPUT_NAME=name                   Optional output directory name for qsub -v.
#   FORCE=false                        Overwrite output when true.
#   SAVE_SAFETENSORS=true              Set false to keep pytorch_model.bin only.
#   TRUST_REMOTE_CODE=false            Trust custom HF code when rewriting safetensors.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/convert_nemo_rl_dcp_to_hf.sh [options] <nemo-rl-checkpoint-path>

Input path can be one of:
  - step directory:    .../step_1154
  - policy directory:  .../step_1154/policy
  - weights directory: .../step_1154/policy/weights
  - run directory:     .../experiment_name  (latest step_* is used)

Options:
  -o, --output NAME   Output directory name under HF_CHECKPOINT_ROOT.
                      Default: <run-directory-name>-<step-directory-name>
  --bin-only          Stop after creating a Hugging Face pytorch_model.bin checkpoint.
  --force             Remove an existing output directory before converting.
  -h, --help          Show this help.

Environment:
  HF_CHECKPOINT_ROOT  Output root. Default: hf_checkpoint
  NEMO_RL_DIR         NeMo-RL checkout. Default: deps/nemo-rl
  MAX_SHARD_SIZE      Shard size for safetensors. Default: 5GB
  KEEP_INTERMEDIATE   Keep temporary bin export when safetensors is enabled. Default: false
  INPUT_PATH          Input checkpoint path. Useful with qsub -v.
  OUTPUT_NAME         Optional output directory name. Useful with qsub -v.
  FORCE               Overwrite output when true. Default: false
  SAVE_SAFETENSORS    Save safetensors when true. Default: true
  TRUST_REMOTE_CODE   Trust custom HF code when rewriting safetensors. Default: false
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

INPUT_PATH="${INPUT_PATH:-}"
OUTPUT_NAME="${OUTPUT_NAME:-}"
SAVE_SAFETENSORS="${SAVE_SAFETENSORS:-true}"
FORCE="${FORCE:-false}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output)
            [[ $# -ge 2 ]] || die "$1 requires a value"
            OUTPUT_NAME="$2"
            shift 2
            ;;
        --bin-only)
            SAVE_SAFETENSORS=false
            shift
            ;;
        --force)
            FORCE=true
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

HF_CHECKPOINT_ROOT="${HF_CHECKPOINT_ROOT:-hf_checkpoint}"
NEMO_RL_DIR="${NEMO_RL_DIR:-deps/nemo-rl}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-5GB}"
KEEP_INTERMEDIATE="${KEEP_INTERMEDIATE:-false}"

[[ -d "${NEMO_RL_DIR}" ]] || die "NEMO_RL_DIR does not exist: ${NEMO_RL_DIR}"
CONVERTER="${NEMO_RL_DIR}/examples/converters/convert_dcp_to_hf.py"
[[ -f "${CONVERTER}" ]] || die "Converter does not exist: ${CONVERTER}"

INPUT_PATH="${INPUT_PATH%/}"
[[ -d "${INPUT_PATH}" ]] || die "Input path is not a directory: ${INPUT_PATH}"

STEP_DIR=""
POLICY_DIR=""
WEIGHTS_DIR=""

if [[ -d "${INPUT_PATH}/policy/weights" && -f "${INPUT_PATH}/config.yaml" ]]; then
    STEP_DIR="${INPUT_PATH}"
    POLICY_DIR="${STEP_DIR}/policy"
    WEIGHTS_DIR="${POLICY_DIR}/weights"
elif [[ -d "${INPUT_PATH}/weights" && -f "$(dirname "${INPUT_PATH}")/config.yaml" ]]; then
    POLICY_DIR="${INPUT_PATH}"
    STEP_DIR="$(dirname "${POLICY_DIR}")"
    WEIGHTS_DIR="${POLICY_DIR}/weights"
elif [[ -f "${INPUT_PATH}/.metadata" ]]; then
    WEIGHTS_DIR="${INPUT_PATH}"
    POLICY_DIR="$(dirname "${WEIGHTS_DIR}")"
    STEP_DIR="$(dirname "${POLICY_DIR}")"
elif compgen -G "${INPUT_PATH}/step_*" >/dev/null; then
    STEP_DIR="$(find "${INPUT_PATH}" -maxdepth 1 -type d -name 'step_*' | sort -V | tail -n 1)"
    POLICY_DIR="${STEP_DIR}/policy"
    WEIGHTS_DIR="${POLICY_DIR}/weights"
else
    die "Could not infer checkpoint layout from: ${INPUT_PATH}"
fi

CONFIG_PATH="${STEP_DIR}/config.yaml"
TOKENIZER_DIR="${POLICY_DIR}/tokenizer"

[[ -f "${CONFIG_PATH}" ]] || die "config.yaml not found: ${CONFIG_PATH}"
[[ -d "${WEIGHTS_DIR}" ]] || die "weights directory not found: ${WEIGHTS_DIR}"
[[ -f "${WEIGHTS_DIR}/.metadata" ]] || die "DCP metadata not found: ${WEIGHTS_DIR}/.metadata"

if [[ -z "${OUTPUT_NAME}" ]]; then
    RUN_NAME="$(basename "$(dirname "${STEP_DIR}")")"
    STEP_NAME="$(basename "${STEP_DIR}")"
    OUTPUT_NAME="${RUN_NAME}-${STEP_NAME}"
fi

mkdir -p "${HF_CHECKPOINT_ROOT}"

FINAL_DIR="${HF_CHECKPOINT_ROOT%/}/${OUTPUT_NAME}"
if bool_true "${SAVE_SAFETENSORS}"; then
    BIN_DIR="${HF_CHECKPOINT_ROOT%/}/.tmp-${OUTPUT_NAME}-pytorch-bin"
else
    BIN_DIR="${FINAL_DIR}"
fi

if [[ -e "${FINAL_DIR}" || -e "${BIN_DIR}" ]]; then
    if bool_true "${FORCE}"; then
        rm -rf "${FINAL_DIR}" "${BIN_DIR}"
    else
        die "Output already exists. Use --force to overwrite: ${FINAL_DIR}"
    fi
fi

echo "[INFO] Config: ${CONFIG_PATH}"
echo "[INFO] DCP weights: ${WEIGHTS_DIR}"
echo "[INFO] Output: ${FINAL_DIR}"
echo "[INFO] Safetensors: ${SAVE_SAFETENSORS}"

export PYTHONPATH="${NEMO_RL_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

uv run python "${CONVERTER}" \
    --config "${CONFIG_PATH}" \
    --dcp-ckpt-path "${WEIGHTS_DIR}" \
    --hf-ckpt-path "${BIN_DIR}"

if [[ -d "${TOKENIZER_DIR}" ]]; then
    echo "[INFO] Copying checkpoint tokenizer: ${TOKENIZER_DIR}"
    cp -a "${TOKENIZER_DIR}/." "${BIN_DIR}/"
fi

if bool_true "${SAVE_SAFETENSORS}"; then
    export HF_BIN_DIR="${BIN_DIR}"
    export HF_FINAL_DIR="${FINAL_DIR}"
    export MAX_SHARD_SIZE
    export TRUST_REMOTE_CODE
    uv run python - <<'PY'
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

src = os.environ["HF_BIN_DIR"]
dst = os.environ["HF_FINAL_DIR"]
max_shard_size = os.environ.get("MAX_SHARD_SIZE", "5GB")
trust_remote_code = os.environ.get("TRUST_REMOTE_CODE", "").lower() in {"1", "true", "yes"}

model = AutoModelForCausalLM.from_pretrained(
    src,
    torch_dtype="auto",
    trust_remote_code=trust_remote_code,
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=trust_remote_code)

model.save_pretrained(
    dst,
    safe_serialization=True,
    max_shard_size=max_shard_size,
)
tokenizer.save_pretrained(dst)
PY

    if ! bool_true "${KEEP_INTERMEDIATE}"; then
        rm -rf "${BIN_DIR}"
    else
        echo "[INFO] Kept intermediate bin checkpoint: ${BIN_DIR}"
    fi
fi

echo "[INFO] HF checkpoint ready: ${FINAL_DIR}"
