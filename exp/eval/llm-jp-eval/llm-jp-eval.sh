#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash exp/eval/llm-jp-eval/llm-jp-eval.sh <hf-checkpoint-name-or-path>

Examples:
  bash exp/eval/llm-jp-eval/llm-jp-eval.sh \
    llm-jp-4-8b-nemo-rl-dolci-think-dpo-7b-multinode-seq12288-tp1-20260513-step_1154

  bash exp/eval/llm-jp-eval/llm-jp-eval.sh \
    hf_checkpoint/llm-jp-4-8b-thinking-GRPO-Olmo3-Math-2node-20260502-global_step_300

Environment:
  MODEL_DIR              Default: <repo>/hf_checkpoint
  EVAL_ROOT              Default: <repo>/evals
  EVAL_EFFORTS           Default: "low medium"
  EVAL_VERSION           Default: v2.1.3
  EVAL_SAMPLES           Default: 100
  REASONING_PARSER       Default: openai_gptoss
  QSUB_PY                Default: /groups/gcg51557/evaluation2/qsub.py
  HF_HOME                Default: <repo>/.cache/huggingface
  EVAL_EXPERIMENT_DIR    Passed to qsub.py when set
  EVAL_RTYPE             Default: rt_HG
  EVAL_SELECT            Default: 1
  EVAL_PBS_QUEUE         Default: R9920261000
  EVAL_JOB_PREFIX        Default: 0316-llmjp-eval
  EVAL_WANDB             Default: 0. Set to 1 to upload llm-jp-eval results to W&B.
  EVAL_WANDB_ENTITY      Default: WANDB_ENTITY. Required when EVAL_WANDB=1.
  EVAL_WANDB_PROJECT     Default: WANDB_PROJECT or llm-jp-eval.
  EVAL_WANDB_OUTPUT_TABLE Default: 1. Upload output examples table.
  EVAL_WANDB_OUTPUT_TOP_N Default: 5. Set to empty or null to upload all rows.
EOF
}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

is_truthy() {
    case "${1,,}" in
        1 | true | yes | on) return 0 ;;
        *) return 1 ;;
    esac
}

build_exporters_json() {
    python3 - "$@" <<'PY'
import json
import sys

entity, project, output_table, output_top_n = sys.argv[1:5]

output_table_enabled = output_table.lower() in {"1", "true", "yes", "on"}
if output_top_n == "" or output_top_n.lower() == "null":
    output_top_n_value = None
else:
    output_top_n_value = int(output_top_n)

exporters = {
    "local": {
        "filename_format": "result_{run_name}.json",
        "export_output_table": True,
        "output_top_n": 5,
    },
    "wandb": {
        "export_output_table": output_table_enabled,
        "output_top_n": output_top_n_value,
        "entity": entity,
        "project": project,
        "inherit_wandb_run": False,
    },
}
print(json.dumps(exporters, separators=(",", ":")))
PY
}

if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"

MODEL_ARG="$1"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/hf_checkpoint}"
EVAL_ROOT="${EVAL_ROOT:-${REPO_ROOT}/evals}"
EVAL_VERSION="${EVAL_VERSION:-v2.1.3}"
EVAL_SAMPLES="${EVAL_SAMPLES:-100}"
EVAL_EFFORTS="${EVAL_EFFORTS:-low medium}"
REASONING_PARSER="${REASONING_PARSER:-openai_gptoss}"
QSUB_PY="${QSUB_PY:-/groups/gcg51557/evaluation2/qsub.py}"
EVAL_RTYPE="${EVAL_RTYPE:-rt_HG}"
EVAL_SELECT="${EVAL_SELECT:-1}"
EVAL_PBS_QUEUE="${EVAL_PBS_QUEUE:-R9920261000}"
EVAL_JOB_PREFIX="${EVAL_JOB_PREFIX:-0316-llmjp-eval}"
EVAL_WANDB="${EVAL_WANDB:-0}"
EVAL_WANDB_OUTPUT_TABLE="${EVAL_WANDB_OUTPUT_TABLE:-1}"
EVAL_WANDB_OUTPUT_TOP_N="${EVAL_WANDB_OUTPUT_TOP_N:-5}"

if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    set +u
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/.env"
    set -u
    set +a
fi

source ./.env

export WANDB_ENTITY=Research_00
export WANDB_PROJECT=llm-jp-4-rl-llm-jp-eval
export EVAL_WANDB=1


export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"

[[ -f "${QSUB_PY}" ]] || die "qsub.py not found: ${QSUB_PY}"
[[ -n "${HF_TOKEN:-}" ]] || die "HF_TOKEN is not set. Put it in .env or export it before running."

PASS_ENV_TO_PBS=0
if is_truthy "${EVAL_WANDB}"; then
    EVAL_WANDB_ENTITY="${EVAL_WANDB_ENTITY:-${WANDB_ENTITY:-}}"
    EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-${WANDB_PROJECT:-llm-jp-eval}}"

    [[ -n "${WANDB_API_KEY:-}" ]] || die "WANDB_API_KEY is not set. Put it in .env or export it before running with EVAL_WANDB=1."
    [[ -n "${EVAL_WANDB_ENTITY}" ]] || die "EVAL_WANDB_ENTITY or WANDB_ENTITY must be set when EVAL_WANDB=1."
    [[ -n "${EVAL_WANDB_PROJECT}" ]] || die "EVAL_WANDB_PROJECT or WANDB_PROJECT must be set when EVAL_WANDB=1."

    if [[ -z "${EXPORTERS:-}" ]]; then
        EXPORTERS="$(build_exporters_json \
            "${EVAL_WANDB_ENTITY}" \
            "${EVAL_WANDB_PROJECT}" \
            "${EVAL_WANDB_OUTPUT_TABLE}" \
            "${EVAL_WANDB_OUTPUT_TOP_N}")"
        export EXPORTERS
    else
        echo "[INFO] EXPORTERS is already set; using it instead of generated W&B exporters."
    fi

    export WANDB_API_KEY
    PASS_ENV_TO_PBS=1
elif [[ -n "${EXPORTERS:-}" ]]; then
    export EXPORTERS
    PASS_ENV_TO_PBS=1
fi

if [[ "${MODEL_ARG}" = /* ]]; then
    MODEL_PATH="${MODEL_ARG%/}"
elif [[ -d "${REPO_ROOT}/${MODEL_ARG}" ]]; then
    MODEL_PATH="${REPO_ROOT}/${MODEL_ARG%/}"
else
    MODEL_PATH="${MODEL_DIR%/}/${MODEL_ARG%/}"
fi

[[ -d "${MODEL_PATH}" ]] || die "Model directory not found: ${MODEL_PATH}"
[[ -f "${MODEL_PATH}/config.json" ]] || die "config.json not found. Is this an HF checkpoint?: ${MODEL_PATH}"

MODEL_NAME="$(basename "${MODEL_PATH}")"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${EVAL_ROOT%/}/${MODEL_NAME}}"

read -r -a EFFORTS <<< "${EVAL_EFFORTS}"
[[ ${#EFFORTS[@]} -gt 0 ]] || die "EVAL_EFFORTS is empty"

run_eval() {
    local effort="$1"
    local outdir="${EVAL_OUTPUT_DIR}_${effort}"
    local result_json="${outdir}/llm-jp-eval/${EVAL_VERSION}/results/result.json"

    if [[ -f "${result_json}" ]]; then
        echo "[SKIP] Result exists: ${result_json}"
        return 0
    fi

    if [[ -d "${outdir}" ]]; then
        echo "[CLEAN] Removing incomplete dir: ${outdir}"
        rm -rf -- "${outdir}"
    fi

    echo "[RUN ] model=${MODEL_PATH}"
    echo "[RUN ] reasoning_effort=${effort}"
    echo "[RUN ] output=${outdir}"
    if is_truthy "${EVAL_WANDB}"; then
        echo "[RUN ] wandb=${EVAL_WANDB_ENTITY}/${EVAL_WANDB_PROJECT}"
    fi

    local cmd=(
        python3 "${QSUB_PY}" "${MODEL_PATH}"
        "${outdir}"
        --disable-swallow
        --llm-jp-eval-versions "${EVAL_VERSION}"
        --apply-chat-template
        --reasoning-parser "${REASONING_PARSER}"
        --chat-template-args "reasoning_effort=${effort}"
        --llm-jp-eval-max-num-samples "${EVAL_SAMPLES}"
        --job-name "${EVAL_JOB_PREFIX}-${effort}"
        --rtype "${EVAL_RTYPE}"
        --select "${EVAL_SELECT}"
        --pbs-queue "${EVAL_PBS_QUEUE}"
    )

    if [[ "${PASS_ENV_TO_PBS}" -eq 1 ]]; then
        cmd+=(--options "#PBS -V")
    fi

    if [[ -n "${EVAL_EXPERIMENT_DIR:-}" ]]; then
        cmd+=(--experiment-dir "${EVAL_EXPERIMENT_DIR}")
    fi

    "${cmd[@]}"
}

for effort in "${EFFORTS[@]}"; do
    run_eval "${effort}"
done
