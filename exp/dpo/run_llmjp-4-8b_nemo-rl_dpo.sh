#!/usr/bin/env bash
#PBS -P gcg51557
#PBS -q rt_HF
#PBS -q R9920261000
#PBS -N 0316_llm-jp-4-RL-multinode
#PBS -l select=2
#PBS -l walltime=500:00:00
#PBS -j oe
#PBS -o logs/
#PBS -e logs/

set -xeuo pipefail

echo "Current directory: $(pwd)"
cd "${PBS_O_WORKDIR:-$(pwd)}"
echo "Current directory: $(pwd)"

mkdir -p logs checkpoints

LIVE_LOG="logs/${PBS_JOBID:-manual.$(date +%Y%m%d%H%M%S)}.nemo-rl-dpo.log"
export PYTHONUNBUFFERED=1
if command -v stdbuf >/dev/null 2>&1; then
    exec > >(stdbuf -oL -eL tee -a "${LIVE_LOG}") 2>&1
else
    exec > >(tee -a "${LIVE_LOG}") 2>&1
fi
echo "[INFO] Live log: ${LIVE_LOG}"

module reset
module load cuda/12.2/12.2.2
unset CUDA_VISIBLE_DEVICES

if [[ -f ./.env ]]; then
    source ./.env
fi

REPO_DIR="$(pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_DIR}/.tmp/uv-cache}"
mkdir -p "${UV_CACHE_DIR}"
HF_HOME_DEFAULT="${XDG_CACHE_HOME:-${HOME}/.cache}/huggingface"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME:-${HF_HOME_DEFAULT}}/modules}"
mkdir -p "${HF_MODULES_CACHE}"
case ":${PYTHONPATH:-}:" in
    *":${HF_MODULES_CACHE}:"*) ;;
    *) export PYTHONPATH="${HF_MODULES_CACHE}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

# Clone NVIDIA-NeMo/RL separately and point this variable at the checkout:
#   git clone https://github.com/NVIDIA-NeMo/RL.git deps/nemo-rl
#   git -C deps/nemo-rl switch --detach v0.5.0
NEMO_RL_DIR="${NEMO_RL_DIR:-${REPO_DIR}/deps/nemo-rl}"
CONFIG_PATH="${CONFIG_PATH:-examples/configs/dpo.yaml}"

MODEL_PATH="${MODEL_PATH:-${REPO_DIR}/model/llm-jp-4-8b-thinking}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
TRAIN_DATA="${TRAIN_DATA:-${REPO_DIR}/data/dpo/train.jsonl}"
VAL_DATA="${VAL_DATA:-${REPO_DIR}/data/dpo/val.jsonl}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
CLUSTER_NUM_NODES="${CLUSTER_NUM_NODES:-1}"
MAX_TOTAL_SEQUENCE_LENGTH="${MAX_TOTAL_SEQUENCE_LENGTH:-32768}"
TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE:-128}"
TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-1}"
VAL_GLOBAL_BATCH_SIZE="${VAL_GLOBAL_BATCH_SIZE:-8}"
VAL_MICRO_BATCH_SIZE="${VAL_MICRO_BATCH_SIZE:-1}"


OLMo3_Loss="${DPO_Loss:True}"
LR="${LR:-8.0e-8}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
MAX_NUM_EPOCHS="${MAX_NUM_EPOCHS:-1}"
MAX_NUM_STEPS="${MAX_NUM_STEPS:-1500}"
LR_SCHEDULER="${LR_SCHEDULER:-linear}"
WARMUP_START_FACTOR="${WARMUP_START_FACTOR:-0.1}"
LR_END_FACTOR="${LR_END_FACTOR:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
WARMUP_STEPS="${WARMUP_STEPS:-$(awk -v steps="${MAX_NUM_STEPS}" -v ratio="${WARMUP_RATIO}" '
BEGIN {
    if (ratio !~ /^[0-9]+([.][0-9]+)?$/) {
        print "WARMUP_RATIO must be a non-negative number" > "/dev/stderr"
        exit 2
    }
    warmup = int(steps * ratio)
    if (warmup < 1) {
        warmup = 1
    }
    if (warmup > steps) {
        warmup = steps
    }
    print warmup
}
')}"
LR_DECAY_STEPS="${LR_DECAY_STEPS:-$(awk -v steps="${MAX_NUM_STEPS}" -v warmup="${WARMUP_STEPS}" '
BEGIN {
    decay = steps - warmup
    if (decay < 1) {
        decay = 1
    }
    print decay
}
')}"
VAL_PERIOD="${VAL_PERIOD:-25}"
VAL_BATCHES="${VAL_BATCHES:-8}"
SAVE_PERIOD="${SAVE_PERIOD:-50}"
OLMO3_LOSS="${OLMO3_LOSS:-${OLMo3_Loss:-false}}"
case "${OLMO3_LOSS}" in
    true|True|TRUE|1|yes|Yes|YES)
        KL_PENALTY="${KL_PENALTY:-5}"
        PREFERENCE_AVERAGE_LOG_PROBS="${PREFERENCE_AVERAGE_LOG_PROBS:-true}"
        ;;
    false|False|FALSE|0|no|No|NO)
        KL_PENALTY="${KL_PENALTY:-0.05}"
        PREFERENCE_AVERAGE_LOG_PROBS="${PREFERENCE_AVERAGE_LOG_PROBS:-false}"
        ;;
    *)
        echo "[ERROR] Unsupported OLMO3_LOSS=${OLMO3_LOSS}. Use true or false." >&2
        exit 1
        ;;
esac
SFT_LOSS_WEIGHT="${SFT_LOSS_WEIGHT:-0}"
PREFERENCE_LOSS_WEIGHT="${PREFERENCE_LOSS_WEIGHT:-1}"

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-false}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-true}"
CLEAR_CACHE_EVERY_N_STEPS="${CLEAR_CACHE_EVERY_N_STEPS:-null}"
PYTORCH_CUDA_ALLOC_CONF_VALUE="${PYTORCH_CUDA_ALLOC_CONF:-}"

PROMPT_KEY="${PROMPT_KEY:-prompt}"
CHOSEN_KEY="${CHOSEN_KEY:-chosen}"
REJECTED_KEY="${REJECTED_KEY:-rejected}"

project_name="${PROJECT_NAME:-0316_llm-jp-4-dpo}"
jid="${PBS_JOBID:-local}"
jid="${jid%%.*}"
exp_name="${EXP_NAME:-llm-jp-4-8b-nemo-rl-dpo-$(date +%Y%m%d)-${jid}}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs/nemo-rl-dpo}"
CKPT_DIR="${CKPT_DIR:-${REPO_DIR}/checkpoints/nemo-rl-dpo/${exp_name}}"
CHECKPOINT_METRIC="${CHECKPOINT_METRIC:-val:validation-default_loss}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
TENSORBOARD_ENABLED="${TENSORBOARD_ENABLED:-false}"

if [[ ! -d "${NEMO_RL_DIR}" ]]; then
    echo "[ERROR] NEMO_RL_DIR does not exist: ${NEMO_RL_DIR}" >&2
    echo "[ERROR] Clone NVIDIA-NeMo/RL with submodules, or set NEMO_RL_DIR." >&2
    exit 1
fi

if [[ "${CONFIG_PATH}" = /* ]]; then
    CONFIG_FILE="${CONFIG_PATH}"
else
    CONFIG_FILE="${NEMO_RL_DIR}/${CONFIG_PATH}"
fi

for required_path in "${CONFIG_FILE}" "${MODEL_PATH}" "${TRAIN_DATA}" "${VAL_DATA}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "[ERROR] Required path does not exist: ${required_path}" >&2
        exit 1
    fi
done

mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

echo "[INFO] NeMo-RL dir: ${NEMO_RL_DIR}"
echo "[INFO] Config: ${CONFIG_FILE}"
echo "[INFO] Model: ${MODEL_PATH}"
echo "[INFO] Train data: ${TRAIN_DATA}"
echo "[INFO] Val data: ${VAL_DATA}"
echo "[INFO] Checkpoints: ${CKPT_DIR}"
echo "[INFO] LR: ${LR}, scheduler=${LR_SCHEDULER}, warmup_ratio=${WARMUP_RATIO}, warmup_steps=${WARMUP_STEPS}, decay_steps=${LR_DECAY_STEPS}, lr_end_factor=${LR_END_FACTOR}"

scheduler_args=()
case "${LR_SCHEDULER}" in
    constant)
        scheduler_args=(
            "policy.scheduler=[{name: torch.optim.lr_scheduler.LinearLR, kwargs: {start_factor: ${WARMUP_START_FACTOR}, end_factor: 1.0, total_iters: ${WARMUP_STEPS}}}, {name: torch.optim.lr_scheduler.ConstantLR, kwargs: {factor: 1.0, total_iters: 10000000000}}, {milestones: [${WARMUP_STEPS}]}]"
        )
        ;;
    linear)
        scheduler_args=(
            "policy.scheduler=[{name: torch.optim.lr_scheduler.LinearLR, kwargs: {start_factor: ${WARMUP_START_FACTOR}, end_factor: 1.0, total_iters: ${WARMUP_STEPS}}}, {name: torch.optim.lr_scheduler.LinearLR, kwargs: {start_factor: 1.0, end_factor: ${LR_END_FACTOR}, total_iters: ${LR_DECAY_STEPS}}}, {milestones: [${WARMUP_STEPS}]}]"
        )
        ;;
    none|null)
        scheduler_args=("policy.scheduler=null")
        ;;
    *)
        echo "[ERROR] Unsupported LR_SCHEDULER=${LR_SCHEDULER}. Use linear, constant, or none." >&2
        exit 1
        ;;
esac

args=(
    --config "${CONFIG_FILE}"
    "policy.model_name=${MODEL_PATH}"
    "policy.tokenizer.name=${TOKENIZER_PATH}"
    "policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE}"
    "policy.train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE}"
    "policy.max_total_sequence_length=${MAX_TOTAL_SEQUENCE_LENGTH}"
    "policy.dtensor_cfg.enabled=true"
    "policy.dtensor_cfg.activation_checkpointing=${ACTIVATION_CHECKPOINTING}"
    "policy.dtensor_cfg.tensor_parallel_size=${TENSOR_PARALLEL_SIZE}"
    "policy.dtensor_cfg.context_parallel_size=${CONTEXT_PARALLEL_SIZE}"
    "policy.dtensor_cfg.sequence_parallel=${SEQUENCE_PARALLEL}"
    "policy.dtensor_cfg.clear_cache_every_n_steps=${CLEAR_CACHE_EVERY_N_STEPS}"
    "policy.dtensor_cfg.env_vars.PYTORCH_CUDA_ALLOC_CONF='${PYTORCH_CUDA_ALLOC_CONF_VALUE}'"
    "policy.optimizer.kwargs.lr=${LR}"
    "policy.optimizer.kwargs.weight_decay=${WEIGHT_DECAY}"
    "${scheduler_args[@]}"
    "data.dataset_name=BinaryPreferenceDataset"
    "+data.train_data_path=${TRAIN_DATA}"
    "+data.val_data_path=${VAL_DATA}"
    "+data.train_split=null"
    "+data.val_split=null"
    "+data.prompt_key=${PROMPT_KEY}"
    "+data.chosen_key=${CHOSEN_KEY}"
    "+data.rejected_key=${REJECTED_KEY}"
    "data.max_input_seq_length=${MAX_TOTAL_SEQUENCE_LENGTH}"
    "dpo.max_num_epochs=${MAX_NUM_EPOCHS}"
    "dpo.max_num_steps=${MAX_NUM_STEPS}"
    "dpo.val_period=${VAL_PERIOD}"
    "dpo.val_batches=${VAL_BATCHES}"
    "dpo.val_global_batch_size=${VAL_GLOBAL_BATCH_SIZE}"
    "dpo.val_micro_batch_size=${VAL_MICRO_BATCH_SIZE}"
    "dpo.reference_policy_kl_penalty=${KL_PENALTY}"
    "dpo.preference_loss_weight=${PREFERENCE_LOSS_WEIGHT}"
    "dpo.sft_loss_weight=${SFT_LOSS_WEIGHT}"
    "dpo.preference_average_log_probs=${PREFERENCE_AVERAGE_LOG_PROBS}"
    "dpo.sft_average_log_probs=${PREFERENCE_AVERAGE_LOG_PROBS}"
    "checkpointing.checkpoint_dir=${CKPT_DIR}"
    "checkpointing.metric_name=${CHECKPOINT_METRIC}"
    "checkpointing.save_period=${SAVE_PERIOD}"
    "logger.log_dir=${LOG_DIR}"
    "logger.wandb_enabled=${WANDB_ENABLED}"
    "logger.tensorboard_enabled=${TENSORBOARD_ENABLED}"
    "logger.wandb.project=${project_name}"
    "logger.wandb.name=${exp_name}"
    "logger.tensorboard.log_dir=${LOG_DIR}/tensorboard/${exp_name}"
    "cluster.gpus_per_node=${GPUS_PER_NODE}"
    "cluster.num_nodes=${CLUSTER_NUM_NODES}"
)

uv run --extra nemo-rl python "${NEMO_RL_DIR}/examples/run_dpo.py" "${args[@]}" "$@"

echo "[INFO] DPO training finished."
