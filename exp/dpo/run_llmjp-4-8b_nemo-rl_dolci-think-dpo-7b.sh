#!/bin/sh
#PBS -P gcg51557
#PBS -q rt_HF
#PBS -q R9920261000
#PBS -N 0316_llm-jp-4-DPO
#PBS -l select=1
#PBS -l walltime=500:00:00
#PBS -j oe
#PBS -o logs/
#PBS -e logs/

set -xeuo pipefail

cd "${PBS_O_WORKDIR:-$(pwd)}"
REPO_DIR="$(pwd)"

export TRAIN_DATA="${TRAIN_DATA:-${REPO_DIR}/data/Dolci-Think-DPO-7B/nemo-rl/train.jsonl}"
export VAL_DATA="${VAL_DATA:-${REPO_DIR}/data/Dolci-Think-DPO-7B/nemo-rl/val.jsonl}"
export PROJECT_NAME="${PROJECT_NAME:-0316_llm-jp-4-dpo}"
export MAX_TOTAL_SEQUENCE_LENGTH="${MAX_TOTAL_SEQUENCE_LENGTH:-12288}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
export TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE:-8}"
export VAL_GLOBAL_BATCH_SIZE="${VAL_GLOBAL_BATCH_SIZE:-8}"
export TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-1}"
export VAL_MICRO_BATCH_SIZE="${VAL_MICRO_BATCH_SIZE:-1}"
export PREFERENCE_AVERAGE_LOG_PROBS="${PREFERENCE_AVERAGE_LOG_PROBS:-true}"
export CLEAR_CACHE_EVERY_N_STEPS="${CLEAR_CACHE_EVERY_N_STEPS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"

jid="${PBS_JOBID:-local}"
jid="${jid%%.*}"
case "${OLMO3_LOSS:-false}" in
    true|True|TRUE|1|yes|Yes|YES)
        loss_tag="olmo3loss"
        ;;
    *)
        case "${PREFERENCE_AVERAGE_LOG_PROBS}" in
            true|True|TRUE|1|yes|Yes|YES) loss_tag="avglogprob" ;;
            *) loss_tag="sumlogprob" ;;
        esac
        ;;
esac
export EXP_NAME="${EXP_NAME:-llm-jp-4-8b-nemo-rl-dolci-think-dpo-7b-seq${MAX_TOTAL_SEQUENCE_LENGTH}-tp${TENSOR_PARALLEL_SIZE}-${loss_tag}-sched${LR_SCHEDULER:-linear}-$(date +%Y%m%d)-${jid}}"

exec bash exp/dpo/run_llmjp-4-8b_nemo-rl_dpo.sh "$@"
