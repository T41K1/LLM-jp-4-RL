#!/usr/bin/env bash
#PBS -P gcg51557
#PBS -q rt_HF
#PBS -q R9920261000
#PBS -N 0316_llm-jp-4-dolci-dpo-multinode
#PBS -l select=4
#PBS -l walltime=50:00:00
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
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
export TRAIN_GLOBAL_BATCH_SIZE_PER_NODE="${TRAIN_GLOBAL_BATCH_SIZE_PER_NODE:-8}"
export VAL_GLOBAL_BATCH_SIZE_PER_NODE="${VAL_GLOBAL_BATCH_SIZE_PER_NODE:-8}"
export TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-1}"
export VAL_MICRO_BATCH_SIZE="${VAL_MICRO_BATCH_SIZE:-1}"
export CLEAR_CACHE_EVERY_N_STEPS="${CLEAR_CACHE_EVERY_N_STEPS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"
export EXP_NAME="${EXP_NAME:-llm-jp-4-8b-nemo-rl-dolci-think-dpo-7b-multinode-seq${MAX_TOTAL_SEQUENCE_LENGTH}-tp${TENSOR_PARALLEL_SIZE}-$(date +%Y%m%d)}"

exec bash exp/dpo/run_llmjp-4-8b_nemo-rl_dpo_multinode.sh "$@"
