#!/bin/bash
#SBATCH --job-name=0316_llmjp4-olmes-AIME24,25
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# OLMES (https://github.com/allenai/olmes) で AIME 2024 + 2025 を vLLM backend で評価。
# pass@1/8/16/32/64 を一度の n=64 生成から同時算出 (DeepSeek流儀: temp=0.6, top_p=0.95)。
#
# 前提:
#   1. deps/olmes/ に olmes を clone 済み (既に deps/olmes/ が存在)
#   2. .venv-eval を作成し olmes をインストール済み (exp/eval/READMe.md 参照)
#   3. model/llm-jp-4-8b-thinking を HF から取得済み
#
# 投入: sbatch exp/eval/olmes_aime.sh
# 状態: squeue -u $USER

set -xeuo pipefail

echo "Current directory: $(pwd)"
cd ${SLURM_SUBMIT_DIR:-$(pwd)}
echo "Current directory: $(pwd)"

module reset 2>/dev/null || true
module load cuda/12.2/12.2.2 2>/dev/null || true
unset CUDA_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES

export VLLM_USE_V1=1

# olmes 専用 venv (verl/.venv と dep が衝突するため分離)
source .venv-eval/bin/activate

set -a
source ./.env
set +a
echo "HF_TOKEN: ${HF_TOKEN:0:6}..."

MODEL_PATH=${MODEL_PATH:-model/llm-jp-4-8b-thinking}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/eval/olmes-aime-$(date +%Y%m%d-%H%M%S)}
mkdir -p "${OUTPUT_DIR}"

# pass@k 設定: 研究チームは n=32 だが、ここでは n=64 まで欲しいので generation_kwargs.repeats を override。
# metric_kwargs.pass_at_ks で算出する k を指定。
REPEATS=64
PASS_AT_KS='[1,8,16,32,64]'

AIME2024_TASK=$(cat <<EOF
{"task_name":"aime:zs_cot_r1::pass_at_32_2024_deepseek","generation_kwargs":{"repeats":${REPEATS}},"metric_kwargs":{"pass_at_ks":${PASS_AT_KS}}}
EOF
)
AIME2025_TASK=$(cat <<EOF
{"task_name":"aime:zs_cot_r1::pass_at_32_2025_deepseek","generation_kwargs":{"repeats":${REPEATS}},"metric_kwargs":{"pass_at_ks":${PASS_AT_KS}}}
EOF
)

# 本実行前に --dry-run で merged config を確認したい場合は以下を有効化:
# olmes --model "${MODEL_PATH}" --model-type vllm \
#       --task "${AIME2024_TASK}" "${AIME2025_TASK}" --dry-run

olmes \
    --model "${MODEL_PATH}" \
    --model-type vllm \
    --task "${AIME2024_TASK}" "${AIME2025_TASK}" \
    --model-args '{"max_length":32768,"tensor_parallel_size":2,"gpu_memory_utilization":0.85,"trust_remote_code":true}' \
    --batch-size 1 \
    --output-dir "${OUTPUT_DIR}"

echo "Done. Results saved to ${OUTPUT_DIR}"
