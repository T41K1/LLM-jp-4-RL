#!/usr/bin/env bash
#PBS -P gcg51557
#PBS -v RTYPE=rt_HF
#PBS -q R9920261000
#PBS -N 0316_llm-jp-4-RL-multinode
#PBS -l select=2
#PBS -l walltime=500:00:00
#PBS -j oe
#PBS -o logs/
#PBS -e logs/

# マルチノード版 OLMo3踏襲 GRPO
# select=N でノード数を変更可能

set -xeuo pipefail

echo "Current directory: $(pwd)"
cd "${PBS_O_WORKDIR:-$(pwd)}"
echo "Current directory: $(pwd)"

mkdir -p logs
LIVE_LOG="logs/${PBS_JOBID:-manual.$(date +%Y%m%d%H%M%S)}.live.log"
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

export VLLM_USE_V1=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_TMPDIR="${TMPDIR:-/tmp}/ray"
mkdir -p "${RAY_TMPDIR}"
source .venv/bin/activate

source ./.env
export WANDB_ENTITY=Research_00
echo "WANDB_API_KEY: ${WANDB_API_KEY:0:6}..."
echo "WANDB_ENTITY: ${WANDB_ENTITY}"


# --- Rayマルチノードクラスタ起動 ---
GPUS_PER_NODE=8
source exp/prod_model_RL/lib/ray_multinode.sh
trap cleanup_ray_cluster EXIT
setup_ray_cluster

# --- 学習設定 ---
MODEL_PATH=model/llm-jp-4-8b-thinking

# --- GRPO advantage 正規化方式 ---
# ADV_NORM=mean : グループ平均のみ減算 (mean only, Dr.GRPO 系)  -> norm_adv_by_std_in_grpo=False
# ADV_NORM=std  : (r - mean) / (std + eps) (オリジナル GRPO)     -> norm_adv_by_std_in_grpo=True
ADV_NORM="${ADV_NORM:-std}"
case "${ADV_NORM}" in
    mean) NORM_ADV_BY_STD=False ;;
    std)  NORM_ADV_BY_STD=True  ;;
    *) echo "[ERROR] ADV_NORM must be 'mean' or 'std', got '${ADV_NORM}'" >&2; exit 1 ;;
esac
echo "[INFO] GRPO advantage normalization: ADV_NORM=${ADV_NORM} (norm_adv_by_std_in_grpo=${NORM_ADV_BY_STD})"

project_name='0316_llm-jp-4-RL'
exp_name="${EXP_NAME:-${MODEL_PATH##*/}-GRPO-Olmo3-Math-adv${ADV_NORM}-${NNODES}node-$(date +%Y%m%d)}"

VAL_DUMP_DIR="outputs/val/${exp_name}"
mkdir -p "${VAL_DUMP_DIR}"



N=${MY_N}
NUM_PROMPTS=${NUM_PROMPTS}
ENT=1e-3 
LR=1e-6
MINI_BATCH_SIZE=${MINI_BATCH_SIZE}
BS=$((${N} * ${NUM_PROMPTS}))
MBS=$((${N} * ${MINI_BATCH_SIZE}))


# --- 学習実行 ---
python3 -m verl.trainer.main_ppo \
    trainer.val_metrics.pass_at_k=true \
    +ray_kwargs.ray_init.address="${RAY_ADDRESS}" \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=${NORM_ADV_BY_STD} \
    data.train_files=data/Dolci-Think-RL-7B-math/train.parquet \
    data.val_files='[data/AIME2024/test.parquet,data/AIME2025/test.parquet]' \
    data.train_batch_size=${NUM_PROMPTS} \
    data.max_prompt_length=2048 \
    data.max_response_length=32768 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.calculate_entropy=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=${N} \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    reward.custom_reward_function.path=rewards/math_reward.py \
    reward.custom_reward_function.name=compute_score \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.val_before_train=True \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.log_val_generations=60 \
    trainer.validation_data_dir="${VAL_DUMP_DIR}" \
    trainer.total_epochs=15 "$@"

echo "[INFO] Training finished."
