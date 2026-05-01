#!/bin/sh
#PBS -P gcg51557
#PBS -q rt_HF
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
cd ${PBS_O_WORKDIR:-$(pwd)}
echo "Current directory: $(pwd)"

module reset
module load cuda/12.2/12.2.2
unset CUDA_VISIBLE_DEVICES

export VLLM_USE_V1=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
source .venv/bin/activate

source ./.env
echo "WANDB_API_KEY: ${WANDB_API_KEY:0:6}..."

GPUS_PER_NODE=8

# --- ノード情報取得 ---
NODES=($(cat $PBS_NODEFILE | sort -u))
NNODES=${#NODES[@]}
HEAD_NODE=${NODES[0]}
# ジョブのヘッドノード（自分自身）のIPを取得
HEAD_IP=$(hostname -i | awk '{print $1}')
RAY_PORT=6379

echo "[INFO] NNODES=${NNODES}, HEAD=${HEAD_NODE} (${HEAD_IP}), WORKERS=${NODES[*]:1}"

# ワーカーノードで実行する共通セットアップ
WORKER_SETUP="source /etc/profile.d/modules.sh && module reset && module load cuda/12.2/12.2.2 && unset CUDA_VISIBLE_DEVICES && cd ${PBS_O_WORKDIR} && source .venv/bin/activate"

# --- 既存Rayプロセスを掃除 ---
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray 2>/dev/null || true
for i in $(seq 1 $((NNODES - 1))); do
    pbsdsh -n ${i} -- bash -c "${WORKER_SETUP} && ray stop --force 2>/dev/null; rm -rf /tmp/ray 2>/dev/null" &
done
wait
sleep 2

# --- ヘッドノードでRay起動 ---
ray start --head --port=${RAY_PORT} --num-gpus=${GPUS_PER_NODE}
sleep 5

# --- ワーカーノードをRayクラスタに接続 ---
for i in $(seq 1 $((NNODES - 1))); do
    echo "[INFO] Starting Ray worker on vnode ${i}..."
    pbsdsh -n ${i} -- bash -c "${WORKER_SETUP} && \
        ray start --address=${HEAD_IP}:${RAY_PORT} --num-gpus=${GPUS_PER_NODE}" &
    sleep 5
done
wait

# --- Rayクラスタの接続確認 ---
export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
sleep 10
ray status
python3 -c "
import ray
ray.init(address='auto')
print('[DEBUG] nodes:', ray.nodes())
print('[DEBUG] available_resources:', ray.available_resources())
print('[DEBUG] cluster_resources:', ray.cluster_resources())
import ray._private.state
print('[DEBUG] available_resources_per_node:', ray._private.state.available_resources_per_node())
ray.shutdown()
"

# --- 学習設定 ---
MODEL_PATH=model/llm-jp-4-8b-thinking

export WANDB_ENTITY=Research_00
project_name='0316_llm-jp-4-RL'
exp_name="${MODEL_PATH}-GRPO-Olmo3-Math-${NNODES}node-$(date +%Y%m%d)"

VAL_DUMP_DIR="outputs/val/${exp_name}"
mkdir -p "${VAL_DUMP_DIR}"

# --- 学習実行 ---
python3 -m verl.trainer.main_ppo \
    ray_kwargs.ray_init.address="${HEAD_IP}:${RAY_PORT}" \
    algorithm.adv_estimator=grpo \
    data.train_files=data/Dolci-Think-RL-7B-math/train.parquet \
    data.val_files='[data/AIME2024/test.parquet,data/AIME2025/test.parquet]' \
    data.train_batch_size=512 \
    data.max_prompt_length=2048 \
    data.max_response_length=32768 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
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

# --- 後片付け ---
echo "[INFO] Training finished. Stopping Ray cluster..."
ray stop --force 2>/dev/null || true
for i in $(seq 1 $((NNODES - 1))); do
    pbsdsh -n ${i} -- bash -c "${WORKER_SETUP} && ray stop --force 2>/dev/null" &
done
wait
echo "[INFO] Done."
