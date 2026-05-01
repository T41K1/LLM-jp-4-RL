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


GPUS_PER_NODE=8

# --- ノード情報取得 ---
mapfile -t NODES < <(awk '!seen[$0]++' "$PBS_NODEFILE")
NNODES=${#NODES[@]}
HEAD_NODE=$(hostname -s)
# ジョブのヘッドノード（自分自身）のIPを取得
HEAD_IP=$(hostname -i | awk '{print $1}')
RAY_PORT=6379

echo "[INFO] NNODES=${NNODES}, HEAD=${HEAD_NODE} (${HEAD_IP}), WORKERS=${NODES[*]:1}"

# ワーカーノードで実行する共通セットアップ

WORKER_SETUP="source /etc/profile.d/modules.sh && module reset && module load cuda/12.2/12.2.2 && unset CUDA_VISIBLE_DEVICES && export WANDB_ENTITY=${WANDB_ENTITY} && export RAY_TMPDIR=${RAY_TMPDIR} && mkdir -p ${RAY_TMPDIR} && cd ${PBS_O_WORKDIR} && source .venv/bin/activate"
WORKER_PIDS=()

cleanup() {
    set +e
    trap - EXIT
    echo "[INFO] Stopping Ray cluster..."

    local stop_pids=()
    for ((i = 1; i < NNODES; i++)); do
        pbsdsh -n "${i}" -- bash -lc "${WORKER_SETUP} && ray stop --force 2>/dev/null || true" &
        stop_pids+=("$!")
    done
    for pid in "${stop_pids[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done

    ray stop --force 2>/dev/null || true

    for pid in "${WORKER_PIDS[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done
    echo "[INFO] Ray cleanup done."
}

wait_for_ray_cluster() {
    local expected_nodes=$1
    local expected_gpus=$2

    python3 - "${expected_nodes}" "${expected_gpus}" <<'PY'
import sys
import time

import ray

expected_nodes = int(sys.argv[1])
expected_gpus = float(sys.argv[2])
deadline = time.time() + 300

ray.init(address="auto")
try:
    while time.time() < deadline:
        nodes = ray.nodes()
        alive = [node for node in nodes if node.get("Alive")]
        gpus = sum(node.get("Resources", {}).get("GPU", 0.0) for node in alive)
        print(
            f"[INFO] Ray alive nodes={len(alive)}/{expected_nodes}, "
            f"GPUs={gpus}/{expected_gpus}",
            flush=True,
        )
        if len(alive) >= expected_nodes and gpus >= expected_gpus:
            sys.exit(0)
        time.sleep(5)

    print("[ERROR] Ray cluster did not reach the expected size.", file=sys.stderr)
    for node in ray.nodes():
        print(
            {
                "Alive": node.get("Alive"),
                "NodeManagerAddress": node.get("NodeManagerAddress"),
                "NodeManagerHostname": node.get("NodeManagerHostname"),
                "Resources": node.get("Resources"),
                "DeathReasonMessage": node.get("DeathReasonMessage"),
            },
            file=sys.stderr,
        )
    sys.exit(1)
finally:
    ray.shutdown()
PY
}

trap cleanup EXIT

# --- 既存Rayプロセスを掃除 ---
ray stop --force 2>/dev/null || true
rm -rf "${RAY_TMPDIR}" /tmp/ray 2>/dev/null || true
mkdir -p "${RAY_TMPDIR}"
cleanup_pids=()
for ((i = 1; i < NNODES; i++)); do
    pbsdsh -n "${i}" -- bash -lc "${WORKER_SETUP} && ray stop --force 2>/dev/null || true; rm -rf ${RAY_TMPDIR} /tmp/ray 2>/dev/null; mkdir -p ${RAY_TMPDIR}" &
    cleanup_pids+=("$!")
done
for pid in "${cleanup_pids[@]}"; do
    wait "${pid}"
done
sleep 2

# --- ヘッドノードでRay起動 ---
ray start --head --port=${RAY_PORT} --num-gpus=${GPUS_PER_NODE} --disable-usage-stats
sleep 5

# --- ワーカーノードをRayクラスタに接続 ---
for ((i = 1; i < NNODES; i++)); do
    echo "[INFO] Starting Ray worker on vnode ${i}..."
    pbsdsh -n "${i}" -- bash -lc "${WORKER_SETUP} && ray start --address=${HEAD_IP}:${RAY_PORT} --num-gpus=${GPUS_PER_NODE} --disable-usage-stats --block" &
    WORKER_PIDS+=("$!")
    sleep 3
done

# --- Rayクラスタの接続確認 ---
export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
wait_for_ray_cluster "${NNODES}" "$((NNODES * GPUS_PER_NODE))"
ray status

# --- 学習設定 ---
MODEL_PATH=model/llm-jp-4-8b-thinking

project_name='0316_llm-jp-4-RL'
exp_name="${MODEL_PATH}-GRPO-Olmo3-Math-${NNODES}node-$(date +%Y%m%d)"

VAL_DUMP_DIR="outputs/val/${exp_name}"
mkdir -p "${VAL_DUMP_DIR}"

# --- 学習実行 ---
python3 -m verl.trainer.main_ppo \
    +ray_kwargs.ray_init.address="${HEAD_IP}:${RAY_PORT}" \
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

echo "[INFO] Training finished."
