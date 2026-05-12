#!/usr/bin/env bash
#PBS -P gcg51557
#PBS -q rt_HF
#PBS -q R9920261000
#PBS -N 0316_llm-jp-4-dpo-multinode
#PBS -l select=2
#PBS -l walltime=50:00:00
#PBS -j oe
#PBS -o logs/
#PBS -e logs/

set -xeuo pipefail

echo "Current directory: $(pwd)"
cd "${PBS_O_WORKDIR:-$(pwd)}"
echo "Current directory: $(pwd)"

mkdir -p logs checkpoints

LIVE_LOG="logs/${PBS_JOBID:-manual.$(date +%Y%m%d%H%M%S)}.nemo-rl-dpo-multinode.log"
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
    set +x
    source ./.env
    set -x
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
NEMO_RL_DIR="${NEMO_RL_DIR:-${REPO_DIR}/deps/nemo-rl}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
RAY_PORT="${RAY_PORT:-6379}"

if [[ ! -d "${NEMO_RL_DIR}" ]]; then
    echo "[ERROR] NEMO_RL_DIR does not exist: ${NEMO_RL_DIR}" >&2
    exit 1
fi

if [[ -z "${PBS_NODEFILE:-}" || ! -f "${PBS_NODEFILE}" ]]; then
    echo "[ERROR] PBS_NODEFILE is not available. Submit this script with qsub." >&2
    exit 1
fi

mapfile -t NODES < <(awk '!seen[$0]++' "${PBS_NODEFILE}")
NNODES="${#NODES[@]}"
HEAD_NODE="$(hostname -s)"
HEAD_IP="$(hostname -i | awk '{print $1}')"

echo "[INFO] NNODES=${NNODES}, HEAD=${HEAD_NODE} (${HEAD_IP}), WORKERS=${NODES[*]:1}"

export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
RAY_TMP_JOB_ID="${PBS_JOBID:-manual}"
RAY_TMP_JOB_ID="${RAY_TMP_JOB_ID%%.*}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-${RAY_TMP_JOB_ID}}"
mkdir -p "${RAY_TMPDIR}"

WORKER_SETUP="source /etc/profile.d/modules.sh && module reset && module load cuda/12.2/12.2.2 && unset CUDA_VISIBLE_DEVICES && export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 && export RAY_TMPDIR=${RAY_TMPDIR} && export UV_CACHE_DIR=${UV_CACHE_DIR} && export HF_MODULES_CACHE=${HF_MODULES_CACHE} && export PYTHONPATH=${PYTHONPATH} && mkdir -p ${RAY_TMPDIR} ${UV_CACHE_DIR} ${HF_MODULES_CACHE} && cd ${REPO_DIR}"
WORKER_PIDS=()

cleanup() {
    set +e
    trap - EXIT
    echo "[INFO] Stopping Ray cluster..."

    local stop_pids=()
    for ((i = 1; i < NNODES; i++)); do
        pbsdsh -n "${i}" -- bash -lc "${WORKER_SETUP} && uv run --extra nemo-rl ray stop --force 2>/dev/null || true" &
        stop_pids+=("$!")
    done
    for pid in "${stop_pids[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done

    uv run --extra nemo-rl ray stop --force 2>/dev/null || true

    for pid in "${WORKER_PIDS[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done
    echo "[INFO] Ray cleanup done."
}

wait_for_ray_cluster() {
    local expected_nodes=$1
    local expected_gpus=$2

    uv run --extra nemo-rl python - "${expected_nodes}" "${expected_gpus}" <<'PY'
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

uv run --extra nemo-rl ray stop --force 2>/dev/null || true
rm -rf "${RAY_TMPDIR}" /tmp/ray 2>/dev/null || true
mkdir -p "${RAY_TMPDIR}"

cleanup_pids=()
for ((i = 1; i < NNODES; i++)); do
    pbsdsh -n "${i}" -- bash -lc "${WORKER_SETUP} && uv run --extra nemo-rl ray stop --force 2>/dev/null || true; rm -rf ${RAY_TMPDIR} /tmp/ray 2>/dev/null; mkdir -p ${RAY_TMPDIR}" &
    cleanup_pids+=("$!")
done
for pid in "${cleanup_pids[@]}"; do
    wait "${pid}"
done
sleep 2

uv run --extra nemo-rl ray start \
    --head \
    --port="${RAY_PORT}" \
    --num-gpus="${GPUS_PER_NODE}" \
    --disable-usage-stats \
    --resources='{"slurm_managed_ray_cluster": 1}'
sleep 5

for ((i = 1; i < NNODES; i++)); do
    echo "[INFO] Starting Ray worker on vnode ${i}..."
    pbsdsh -n "${i}" -- bash -lc "${WORKER_SETUP} && uv run --extra nemo-rl ray start --address=${HEAD_IP}:${RAY_PORT} --num-gpus=${GPUS_PER_NODE} --disable-usage-stats --block" &
    WORKER_PIDS+=("$!")
    sleep 3
done

export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
wait_for_ray_cluster "${NNODES}" "$((NNODES * GPUS_PER_NODE))"

uv run --extra nemo-rl ray status

export CLUSTER_NUM_NODES="${NNODES}"
export GPUS_PER_NODE="${GPUS_PER_NODE}"

if [[ -z "${TRAIN_GLOBAL_BATCH_SIZE:-}" ]]; then
    export TRAIN_GLOBAL_BATCH_SIZE="$((${TRAIN_GLOBAL_BATCH_SIZE_PER_NODE:-128} * NNODES))"
fi
if [[ -z "${VAL_GLOBAL_BATCH_SIZE:-}" ]]; then
    export VAL_GLOBAL_BATCH_SIZE="$((${VAL_GLOBAL_BATCH_SIZE_PER_NODE:-8} * NNODES))"
fi

echo "[INFO] CLUSTER_NUM_NODES=${CLUSTER_NUM_NODES}"
echo "[INFO] GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "[INFO] TRAIN_GLOBAL_BATCH_SIZE=${TRAIN_GLOBAL_BATCH_SIZE}"
echo "[INFO] VAL_GLOBAL_BATCH_SIZE=${VAL_GLOBAL_BATCH_SIZE}"

bash exp/dpo/run_llmjp-4-8b_nemo-rl_dpo.sh "$@"

echo "[INFO] Multi-node DPO training finished."
