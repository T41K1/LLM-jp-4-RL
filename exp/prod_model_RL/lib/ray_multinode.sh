#!/bin/sh
# Ray マルチノードクラスタ起動/停止のヘルパー関数群
#
# 使い方 (呼び出し側スクリプト):
#   source ./lib/ray_multinode.sh
#   trap cleanup_ray_cluster EXIT
#   setup_ray_cluster
#   # ↑ここまでで RAY_ADDRESS, NNODES, HEAD_IP, GPUS_PER_NODE が利用可能
#   python3 -m verl.trainer.main_ppo \
#       +ray_kwargs.ray_init.address="${RAY_ADDRESS}" \
#       trainer.nnodes=${NNODES} \
#       trainer.n_gpus_per_node=${GPUS_PER_NODE} ...
#
# 前提:
#   - PBS_NODEFILE が存在する (PBSジョブ内で実行)
#   - 呼び出し側で module load cuda / venv 有効化 / WANDB_* export 済み
#     (これらは WORKER_SETUP でワーカー側にも再現される)
#
# カスタマイズ可能な環境変数 (呼び出し前に export して上書き):
#   GPUS_PER_NODE   1ノードあたりGPU数 (default: 8)
#   RAY_PORT        Ray GCS のポート (default: 6379)
#   RAY_TMPDIR      Ray 一時ディレクトリ (default: ${TMPDIR:-/tmp}/ray)
#   CUDA_MODULE     module load するCUDAバージョン (default: cuda/12.2/12.2.2)

: "${GPUS_PER_NODE:=8}"
: "${RAY_PORT:=6379}"
: "${RAY_TMPDIR:=${TMPDIR:-/tmp}/ray}"
: "${CUDA_MODULE:=cuda/12.2/12.2.2}"

WORKER_PIDS=()

# --- 内部用: ワーカーノードでのセットアップコマンド ---
_build_worker_setup() {
    WORKER_SETUP="source /etc/profile.d/modules.sh && module reset && module load ${CUDA_MODULE} && unset CUDA_VISIBLE_DEVICES && export WANDB_ENTITY=${WANDB_ENTITY:-} && export RAY_TMPDIR=${RAY_TMPDIR} && mkdir -p ${RAY_TMPDIR} && cd ${PBS_O_WORKDIR} && source .venv/bin/activate"
}

# --- Rayクラスタが期待規模に到達するまで待機 ---
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

# --- Rayクラスタ全停止 (trap EXIT 用) ---
cleanup_ray_cluster() {
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

# --- メイン: ノード情報取得 → 既存Ray掃除 → head起動 → worker接続 → 接続確認 ---
setup_ray_cluster() {
    # ノード情報取得
    mapfile -t NODES < <(awk '!seen[$0]++' "$PBS_NODEFILE")
    NNODES=${#NODES[@]}
    HEAD_NODE=$(hostname -s)
    HEAD_IP=$(hostname -i | awk '{print $1}')

    echo "[INFO] NNODES=${NNODES}, HEAD=${HEAD_NODE} (${HEAD_IP}), WORKERS=${NODES[*]:1}"

    _build_worker_setup

    # 既存Rayプロセスを掃除
    ray stop --force 2>/dev/null || true
    rm -rf "${RAY_TMPDIR}" /tmp/ray 2>/dev/null || true
    mkdir -p "${RAY_TMPDIR}"
    local cleanup_pids=()
    for ((i = 1; i < NNODES; i++)); do
        pbsdsh -n "${i}" -- bash -lc "${WORKER_SETUP} && ray stop --force 2>/dev/null || true; rm -rf ${RAY_TMPDIR} /tmp/ray 2>/dev/null; mkdir -p ${RAY_TMPDIR}" &
        cleanup_pids+=("$!")
    done
    for pid in "${cleanup_pids[@]}"; do
        wait "${pid}"
    done
    sleep 2

    # ヘッドノードでRay起動
    ray start --head --port=${RAY_PORT} --num-gpus=${GPUS_PER_NODE} --disable-usage-stats
    sleep 5

    # ワーカーノードをRayクラスタに接続
    for ((i = 1; i < NNODES; i++)); do
        echo "[INFO] Starting Ray worker on vnode ${i}..."
        pbsdsh -n "${i}" -- bash -lc "${WORKER_SETUP} && ray start --address=${HEAD_IP}:${RAY_PORT} --num-gpus=${GPUS_PER_NODE} --disable-usage-stats --block" &
        WORKER_PIDS+=("$!")
        sleep 3
    done

    # Rayクラスタの接続確認
    export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
    wait_for_ray_cluster "${NNODES}" "$((NNODES * GPUS_PER_NODE))"
    ray status
}
