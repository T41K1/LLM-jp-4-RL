#!/usr/bin/env bash
#PBS -P gcg51557
#PBS -v RTYPE=rt_HF
#PBS -q R9920261000
#PBS -N 0316_llm-jp-4-RL-prompt-dapo
#PBS -l select=4
#PBS -l walltime=500:00:00
#PBS -j oe
#PBS -o logs/
#PBS -e logs/

# マルチノード版 OLMo3踏襲 GRPO + DAPO dynamic sampling
# run_llmjp-4-8b-prod_olomo3_math_multinode.sh の DAPO 派生版。
#
# 主な違い:
#   - エントリポイントを verl.trainer.main_ppo -> recipe.dapo.main_dapo に変更
#     (trainer を RayDAPOTrainer に差し替えるだけで、他の引数は ppo_trainer 継承でそのまま通る)
#   - DAPO の dynamic sampling (algorithm.filter_groups) を有効化
#     acc が全部同じ (全問正解 or 全問不正解) グループを捨て、train_batch_size 分の
#     有効グループが集まるまで gen_batch_size 単位で生成を繰り返す。
#   - reward_manager は dapo を使うが overlong reward shaping は OFF (enable=False)。
#     reward 挙動は現行スクリプトと同一 (max_resp_len は dapo manager の assert を通すためだけに渡す)。
#   - Clip-Higher (DAPO) ON: clip_ratio_low=0.2 / clip_ratio_high=0.28。
#   - overlong reward shaping は OFF (任意要素)。有効化したい場合は末尾コメント参照。
#
# recipe submodule が必要: deps/verl/recipe が空なら
#   git -C deps/verl submodule update --init recipe
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

# recipe submodule が初期化済みか確認 (DAPO trainer はここに入っている)
if [[ ! -f deps/verl/recipe/dapo/main_dapo.py ]]; then
    echo "[ERROR] deps/verl/recipe/dapo/main_dapo.py が見つかりません。" >&2
    echo "[ERROR] 先に: git -C deps/verl submodule update --init recipe" >&2
    exit 1
fi

# --- reward verifier 選択 ---
# REWARD_VERIFIER=math_verify : math_verify ベース (rewards/math_verify_verifier.py)
# REWARD_VERIFIER=legacy      : 旧 MathVerifier (rewards/ground_truth_utils.py, 既定)
# rewards/math_reward.py が import 時にこの環境変数を読むため、学習プロセス起動前に export する。
export REWARD_VERIFIER="${REWARD_VERIFIER:-math_verify}"
case "${REWARD_VERIFIER}" in
    math_verify|legacy) ;;
    *) echo "[ERROR] REWARD_VERIFIER must be 'math_verify' or 'legacy', got '${REWARD_VERIFIER}'" >&2; exit 1 ;;
esac
echo "[INFO] reward verifier: REWARD_VERIFIER=${REWARD_VERIFIER}"


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
    mean) NORM_ADV_BY_STD=False; loss_agg_mode="seq-mean-token-sum-norm"; loss_scale_factor=32768 ;;
    std)  NORM_ADV_BY_STD=True;  loss_agg_mode="token-mean";             loss_scale_factor=32768 ;;
    *) echo "[ERROR] ADV_NORM must be 'mean' or 'std', got '${ADV_NORM}'" >&2; exit 1 ;;
esac
echo "[INFO] GRPO advantage normalization: ADV_NORM=${ADV_NORM} (norm_adv_by_std_in_grpo=${NORM_ADV_BY_STD})"

# --- DAPO dynamic sampling 設定 ---
# FILTER_GROUPS=1 で acc が全部同じグループを除外 (DAPO dynamic sampling)。
# FILTER_METRIC: 判定に使う指標 (math_reward.py が返す acc を使用)。
# MAX_NUM_GEN_BATCHES: 有効グループが揃うまでの最大生成回数 (0=無制限)。
# GEN_PROMPTS: 1回の生成で投入する prompt 数 (= gen_batch_size)。
#              DAPO 公式は train_batch_size の 3 倍にオーバーサンプリングしている
#              (filter で acc 一様なグループを捨てた後も train_batch_size 分が揃いやすく、
#               Keep generating... の再生成ラウンドが減る)。既定もそれに倣って 3 倍。
FILTER_GROUPS="${FILTER_GROUPS:-1}"
FILTER_METRIC="${FILTER_METRIC:-acc}"
MAX_NUM_GEN_BATCHES="${MAX_NUM_GEN_BATCHES:-64}"
case "${FILTER_GROUPS}" in
    1) ENABLE_FILTER_GROUPS=True ;;
    0) ENABLE_FILTER_GROUPS=False ;;
    *) echo "[ERROR] FILTER_GROUPS must be 0 or 1, got '${FILTER_GROUPS}'" >&2; exit 1 ;;
esac
echo "[INFO] DAPO dynamic sampling: FILTER_GROUPS=${FILTER_GROUPS} (enable=${ENABLE_FILTER_GROUPS}) metric=${FILTER_METRIC} max_num_gen_batches=${MAX_NUM_GEN_BATCHES}"

project_name='0316_llm-jp-4-RL'
exp_name="${EXP_NAME:-${MODEL_PATH##*/}-DAPO-Olmo3-Math-adv${ADV_NORM}-${REWARD_VERIFIER}-ds-${NNODES}node-$(date +%Y%m%d)}"

VAL_DUMP_DIR="outputs/val/${exp_name}"
mkdir -p "${VAL_DUMP_DIR}"

# train 中の rollout dump は巨大になりうるため opt-in。
# SAVE_TRAIN_ROLLOUTS=1 なら outputs/rollout/${exp_name} に保存する。
# TRAIN_ROLLOUT_DUMP_DIR=/path を指定した場合はそのディレクトリに保存する。
SAVE_TRAIN_ROLLOUTS="${SAVE_TRAIN_ROLLOUTS:-1}"
TRAIN_ROLLOUT_DUMP_DIR="${TRAIN_ROLLOUT_DUMP_DIR:-}"
if [[ "${SAVE_TRAIN_ROLLOUTS}" == "1" && -z "${TRAIN_ROLLOUT_DUMP_DIR}" ]]; then
    TRAIN_ROLLOUT_DUMP_DIR="outputs/rollout/${exp_name}"
fi
TRAIN_ROLLOUT_DUMP_ARGS=()
if [[ -n "${TRAIN_ROLLOUT_DUMP_DIR}" ]]; then
    mkdir -p "${TRAIN_ROLLOUT_DUMP_DIR}"
    TRAIN_ROLLOUT_DUMP_ARGS=(trainer.rollout_data_dir="${TRAIN_ROLLOUT_DUMP_DIR}")
    echo "[INFO] train rollout dump: ${TRAIN_ROLLOUT_DUMP_DIR}"
else
    echo "[INFO] train rollout dump: disabled"
fi



N=${MY_N}
NUM_PROMPTS=${NUM_PROMPTS}
ENT=1e-3
LR=1e-6
MINI_BATCH_SIZE=${MINI_BATCH_SIZE}
BS=$((${N} * ${NUM_PROMPTS}))
MBS=$((${N} * ${MINI_BATCH_SIZE}))

# dynamic sampling の 1 回あたり生成 prompt 数 (= gen_batch_size)。DAPO 公式に倣い既定 3 倍のところ高速化を考え、2に変更
GEN_PROMPTS="${GEN_PROMPTS:-$((NUM_PROMPTS * 2))}"
# Clip-Higher (DAPO): clip 上限を緩める非対称クリップ。公式既定 low=0.2 / high=0.28。
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"
echo "[INFO] DAPO clip-higher: clip_ratio_low=${CLIP_RATIO_LOW} clip_ratio_high=${CLIP_RATIO_HIGH}"
# overlong reward shaping は OFF だが、dapo reward_manager の assert を通すため max_resp_len を渡す
MAX_RESP_LEN=32768

# dapo_trainer.yaml は base config を相対パス (file://verl/trainer/config) で探すため、
# プロジェクトルートから起動すると ppo_trainer をロードできない。
# cwd はルートのまま (reward/data の相対パスを保つ) 、searchpath を絶対パスで上書きする。
VERL_TRAINER_CONFIG_DIR="$(pwd)/deps/verl/verl/trainer/config"
if [[ ! -f "${VERL_TRAINER_CONFIG_DIR}/ppo_trainer.yaml" ]]; then
    echo "[ERROR] ${VERL_TRAINER_CONFIG_DIR}/ppo_trainer.yaml が見つかりません" >&2
    exit 1
fi


# --- 学習実行 ---
# エントリポイントは recipe.dapo.main_dapo (RayDAPOTrainer)。
# deps/verl が editable install 済みのため recipe パッケージはそのまま import 可能。
python3 -m recipe.dapo.main_dapo \
    hydra.searchpath="[file://${VERL_TRAINER_CONFIG_DIR}]" \
    trainer.val_metrics.pass_at_k=true \
    +ray_kwargs.ray_init.address="${RAY_ADDRESS}" \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=${NORM_ADV_BY_STD} \
    algorithm.filter_groups.enable=${ENABLE_FILTER_GROUPS} \
    algorithm.filter_groups.metric=${FILTER_METRIC} \
    algorithm.filter_groups.max_num_gen_batches=${MAX_NUM_GEN_BATCHES} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.loss_scale_factor=${loss_scale_factor} \
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
    data.train_files=data/Dolci-Think-RL-7B-math/train_boxed.parquet \
    data.val_files='[data/AIME2024/test.parquet,data/AIME2025/test.parquet]' \
    data.train_batch_size=${NUM_PROMPTS} \
    data.gen_batch_size=${GEN_PROMPTS} \
    data.max_prompt_length=2048 \
    data.max_response_length=${MAX_RESP_LEN} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.rollout.max_model_len=36864 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.90 \
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
    reward.reward_manager.name=dapo \
    reward.reward_kwargs.max_resp_len=${MAX_RESP_LEN} \
    reward.reward_kwargs.overlong_buffer_cfg.enable=False \
    reward.reward_kwargs.overlong_buffer_cfg.len=0 \
    reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=0.0 \
    reward.reward_kwargs.overlong_buffer_cfg.log=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.val_before_train=True \
    trainer.save_freq=${SAVE_FREQ:-10} \
    trainer.test_freq=${TEST_FREQ:-20} \
    trainer.log_val_generations=60 \
    trainer.validation_data_dir="${VAL_DUMP_DIR}" \
    "${TRAIN_ROLLOUT_DUMP_ARGS[@]}" \
    trainer.total_epochs=15 "$@"

echo "[INFO] Training finished."

# --- overlong reward shaping を有効化する場合 (上の reward.reward_kwargs.overlong_buffer_cfg を差し替え) ---
#   reward.reward_kwargs.overlong_buffer_cfg.enable=True \
#   reward.reward_kwargs.overlong_buffer_cfg.len=4096 \
#   reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
