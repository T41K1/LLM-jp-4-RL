#!/bin/bash
#SBATCH --job-name=0316_llmjp4-eval-AIME24,25
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# AIME2024 + AIME2025 を verl の val_only モードで評価
# 前提: 下の parquet が存在していること
#   data/AIME2024/test.parquet  (uv run python data_load/aime2024.py)
#   data/AIME2025/test.parquet  (uv run python data_load/aime2025.py)
#
# 投入: sbatch exp/eval/val_aime_2024_2025.sh
# 状態: squeue -u $USER

set -xeuo pipefail

echo "Current directory: $(pwd)"
cd ${SLURM_SUBMIT_DIR:-$(pwd)}
echo "Current directory: $(pwd)"

# moduleが使える環境なら (無ければコメントアウトOK)
module reset 2>/dev/null || true
module load cuda/12.2/12.2.2 2>/dev/null || true
unset CUDA_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES   # SLURMが自動設定する場合あり。verl workerがCUDAと衝突検知するのを防ぐ

export VLLM_USE_V1=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
source .venv/bin/activate

set -a          # 以降の source で定義された変数を自動 export
source ./.env
set +a
echo "WANDB_API_KEY: ${WANDB_API_KEY:0:6}..."
echo "HF_TOKEN: ${HF_TOKEN:0:6}..."

export WANDB_ENTITY=Research_00

# 評価対象モデル (HFから model/ 以下にDL済み前提)
#   uv run hf download llm-jp/llm-jp-4-8b-thinking --local-dir model/llm-jp-4-8b-thinking
MODEL_PATH=model/llm-jp-4-8b-thinking

#valのtemperature
val_temperature=0.6
#何回問題を解くか
pass_at_k=128

project_name='0316_llm-jp-4-rl-eval'
exp_name="val-aime-2024-2025-llmjp-4-8b-thinking-${val_temperature}_${pass_at_k}-fix"


# 生成サンプルを jsonl として落とす先 (per-sample で目視確認用)
VAL_DUMP_DIR="outputs/val/${exp_name}"
mkdir -p "${VAL_DUMP_DIR}"

# train_files は verl の要求で必須。val_only=True では使われないのでダミーに val と同じものを渡す
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/AIME2024/test.parquet \
    data.val_files='[data/AIME2024/test.parquet,data/AIME2025/test.parquet]' \
    data.train_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=32768 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n="${pass_at_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature="${val_temperature}" \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    reward.custom_reward_function.path=rewards/math_reward.py \
    reward.custom_reward_function.name=compute_score \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.log_val_generations=32 \
    trainer.validation_data_dir="${VAL_DUMP_DIR}" \
    trainer.total_epochs=1 $@
