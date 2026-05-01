#!/bin/sh
#PBS -P gcg51557
#PBS -q rt_HF
#PBS -q R9920261000  
#PBS -N 0316_llm-jp-4-RL
#PBS -l select=1
#PBS -l walltime=500:00:00
#PBS -j oe
#PBS -o logs/
#PBS -e logs/

# OLMo3踏襲: MathVerifier (rewards/math_reward.py) を使用したGRPO

#set upとか
set -xeuo pipefail

echo "Current directory: $(pwd)"
# PBSの作業ディレクトリ（ジョブ投稿ディレクトリ）に移動
cd ${PBS_O_WORKDIR:-$(pwd)}
echo "Current directory: $(pwd)"

#moduleをloadする
module reset

module load cuda/12.2/12.2.2
unset CUDA_VISIBLE_DEVICES

export VLLM_USE_V1=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 
source .venv/bin/activate

source ./.env
echo "WANDB_API_KEY: ${WANDB_API_KEY:0:6}..."



#モデル名の変更
#MODEL_PATH=/groups/gcg51557/experiments/0213_v4-8b/tasks/v4-8b-decay3m/checkpoints_hf/iter_3000000
# MODEL_PATH=/groups/gcg51557/experiments/0309_llmjp4_instruct5/checkpoints/nemo_to_hf/sft-1960947055
MODEL_PATH=model/llm-jp-4-8b-thinking #本番モデル



#WANDBの設定
export WANDB_ENTITY=Research_00
project_name='0316_llm-jp-4-RL'
exp_name="${MODEL_PATH}-GRPO-Olmo3-Math-$(date +%Y%m%d)"


# 生成サンプルを jsonl として落とす先 (per-sample で目視確認用)
VAL_DUMP_DIR="outputs/val/${exp_name}"
mkdir -p "${VAL_DUMP_DIR}"


python3 -m verl.trainer.main_ppo \
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
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.log_val_generations=60 \
    trainer.validation_data_dir="${VAL_DUMP_DIR}" \
    trainer.total_epochs=15 $@