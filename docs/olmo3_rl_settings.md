# OLMo3 RL Settings Research Notes

This note summarizes the OLMo 3 reinforcement-learning settings that are most relevant to this repository's math GRPO runs, using AllenAI primary sources.

## Sources

- OLMo 3 technical report landing page: https://arxiv.org/abs/2512.13961
- AllenAI OLMo 3 training script index: https://raw.githubusercontent.com/allenai/open-instruct/main/scripts/train/olmo3/README.md
- OLMo 3 7B RL Zero Math script: https://raw.githubusercontent.com/allenai/open-instruct/main/scripts/train/olmo3/7b_rlzero_math.sh
- OLMo 3 7B RL Zero Code script: https://raw.githubusercontent.com/allenai/open-instruct/main/scripts/train/olmo3/7b_rlzero_code.sh
- OLMo 3 7B RL Zero IF script: https://raw.githubusercontent.com/allenai/open-instruct/main/scripts/train/olmo3/7b_rlzero_instruction_following.sh
- OLMo 3 7B RL Zero General script: https://raw.githubusercontent.com/allenai/open-instruct/main/scripts/train/olmo3/7b_rlzero_general.sh
- OLMo 3 7B Think RL script: https://raw.githubusercontent.com/allenai/open-instruct/main/scripts/train/olmo3/7b_think_rl.sh
- OLMo 3 32B Think RL script: https://raw.githubusercontent.com/allenai/open-instruct/main/scripts/train/olmo3/32b_think_rl.sh

## What OLMo3 Actually Released

AllenAI's OLMo 3 training README maps each released model to a concrete public training script and commit. For RL-relevant models, the official scripts are:

- `7b_instruct_rl.sh`
- `7b_think_rl.sh`
- `7b_rlzero_math.sh`
- `7b_rlzero_code.sh`
- `7b_rlzero_instruction_following.sh`
- `7b_rlzero_general.sh`
- `32b_think_rl.sh`

For this repository, the closest match is `7b_rlzero_math.sh`, because the local experiment:

- uses verifiable math reward
- samples multiple completions per prompt
- disables KL loss
- starts from an instruction-tuned checkpoint rather than a base pretrain checkpoint

## OLMo3 RL Zero Math: Core Hyperparameters

Extracted from `7b_rlzero_math.sh`.

| Setting | OLMo3 value | Notes |
|---|---:|---|
| Objective | GRPO / RLVR | `open_instruct/grpo_fast.py` with verifiable reward |
| KL coefficient | `beta = 0.0` | No KL penalty in the RL objective |
| Samples per prompt | `8` | `--num_samples_per_prompt_rollout 8` |
| Unique prompts per rollout | `32` | `--num_unique_prompts_rollout 32` |
| Mini-batches per update | `1` | `--num_mini_batches 1` |
| Epochs per sampled batch | `1` | `--num_epochs 1` |
| Learning rate | `1e-6` | Constant scheduler |
| Per-device train batch | `1` | Learner-side micro batch |
| KL estimator | `2` | Open-instruct internal setting, no direct `verl` knob here |
| Prompt length | `2048` | `--max_prompt_token_length 2048` |
| Response length | `16384` | `--response_length 16384` |
| Pack length | `18432` | Prompt + response budget |
| Temperature | `1.0` | Sampling during rollout |
| Non-stop penalty | `False` | Do not penalize unfinished answers |
| Mask truncated completions | `False` | Keep truncated samples in training |
| Active sampling | enabled | `--active_sampling` |
| No-resampling pass rate | `0.875` | `--no_resampling_pass_rate 0.875` |
| Total episodes | `768000` | Open-instruct episode budget |
| Reward type | verifiable reward | `--apply_verifiable_reward true` |
| Save frequency | `100` | in open-instruct trainer steps |
| Eval frequency | `25` | local eval cadence |

## Shared RL Zero Pattern Across OLMo3 7B

The math/code/IF/general RL Zero scripts share a stable recipe:

- `beta=0.0`
- `num_samples_per_prompt_rollout=8`
- `num_unique_prompts_rollout=32`
- `num_mini_batches=1`
- `num_epochs=1`
- `learning_rate=1e-6`
- `per_device_train_batch_size=1`
- `max_prompt_token_length=2048`
- `response_length=16384`
- `pack_length=18432`
- `temperature=1.0`
- `lr_scheduler_type=constant`
- `apply_verifiable_reward=true`

This is the strongest evidence for the "house style" of OLMo3 RL Zero.

## How This Maps To `verl`

`open-instruct` and `verl` do not expose identical trainer knobs, so the mapping is approximate:

| OLMo3 open-instruct | `verl` mapping used here | Rationale |
|---|---|---|
| `num_samples_per_prompt_rollout=8` | `actor_rollout_ref.rollout.n=8` | exact match |
| `num_unique_prompts_rollout=32` | `data.train_batch_size=32` | in `verl`, prompt batch size is separate from `rollout.n` |
| `num_mini_batches=1` | `actor_rollout_ref.actor.ppo_mini_batch_size=32` | one learner mini-batch over the prompt batch |
| `per_device_train_batch_size=1` | `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1` | closest micro-batch analogue |
| `beta=0.0` | `actor_rollout_ref.actor.use_kl_loss=False` and `algorithm.use_kl_in_reward=False` | disables KL regularization paths |
| `max_prompt_token_length=2048` | `data.max_prompt_length=2048` | exact match |
| `response_length=16384` | `data.max_response_length=16384` | exact match |
| `temperature=1.0` | `actor_rollout_ref.rollout.temperature=1.0` | exact match |
| default sampling without top-p truncation | `actor_rollout_ref.rollout.top_p=1.0` | avoids added nucleus filtering |

## Settings Not Directly Transferable To `verl`

These OLMo3 settings do not have a clean 1:1 translation in the current local `verl` scripts:

- `active_sampling`
- `no_resampling_pass_rate=0.875`
- `kl_estimator=2`
- `pack_length`
- `total_episodes=768000`
- cluster and engine sizing such as `num_nodes`, `vllm_num_engines`, and Beaker-specific launch settings

Those are either framework-specific or infrastructure-specific. They should be treated as design intent, not copied literally.

## Recommended Local `verl` Adaptation

For a 1-node, 8-GPU `verl + vLLM + FSDP` math run in this repository, the closest OLMo3-aligned configuration is:

- `algorithm.adv_estimator=grpo`
- `data.train_batch_size=32`
- `data.max_prompt_length=2048`
- `data.max_response_length=16384`
- `actor_rollout_ref.actor.optim.lr=1e-6`
- `actor_rollout_ref.actor.ppo_mini_batch_size=32`
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1`
- `actor_rollout_ref.actor.use_kl_loss=False`
- `actor_rollout_ref.rollout.n=8`
- `actor_rollout_ref.rollout.temperature=1.0`
- `actor_rollout_ref.rollout.top_p=1.0`
- `actor_rollout_ref.rollout.val_kwargs.n=8`
- `actor_rollout_ref.rollout.val_kwargs.temperature=1.0`
- `actor_rollout_ref.rollout.val_kwargs.top_p=1.0`
- `reward.custom_reward_function.path=rewards/math_reward.py`
- `reward.custom_reward_function.name=compute_score`
- `trainer.critic_warmup=0`

## Practical Implication For This Repo

The existing local OLMo3-style script already matched some of the recipe:

- GRPO objective
- no KL loss
- reward via `MathVerifier`
- `n=8`
- `temperature=1.0`
- prompt length `2048`

The main mismatches were:

- local prompt batch was much larger than OLMo3 RL Zero Math
- response length was `32768`, not `16384`
- validation `top_p` was `0.95`, while the official RL Zero scripts do not add that truncation

The new local aligned script fixes those mismatches while keeping infrastructure-specific settings compatible with this repository.
