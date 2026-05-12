# NeMo-RL DPO Runbook

このメモは、この repo から NeMo-RL の DPO を動かすための手順です。

## 方針

`deps/nemo-rl` は親 repo の submodule ではなく、通常の clone として置く想定です。NeMo-RL は親 repo の `uv` 環境に optional extra として入れ、`uv run --extra nemo-rl ...` で起動します。

```text
0316_llm-jp-4-rl/
  deps/
    verl/          # parent repo の submodule
    nemo-rl/       # 通常 clone した NeMo-RL repo
```

通常 clone では親 repo の `.gitmodules` は変わりません。この repo では Python 3.12 と既存 `verl[vllm]` 環境に寄せるため、NeMo-RL は `v0.5.0` を使います。`main` / `v0.6.0` は Python 3.13.13+ と CUDA13 系に寄っているため、この環境には統合しません。

## 1. NeMo-RL を clone

```bash
git clone https://github.com/NVIDIA-NeMo/RL.git deps/nemo-rl
git -C deps/nemo-rl switch --detach v0.5.0
```

FSDP DPO だけなら NeMo-RL 側の submodule は基本的に不要です。Megatron/MCore/Automodel/Gym backend を使う場合だけ、`git -C deps/nemo-rl submodule update --init --recursive` を実行します。

repo 外に clone する場合は、ジョブ投入時に `NEMO_RL_DIR` を指定します。

```bash
qsub -v NEMO_RL_DIR=/groups/gcg51557/experiments/nemo-rl exp/dpo/run_llmjp-4-8b_nemo-rl_dpo.sh
```

## 2. NeMo-RL の環境を確認

親 repo の optional extra として NeMo-RL を同期します。初回は torch/ray/transformers などの更新が走るため時間がかかります。

```bash
uv sync --extra nemo-rl
uv run --extra nemo-rl python deps/nemo-rl/examples/run_dpo.py --help
```

Hugging Face の gated model を使う場合は、事前に login しておきます。

```bash
huggingface-cli login
```

W&B を使う場合、この repo の `.env` に `WANDB_API_KEY` を置くか、ジョブ環境に渡します。

## 3. DPO データ形式

この repo の DPO スクリプトは、NeMo-RL の `BinaryPreferenceDataset` を使います。入力 JSONL は 1 行 1 JSON で、デフォルトでは以下の 3 key が必要です。

```jsonl
{"prompt":"問題文","chosen":"好ましい回答","rejected":"好ましくない回答"}
```

key 名を変えたい場合は、ジョブ投入時に `PROMPT_KEY`, `CHOSEN_KEY`, `REJECTED_KEY` を上書きします。

```bash
qsub -v PROMPT_KEY=context,CHOSEN_KEY=winner,REJECTED_KEY=loser exp/dpo/run_llmjp-4-8b_nemo-rl_dpo.sh
```

## 4. DPO JSONL を作る

### Dolci-Think-DPO-7B を使う場合

`allenai/Dolci-Think-DPO-7B` は、Hugging Face 上では `prompt` が文字列、`chosen/rejected` が user+assistant の message list です。NeMo-RL の `BinaryPreferenceDataset` は `prompt/chosen/rejected` の文字列 JSONL を読むため、この repo では `chosen/rejected` から最後の assistant 応答だけを抜き出して JSONL に変換します。

長さ上限を決める前に、prompt/response の文字数分布だけ確認できます。これは JSONL を出力せずに stream して集計します。

```bash
uv run python data_load/dolci_think_dpo_7b.py \
  --stats-only \
  --max-prompt-chars 50000 \
  --max-response-chars 80000
```

まず一部だけ見る場合:

```bash
uv run python data_load/dolci_think_dpo_7b.py \
  --stats-only \
  --max-stats-examples 10000 \
  --max-prompt-chars 50000 \
  --max-response-chars 80000
```

出力には `prompt_chars`, `chosen_chars`, `rejected_chars`, `max_response_chars`, `total_chars` の p50/p90/p95/p99/p99.5/p99.9/max と、指定した上限を超える件数が出ます。

まず小さく smoke 用データを作る場合:

```bash
uv run python data_load/dolci_think_dpo_7b.py \
  --save-dir data/Dolci-Think-DPO-7B/nemo-rl-smoke \
  --max-train-examples 1000 \
  --max-val-examples 128 \
  --overwrite
```

本番用に全件を変換する場合:

```bash
uv run python data_load/dolci_think_dpo_7b.py \
  --save-dir data/Dolci-Think-DPO-7B/nemo-rl \
  --val-ratio 0.01 \
  --max-prompt-chars 50000 \
  --max-response-chars 80000 \
  --overwrite
```

出力:

```text
data/Dolci-Think-DPO-7B/nemo-rl/train.jsonl
data/Dolci-Think-DPO-7B/nemo-rl/val.jsonl
```

`dataset_source` を絞りたい場合は `--dataset-source` を複数回指定できます。

```bash
uv run python data_load/dolci_think_dpo_7b.py \
  --dataset-source tulu-3-sft-personas-math \
  --dataset-source ultrafeedback_cleaned_olmo2_7b \
  --save-dir data/Dolci-Think-DPO-7B/nemo-rl-filtered \
  --overwrite
```

変換後は、専用ラッパーでそのまま投入できます。

```bash
qsub exp/dpo/run_llmjp-4-8b_nemo-rl_dolci-think-dpo-7b.sh
```

smoke データで短く回す場合:

```bash
qsub -v TRAIN_DATA=data/Dolci-Think-DPO-7B/nemo-rl-smoke/train.jsonl,VAL_DATA=data/Dolci-Think-DPO-7B/nemo-rl-smoke/val.jsonl,MAX_NUM_STEPS=5,WANDB_ENABLED=false exp/dpo/run_llmjp-4-8b_nemo-rl_dolci-think-dpo-7b.sh
```

### 汎用 pairwise preference データを使う場合

既に `prompt/chosen/rejected` 相当の pairwise preference データがある場合:

```bash
uv run python data_load/dpo_jsonl.py path/to/pairs.jsonl \
  --output-path data/dpo/train.jsonl \
  --overwrite
```

列名が違う場合:

```bash
uv run python data_load/dpo_jsonl.py path/to/pairs.parquet \
  --prompt-key context \
  --chosen-key winner \
  --rejected-key loser \
  --output-path data/dpo/train.jsonl \
  --overwrite
```

GRPO validation dump のような `input/output/score` 形式から、同じ prompt 内の最高 score と最低 score をペアにする場合:

```bash
uv run python data_load/dpo_jsonl.py outputs/val/<run>/0.jsonl \
  --output-path data/dpo/train.jsonl \
  --group-key input \
  --response-key output \
  --score-key score \
  --min-score-gap 1 \
  --max-response-chars 80000 \
  --overwrite
```

検証用にも同じ形式で `data/dpo/val.jsonl` を作ります。最初は小さく確認するなら `--max-examples` を使います。

```bash
uv run python data_load/dpo_jsonl.py outputs/val/<run>/0.jsonl \
  --output-path data/dpo/val.jsonl \
  --group-key input \
  --response-key output \
  --score-key score \
  --min-score-gap 1 \
  --max-response-chars 80000 \
  --max-examples 128 \
  --overwrite
```

生成後に件数と長さを確認します。

```bash
wc -l data/dpo/train.jsonl data/dpo/val.jsonl
python3 - <<'PY'
import json
for path in ["data/dpo/train.jsonl", "data/dpo/val.jsonl"]:
    with open(path) as f:
        rows = [json.loads(next(f)) for _ in range(3)]
    print(path)
    for row in rows:
        print({k: len(v) for k, v in row.items()})
PY
```

## 5. 1 ノード DPO を投入

デフォルトでは以下を使います。

- `NEMO_RL_DIR=deps/nemo-rl`
- `MODEL_PATH=model/llm-jp-4-8b-thinking`
- `TRAIN_DATA=data/dpo/train.jsonl`
- `VAL_DATA=data/dpo/val.jsonl`
- `GPUS_PER_NODE=8`
- `MAX_TOTAL_SEQUENCE_LENGTH=32768`

```bash
qsub exp/dpo/run_llmjp-4-8b_nemo-rl_dpo.sh
```

Dolci-Think-DPO-7B 用の標準 path を使う場合は、以下でも同じ DPO スクリプトに渡せます。

```bash
qsub exp/dpo/run_llmjp-4-8b_nemo-rl_dolci-think-dpo-7b.sh
```

主要パラメータを上書きする例:

```bash
qsub -v MODEL_PATH=model/llm-jp-4-8b-thinking,TRAIN_DATA=data/dpo/train.jsonl,VAL_DATA=data/dpo/val.jsonl,MAX_NUM_STEPS=1500,LR=8.0e-8,WARMUP_RATIO=0.1,LR_SCHEDULER=linear exp/dpo/run_llmjp-4-8b_nemo-rl_dpo.sh
```

W&B を使わない場合:

```bash
qsub -v WANDB_ENABLED=false exp/dpo/run_llmjp-4-8b_nemo-rl_dpo.sh
```

## 6. よく変える設定

| 環境変数 | デフォルト | 用途 |
|---|---:|---|
| `MODEL_PATH` | `model/llm-jp-4-8b-thinking` | policy と tokenizer の model path |
| `TRAIN_DATA` | `data/dpo/train.jsonl` | train preference JSONL |
| `VAL_DATA` | `data/dpo/val.jsonl` | validation preference JSONL |
| `MAX_TOTAL_SEQUENCE_LENGTH` | `32768` | prompt + response の最大長 |
| `TRAIN_GLOBAL_BATCH_SIZE` | `128` | preference pair 単位の global batch |
| `TRAIN_MICRO_BATCH_SIZE` | `1` | DP rank あたり micro batch |
| `LR` | `8.0e-8` | AdamW learning rate |
| `MAX_NUM_STEPS` | `1500` | 最大 training step |
| `LR_SCHEDULER` | `linear` | `linear` は warmup 後に線形減衰、`constant` は warmup 後に固定、`none` は scheduler 無効 |
| `WARMUP_RATIO` | `0.1` | `MAX_NUM_STEPS` に対する LR warmup 比率 |
| `WARMUP_STEPS` | 自動計算 | 明示した場合は `WARMUP_RATIO` より優先 |
| `WARMUP_START_FACTOR` | `0.1` | warmup 開始時の LR factor。開始 LR は `LR * WARMUP_START_FACTOR` |
| `LR_END_FACTOR` | `0.0` | `LR_SCHEDULER=linear` の最終 LR factor。最終 LR は `LR * LR_END_FACTOR` |
| `LR_DECAY_STEPS` | 自動計算 | 明示した場合は warmup 後の減衰 step 数を上書き |
| `OLMO3_LOSS` | `false` | `true` にすると `PREFERENCE_AVERAGE_LOG_PROBS=true`、未指定なら `KL_PENALTY=5` にする alias |
| `KL_PENALTY` | `0.05` | reference policy との DPO penalty |
| `SFT_LOSS_WEIGHT` | `0` | SFT loss を混ぜる係数 |
| `PREFERENCE_AVERAGE_LOG_PROBS` | `false` | response length で log prob を平均するか |
| `TENSOR_PARALLEL_SIZE` | `1` | dtensor tensor parallel size |
| `CONTEXT_PARALLEL_SIZE` | `1` | dtensor context parallel size |
| `WANDB_ENABLED` | `true` | W&B logging |

## 7. NeMo-RL DPO metric の見方

NeMo-RL DPO は console では `Training Results` / `Validation Results` として metric 名を表示し、W&B/TensorBoard では prefix 付きで記録します。validation dataset 名が default の場合、W&B 上の主な名前は `train/<metric>` と `validation-default/<metric>` です。

| console の metric 名 | W&B/TensorBoard 名 | 意味 | DPO での見方 |
|---|---|---|---|
| `loss` | `train/loss`, `validation-default/loss` | 最終 loss。`sft_loss_weight * sft_loss + preference_loss_weight * preference_loss` | 今の設定は `SFT_LOSS_WEIGHT=0` なので、ほぼ `preference_loss` と同じ |
| `preference_loss` | `train/preference_loss`, `validation-default/preference_loss` | DPO preference loss。`-logsigmoid(beta * (reward_chosen - reward_rejected))` | 下がるほど chosen を rejected より好む方向。ただし下がりすぎ・val 悪化は過学習に注意 |
| `sft_loss` | `train/sft_loss`, `validation-default/sft_loss` | chosen 側にかける optional SFT loss | `SFT_LOSS_WEIGHT=0` なら 0 のままで正常 |
| `accuracy` | `train/accuracy`, `validation-default/accuracy` | `reward_chosen > reward_rejected` になった preference pair の割合 | 0.5 付近から上がるのが自然。batch/val が小さいとかなり揺れる |
| `rewards_chosen_mean` | `train/rewards_chosen_mean`, `validation-default/rewards_chosen_mean` | chosen response の reward 平均 | 単体の絶対値より `rewards_rejected_mean` との差を見る |
| `rewards_rejected_mean` | `train/rewards_rejected_mean`, `validation-default/rewards_rejected_mean` | rejected response の reward 平均 | chosen より低くなる方向なら DPO としては自然 |
| `num_valid_samples` | `train/num_valid_samples`, `validation-default/num_valid_samples` | 有効な preference pair 数 | token 長超過などで落ちた sample は入らない。想定 batch より小さい場合は data/filter を確認 |
| `global_valid_seqs` | `train/global_valid_seqs`, `validation-default/global_valid_seqs` | 有効 sequence 数。DPO では chosen/rejected で pair の約2倍 | `num_valid_samples * 2` に近い値なら自然 |
| `global_valid_toks` | `train/global_valid_toks`, `validation-default/global_valid_toks` | loss/timing に使われた有効 token 数 | throughput や長さ分布を見るための補助指標 |

NeMo-RL の DPO reward は、response token 上の `log pi_theta - log pi_ref` の和です。`PREFERENCE_AVERAGE_LOG_PROBS=true` にすると token 数で平均します。`KL_PENALTY` は config 名では `dpo.reference_policy_kl_penalty` で、DPO 論文の beta に相当します。reported reward 自体に beta を掛けるのではなく、`preference_loss` 内の `beta * (reward_chosen - reward_rejected)` に使われます。

Dolci のように response が長い DPO では、`PREFERENCE_AVERAGE_LOG_PROBS=false` のままだと reward が response token 数ぶんの和になり、`rewards_chosen_mean - rewards_rejected_mean` が数百から千以上まで膨らむことがあります。この状態では `beta * reward_diff` が大きくなりすぎて `logsigmoid` が飽和し、`loss` / `preference_loss` が console 上で `0.0000` に張り付きます。これは W&B 表示だけの問題ではなく、ログにも同じ値が出ます。

Dolci で loss が早すぎる段階から `0.0000` になる場合は、まず `PREFERENCE_AVERAGE_LOG_PROBS=true` にして length normalization を入れます。その上でまだ reward 差が極端なら、`KL_PENALTY` を `0.01` や `0.005` に下げる、`LR` を下げる、`VAL_BATCHES` を増やして validation の標本数を増やす、の順に切り分けます。

OLMo 3 の `dpo_norm` 的に、average log-ratio と `beta=5` の組み合わせで回したい場合は `OLMO3_LOSS=true` を使えます。

```bash
qsub -v OLMO3_LOSS=true exp/dpo/run_llmjp-4-8b_nemo-rl_dolci-think-dpo-7b.sh
```

これは以下をまとめて指定する alias です。

```bash
PREFERENCE_AVERAGE_LOG_PROBS=true
KL_PENALTY=5
```

学習の健全性を見るときは、まず以下の組み合わせを見ます。

```text
validation-default/preference_loss が下がる
validation-default/accuracy が上がる
validation-default/rewards_chosen_mean - validation-default/rewards_rejected_mean が広がる
train と validation の差が広がりすぎない
```

`rewards_chosen_mean` と `rewards_rejected_mean` は絶対値より差が重要です。reward 差が急に極端に広がり、validation loss/accuracy が悪化する場合は、`KL_PENALTY` が強すぎる、LR が高すぎる、または overfit の可能性があります。

timing 系は別 prefix で記録されます。

| metric 名 | 意味 |
|---|---|
| `timing/train/total_step_time` | validation/checkpoint を含む step 全体の時間 |
| `timing/train/policy_training` | policy の forward/backward/optimizer 周りの時間。速度比較ではここを見る |
| `timing/train/checkpointing` | checkpoint 保存時間。`SAVE_PERIOD` が短いと total step time を支配する |
| `timing/train/valid_tokens_per_sec_per_gpu` | 有効 token/sec/GPU。batch や TP/DP 比較では重要 |
| `timing/validation-default/total_validation_time` | default validation set の validation 時間 |
| `timing/validation/total_validation_time` | 全 validation set の合計 validation 時間 |

console に出る `Training FLOPS` と `Training Model Floating Point Utilization` は速度比較用です。W&B には `train/train_fp_utilization` が入ります。checkpoint の選択 metric は W&B 名ではなく NeMo-RL 内部の `val:<metric>` 形式を使うため、今は `val:validation-default_loss` を指定しています。

## 8. 出力

ログ:

```text
logs/<PBS_JOBID>.nemo-rl-dpo.log
logs/nemo-rl-dpo/
```

checkpoint:

```text
checkpoints/nemo-rl-dpo/<EXP_NAME>/
```

base script で `EXP_NAME` を指定しない場合、`llm-jp-4-8b-nemo-rl-dpo-YYYYMMDD-<jobid>` になります。Dolci wrapper は checkpoint の誤 resume を避けるため、loss 種別、scheduler、job id を含む名前を自動生成します。

## 9. トラブルシュート

`NEMO_RL_DIR does not exist`:

`deps/nemo-rl` が無いか、repo 外 clone の path を渡していません。`NEMO_RL_DIR=/path/to/nemo-rl` を指定します。

`Required path does not exist`:

`MODEL_PATH`, `TRAIN_DATA`, `VAL_DATA`, `CONFIG_PATH` のどれかが存在しません。ジョブログ冒頭に実際の path が出ます。

OOM:

まず `TRAIN_MICRO_BATCH_SIZE=1` のまま、`MAX_TOTAL_SEQUENCE_LENGTH` を下げます。次に `TRAIN_GLOBAL_BATCH_SIZE` を下げます。長すぎる response は `data_load/dpo_jsonl.py --max-response-chars` で落としておくと切り分けしやすいです。

validation が重い:

`VAL_BATCHES` と `VAL_GLOBAL_BATCH_SIZE` を下げます。

W&B 周りで落ちる:

`WANDB_ENABLED=false` で切り分けます。

## 参考

- NeMo-RL DPO guide: https://docs.nvidia.com/nemo/rl/0.6.0/guides/dpo.html
- NeMo-RL DPO examples: https://docs.nvidia.com/nemo/rl/latest/about/algorithms/dpo.html
- `BinaryPreferenceDataset`: https://docs.nvidia.com/nemo/rl/nightly/apidocs/nemo_rl/nemo_rl.data.datasets.preference_datasets.binary_preference_dataset.html
- v0.5.0 default DPO config: https://github.com/NVIDIA-NeMo/RL/blob/v0.5.0/examples/configs/dpo.yaml
