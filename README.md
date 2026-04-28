# 0316_llm-jp-4-rl

LLM-JP v4 モデルの GRPO (Group Relative Policy Optimization) による強化学習実験。
verl + vLLM + FSDP を使用。

`deps/verl/` は [T41K1/LLM-jp-4-verl](https://github.com/T41K1/LLM-jp-4-verl) をsubmoduleとして取り込んでおり、本家 [volcengine/verl](https://github.com/volcengine/verl) に独自パッチを当てた状態で利用します。

## 環境構築

### 前提条件

- Python 3.12
- [uv](https://docs.astral.sh/uv/) がインストール済み
- CUDA 対応 GPU ノード

### セットアップ

```bash
# 1. リポジトリを submodule ごとクローン
git clone --recurse-submodules https://github.com/T41K1/LLM-jp-4-RL.git
cd LLM-jp-4-RL

# すでに submodule なしで clone 済みの場合
git submodule update --init --recursive

# 2. 仮想環境の作成と依存関係のインストール
uv sync
```

`deps/verl/` には submodule で fork (`T41K1/LLM-jp-4-verl`) の指定 commit が自動で展開されます。手動で `git clone` する必要はありません。

### モデルの準備

学習・評価に使うモデルは `model/` に配置します（`.gitignore` で除外されているため自分で取得が必要）。Hugging Face Hub から `huggingface-cli` または `hf` で取得します。

```bash
# 例: Qwen3-8B
hf download Qwen/Qwen3-8B --local-dir model/Qwen3-8B

# 例: LLM-JP v4 8B / 32B (Hub に上がっている場合)
hf download llm-jp/llm-jp-4-8b-thinking  --local-dir model/llm-jp-4-8b-thinking
hf download llm-jp/llm-jp-4-32b-a3b-thinking --local-dir model/llm-jp-4-32b-a3b-thinking
```

ジョブスクリプトでは `actor_rollout_ref.model.path` に上記ローカルパス（または HF ID 直指定）を渡します。クラスタ共有ストレージ上の checkpoint を直接指す場合もあるので、各 `exp/stepN/*.sh` の `MODEL_PATH` を実環境に合わせて書き換えてください。

`model/` の標準レイアウト:

```
model/
├── Qwen3-8B/
├── llm-jp-4-8b-thinking/
└── llm-jp-4-32b-a3b-thinking/
```

### データの準備

学習・評価データは `data/` 配下に **verl の parquet 形式** で配置します。`data_load/` のスクリプトを使ってダウンロード+前処理します。

```bash
# AIME 2024 / 2025 (評価用)
uv run python data_load/aime2024.py --save_dir data/AIME2024
uv run python data_load/aime2025.py --save_dir data/AIME2025

# Dolci-Think-RL-7B の math サブセット (学習用)
uv run python data_load/dolci_think_rl_7b_math.py --save_dir data/Dolci-Think-RL-7B-math

# 任意の Hugging Face データセットを parquet として落とす汎用スクリプト
uv run python data_load/download.py openai/gsm8k --config main --save_dir data/gsm8k
```

verl の sample preprocessor を使う場合（GSM8K など）:

```bash
uv run python deps/verl/examples/data_preprocess/gsm8k.py
# ~/data/gsm8k/{train,test}.parquet が生成される
```

各 parquet は verl 標準スキーマ (`data_source / prompt / ability / reward_model / extra_info`) に揃えてあります。

### 環境変数

```bash
# .env ファイルを作成（.gitignore に含まれているため手動で作成が必要）
echo 'export WANDB_API_KEY=<your_key>' > .env
```

## 実験の実行

### Step 1: Qwen3-8B での GRPO 学習

```bash
# ABCI で PBS ジョブとして投入（1ノード・8GPU）
qsub exp/step1/run_qwen3-8b.sh
```

ジョブスクリプト内で以下の環境変数を設定しています：

| 環境変数 | 値 | 説明 |
|---|---|---|
| `VLLM_USE_V1` | `1` | vLLM v1 エンジンを使用 |
| `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO` | `0` | Ray が GPU 可視性を上書きするのを防止 |

### Step 3: LLM-JP v4 8B Instruct で OLMo3 流の math GRPO

`MathVerifier` (`rewards/math_reward.py`) を使った GRPO 学習。学習データは `data/Dolci-Think-RL-math-verl/`、検証データは `data/AIME2024/` を使用します。

```bash
# ABCI で PBS ジョブとして投入（1ノード・8GPU、walltime 50h）
qsub exp/step3/run_llmjp-4-8b-instruct-dev_phase1_olomo3_math.sh
```

スクリプト先頭の `MODEL_PATH` をローカルの checkpoint に書き換えてから投入してください。後継版 `..._v2.sh` も同ディレクトリにあります。

## verl 改造のワークフロー

`deps/verl/` は独立した git リポジトリ (submodule) で、`origin = T41K1/LLM-jp-4-verl`、`upstream = volcengine/verl` を指しています。改造は fork 側に push し、親リポは「どの commit を使っているか」を pointer で記録する運用です。

### 改造を加えて反映させる手順

```bash
# 1. submodule 内で編集
cd deps/verl
git checkout main          # detached HEAD を避けるため必ず実行
# ... ファイル編集 ...
git add <files>
git commit -m "feat: ..."

# 2. fork に push
git push origin main

# 3. 親リポに戻って submodule pointer を更新
cd ../..
git add deps/verl
git commit -m "chore(verl): bump submodule"
git push
```

⚠️ submodule 側を push し忘れると、他の人が pull したとき pointer が指す commit が存在せず壊れます。安全装置として一度だけ:

```bash
git config push.recurseSubmodules check
```

を設定しておくと、submodule 側に未 push の commit があった場合に親リポの push がエラーで止まります。

### upstream (volcengine/verl) の更新を取り込む

```bash
cd deps/verl
git fetch upstream
git checkout main
git merge upstream/main      # または git rebase upstream/main
git push origin main
cd ../..
git add deps/verl
git commit -m "chore(verl): sync with upstream"
```

### 別の fork を使いたい場合（ローカル上書き）

`.gitmodules` を変更せず、自分のローカルだけ別の fork に向けることも可能です:

```bash
git config submodule.deps/verl.url https://github.com/<your_account>/verl.git
git submodule sync deps/verl
git submodule update --init deps/verl
```

## ディレクトリ構成

```
.
├── deps/verl/          # verl submodule (T41K1/LLM-jp-4-verl)
├── exp/                # 実験用ジョブスクリプト
│   ├── step1/          #   Qwen3-8B GRPO
│   ├── step2/          #   LLM-JP v4-8B GRPO
│   ├── step3/          #   後続ステップ
│   ├── eval/           #   評価ジョブ
│   └── ex/             #   その他試験的な実行
├── data_load/          # データダウンロード/前処理スクリプト
├── data/               # 学習・評価用 parquet（git管理外）
├── model/              # HF モデル checkpoint（git管理外）
├── scripts/            # 補助スクリプト
├── logs/               # PBS ジョブログ（git管理外）
├── checkpoints/        # 学習チェックポイント（git管理外）
├── pyproject.toml      # 依存関係定義
└── uv.lock             # 依存関係のロックファイル
```
