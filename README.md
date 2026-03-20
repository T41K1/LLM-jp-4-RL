# 0316_llm-jp-4-rl

LLM-JP v4 モデルの GRPO (Group Relative Policy Optimization) による強化学習実験。
verl + vLLM + FSDP を使用。

## 環境構築

### 前提条件

- Python 3.12
- [uv](https://docs.astral.sh/uv/) がインストール済み
- CUDA 対応 GPU ノード

### セットアップ

```bash
# 1. リポジトリをクローン
git clone <repository_url>
cd 0316_llm-jp-4-rl

# 2. verl を deps/ に配置
mkdir -p deps
git clone https://github.com/volcengine/verl.git deps/verl

# 3. 仮想環境の作成と依存関係のインストール
uv sync
```

### データの準備

```bash
# GSM8K データセットのダウンロードと前処理（verl 形式の parquet に変換）
uv run python deps/verl/examples/data_preprocess/gsm8k.py

# ~/data/gsm8k/{train,test}.parquet が生成される
```

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

## ディレクトリ構成

```
.
├── deps/verl/          # verl（editable install、git管理外）
├── exp/                # 実験用ジョブスクリプト
│   ├── step1/          #   Qwen3-8B GRPO
│   └── step2/          #   LLM-JP v4-8B GRPO
├── data_load/          # データダウンロードスクリプト
├── logs/               # PBS ジョブログ（git管理外）
├── checkpoints/        # 学習チェックポイント（git管理外）
├── pyproject.toml      # 依存関係定義
└── uv.lock             # 依存関係のロックファイル
```
