# Eval

AIME 2024 / 2025 をはじめとしたベンチマークで `model/llm-jp-4-8b-thinking` の性能を評価する。
必要な指標は最低 pass@1/32/64、余裕があれば pass@8/16 も。

## 評価方法の選択

| 方法 | スクリプト | venv | 用途 |
|---|---|---|---|
| **verl `val_only`** | `val_aime_2024_2025.sh` | `.venv` | 学習中の val として使うパイプライン検証 |
| **OLMES** (推奨) | `olmes_aime.sh` | `.venv-eval` | OLMo3/DeepSeek 等と同条件の公式ベンチマーク |

OLMES は [AI2 の公式評価フレームワーク](https://github.com/allenai/olmes) で、OLMo3 Think や DeepSeek-R1-Distill などが同じ設定で公開スコアを出しているので、直接比較ができる。

---

## OLMES 初回セットアップ

olmes は `ai2-olmo-core`, `ai2-olmo`, `alpaca_eval`, `lm_eval`, `torch>=2.8,<2.9` など重い依存を持ち、verl と衝突するので **別 venv** に分離する。

```bash
# deps/olmes は既に clone 済み (無ければ: git clone https://github.com/allenai/olmes.git deps/olmes)
cd deps/olmes
uv venv ../../.venv-eval --python 3.12
source ../../.venv-eval/bin/activate
uv sync --group gpu    # vllm==0.11.0 + xformers 込み
cd ../..

# 動作確認 (dry-run でタスク config を表示)
olmes --model model/llm-jp-4-8b-thinking --model-type vllm \
      --task aime:zs_cot_r1::pass_at_32_2024_deepseek --dry-run
```

## OLMES で AIME 24/25 を評価

```bash
sbatch exp/eval/olmes_aime.sh
```

- デフォルト: n=64 サンプリング, pass@{1,8,16,32,64} 算出
- DeepSeek 流儀: temp=0.6, top_p=0.95, max_gen_toks=16384
- 出力: `outputs/eval/olmes-aime-<timestamp>/`

### パラメータ変更

```bash
MODEL_PATH=model/other-model \
OUTPUT_DIR=outputs/eval/custom \
sbatch exp/eval/olmes_aime.sh
```

pass@k を変えたい場合は `olmes_aime.sh` の `REPEATS` / `PASS_AT_KS` を編集。

### 他ベンチマークの追加

OLMES の `olmo3:adapt` スイートには AIME 以外に MATH / GPQA / BBH / HumanEval+ / MBPP+ 等が含まれる:

```bash
olmes --model ... --task olmo3:adapt --output-dir ...
```

タスク一覧: `deps/olmes/oe_eval/configs/tasks.py`, スイート: `deps/olmes/oe_eval/configs/task_suites.py`

---

## verl val_only (参考)

```bash
sbatch exp/eval/val_aime_2024_2025.sh
```

学習パイプラインと同一環境で動かすぶんには便利。ただし pass@k を変えるたびに n を変更して再生成が必要で、FSDP actor を立ち上げるためオーバーヘッドが大きい。
