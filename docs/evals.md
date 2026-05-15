# evals の整理

最終更新: 2026-05-14

## 概要

`evals/` には `llm-jp-eval v2.1.3` を使った評価実行の入力、生成結果、集計結果、実行ログがモデル別に保存されている。トップレベルの各ディレクトリは「モデル checkpoint + reasoning effort」の1実行に対応する。

主に集まっている情報は次の通り。

| パス | 内容 |
|---|---|
| `evals/<run>/config.json` | 評価対象モデル、出力先、PBS queue/group、GPU設定、`reasoning_effort` などの実行設定 |
| `evals/<run>/qsub.sh` | PBS 投入スクリプト。実行コマンドと環境変数を含むため、外部共有前にシークレット混入を確認する |
| `evals/<run>/llm-jp-eval/v2.1.3/prompts_*/` | 各ベンチマークの prompt JSON。57ベンチマーク分が生成されている |
| `evals/<run>/llm-jp-eval/v2.1.3/offline/` | vLLM 推論結果。完了済み実行では `<benchmark>.eval-generated.json` が57個ある |
| `evals/<run>/llm-jp-eval/v2.1.3/results/result.json` | 集計済みスコア、言語別/カテゴリ別スコア、time profile、出力例レコード |
| `evals/<run>/logs/` | `qsub.out`, `qsub.err`, `llm-jp-eval-v2.1.3.log`, `llm-jp-eval-v2.1.3.err` |

## 評価条件

現在の4実行は、同じ `llm-jp-eval v2.1.3` 設定で `reasoning_effort=low` と `reasoning_effort=medium` を比較する形になっている。

| 項目 | 値 |
|---|---|
| eval suite | `llm-jp-eval v2.1.3` |
| target split | `test` |
| few-shot | `4` |
| max samples | `100` |
| chat template | `apply_chat_template=true` |
| reasoning parser | `openai_gptoss` |
| generation | `n=1`, `temperature=1.0`, `top_p=1.0` |
| inference backend | `vllm0.11.2`, `H200`, `bfloat16`, `tensor_parallel_size=1` |
| PBS | `select=1`, `RTYPE=rt_HG`, queue `R9920261000`, group `gcg51557` |

## 実行一覧

| Alias | モデル checkpoint | Effort | 状態 | 収集済みファイル |
|---|---|---|---|---|
| `GRPO-Math-step300-low` | `hf_checkpoint/llm-jp-4-8b-thinking-GRPO-Olmo3-Math-2node-20260502-global_step_300` | `low` | 完了 | prompts 57, generated 57, `result.json` あり |
| `GRPO-Math-step300-medium` | `hf_checkpoint/llm-jp-4-8b-thinking-GRPO-Olmo3-Math-2node-20260502-global_step_300` | `medium` | 完了 | prompts 57, generated 57, `result.json` あり |
| `Nemo-DOLCI-DPO-step1154-low` | `hf_checkpoint/llm-jp-4-8b-nemo-rl-dolci-think-dpo-7b-multinode-seq12288-tp1-20260513-step_1154` | `low` | 未完了 | prompts 57, generated 0, `result.json` なし |
| `Nemo-DOLCI-DPO-step1154-medium` | `hf_checkpoint/llm-jp-4-8b-nemo-rl-dolci-think-dpo-7b-multinode-seq12288-tp1-20260513-step_1154` | `medium` | 未完了 | prompts 57, generated 0, `result.json` なし |

`Nemo-DOLCI-DPO-step1154-*` は prompt 生成後、tokenizer/model 読み込み時に停止している。ログ上の直接原因は checkpoint が custom code を要求しているのに `trust_remote_code=False` のまま実行されていることで、再実行時は tokenizer/model 側に `trust_remote_code=True` を渡す必要がある。

## 完了済みスコアの概要

以下のスコアは `GRPO-Math-step300` の `low` と `medium` の比較。`Nemo-DOLCI-DPO-step1154` は現時点でスコア未生成。

| Scope | low | medium |
|---|---:|---:|
| `AVG` | 0.587 | 0.619 |
| `JA AVG` | 0.582 | 0.616 |
| `EN AVG` | 0.643 | 0.682 |

## カテゴリ集計

`jsts` は dataset list には含まれているが、`result.json` の category 定義には入っていないため、下のカテゴリ集計には含まれない。ベンチマーク別表では、利用可能な指標のうち `pearson` を primary metric として載せている。

| Category | Description | Metric / note | Benchmarks | low | medium |
|---|---|---|---:|---:|---:|
| `NLI` | Natural Language Inference | `exact_match` | 5 | 0.830 | 0.812 |
| `QA` | Question Answering | `exact_match`; overrides: `triviaqa=triviaqa_f1`, `drop=drop_f1` | 6 | 0.579 | 0.615 |
| `RC` | Reading Comprehension | `exact_match` | 1 | 0.620 | 0.690 |
| `CR` | Commonsense Reasoning | `exact_match` | 4 | 0.863 | 0.860 |
| `HE-JA` | Human Evaluation | `exact_match`; override: `jhle=hle_exact_match` | 7 | 0.507 | 0.531 |
| `HE-EN` | Human Evaluation | `exact_match`; override: `hle=hle_exact_match` | 7 | 0.543 | 0.581 |
| `EL` | Entity Linking | `set_f1` | 1 | 0.573 | 0.582 |
| `FA` | Fine-grained Analysis | `set_f1`; override: `wiki_reading=char_f1` | 5 | 0.263 | 0.284 |
| `MR` | Math Reasoning | `mathematical_equivalence`; overrides: `polymath-en=polymath_weighted_accuracy`, `polymath-ja=polymath_weighted_accuracy` | 7 | 0.546 | 0.619 |
| `MT` | Machine Translation | `comet_wmt22` | 4 | 0.836 | 0.836 |
| `CG` | Code Generation | `code_exec_sandbox` | 2 | 0.865 | 0.920 |
| `SUM` | Summarization | `rouge2_scaling` | 1 | 0.006 | 0.050 |
| `IF` | Instruction Following | `mifeval_strict` | 2 | 0.455 | 0.525 |
| `BBH` | BIG-Bench Hard | `exact_match` | 4 | 0.728 | 0.758 |

## ベンチマーク別スコア

`low` と `medium` は `GRPO-Math-step300` の primary metric。`Delta` は `medium - low`。

| Category | Benchmark | Primary metric | low | medium | Delta |
|---|---|---|---:|---:|---:|
| `MR` | `aime2024` | `mathematical_equivalence` | 0.367 | 0.467 | 0.100 |
| `MR` | `aime2025` | `mathematical_equivalence` | 0.333 | 0.533 | 0.200 |
| `QA` | `aio` | `exact_match` | 0.660 | 0.690 | 0.030 |
| `MT` | `alt-j-to-e` | `comet_wmt22` | 0.888 | 0.885 | -0.003 |
| `MT` | `alt-e-to-j` | `comet_wmt22` | 0.907 | 0.913 | 0.006 |
| `BBH` | `bigbenchhard_direct` | `exact_match` | 0.830 | 0.790 | -0.040 |
| `BBH` | `bigbenchhard_cot` | `exact_match` | 0.740 | 0.780 | 0.040 |
| `BBH` | `bigbenchhard_ja_direct` | `exact_match` | 0.680 | 0.750 | 0.070 |
| `BBH` | `bigbenchhard_ja_cot` | `exact_match` | 0.660 | 0.710 | 0.050 |
| `EL` | `chabsa` | `set_f1` | 0.573 | 0.582 | 0.009 |
| `CR` | `commonsensemoralja` | `exact_match` | 0.920 | 0.900 | -0.020 |
| `QA` | `drop` | `drop_f1` | 0.830 | 0.888 | 0.057 |
| `MR` | `gsm8k` | `mathematical_equivalence` | 0.940 | 0.960 | 0.020 |
| `HE-EN` | `gpqa_diamond_en` | `exact_match` | 0.460 | 0.550 | 0.090 |
| `HE-EN` | `gpqa_extended_en` | `exact_match` | 0.420 | 0.430 | 0.010 |
| `HE-EN` | `gpqa_main_en` | `exact_match` | 0.420 | 0.480 | 0.060 |
| `HE-JA` | `gpqa_diamond_ja` | `exact_match` | 0.480 | 0.520 | 0.040 |
| `HE-JA` | `gpqa_extended_ja` | `exact_match` | 0.340 | 0.500 | 0.160 |
| `HE-JA` | `gpqa_main_ja` | `exact_match` | 0.430 | 0.390 | -0.040 |
| `QA` | `jamc-qa` | `exact_match` | 0.570 | 0.570 | 0.000 |
| `NLI` | `jamp` | `exact_match` | 0.710 | 0.730 | 0.020 |
| `NLI` | `janli` | `exact_match` | 0.990 | 0.990 | 0.000 |
| `CR` | `jcommonsenseqa` | `exact_match` | 0.990 | 0.980 | -0.010 |
| `QA` | `jemhopqa` | `exact_match` | 0.540 | 0.600 | 0.060 |
| `CG` | `jhumaneval` | `code_exec_sandbox` | 0.890 | 0.920 | 0.030 |
| `HE-JA` | `jmmlu` | `exact_match` | 0.780 | 0.780 | 0.000 |
| `NLI` | `jnli` | `exact_match` | 0.830 | 0.790 | -0.040 |
| `NLI` | `jsem` | `exact_match` | 0.790 | 0.770 | -0.020 |
| `NLI` | `jsick` | `exact_match` | 0.830 | 0.780 | -0.050 |
| `RC` | `jsquad` | `exact_match` | 0.620 | 0.690 | 0.070 |
| `Uncategorized` | `jsts` | `pearson` | 0.910 | 0.899 | -0.011 |
| `CR` | `kuci` | `exact_match` | 0.770 | 0.740 | -0.030 |
| `MR` | `mawps` | `mathematical_equivalence` | 0.970 | 0.980 | 0.010 |
| `CG` | `mbpp` | `code_exec_sandbox` | 0.840 | 0.920 | 0.080 |
| `MR` | `mgsm` | `mathematical_equivalence` | 0.810 | 0.890 | 0.080 |
| `HE-EN` | `mmlu_en` | `exact_match` | 0.840 | 0.860 | 0.020 |
| `HE-JA` | `mmlu_prox_ja` | `exact_match` | 0.680 | 0.720 | 0.040 |
| `HE-EN` | `mmlu_prox_en` | `exact_match` | 0.680 | 0.770 | 0.090 |
| `IF` | `mif_eval_ja` | `mifeval_strict` | 0.330 | 0.460 | 0.130 |
| `IF` | `mif_eval_en` | `mifeval_strict` | 0.580 | 0.590 | 0.010 |
| `HE-JA` | `mmmlu` | `exact_match` | 0.780 | 0.730 | -0.050 |
| `QA` | `niilc` | `exact_match` | 0.260 | 0.320 | 0.060 |
| `HE-EN` | `openbookqa` | `exact_match` | 0.950 | 0.950 | 0.000 |
| `MR` | `polymath-en` | `polymath_weighted_accuracy` | 0.189 | 0.279 | 0.090 |
| `MR` | `polymath-ja` | `polymath_weighted_accuracy` | 0.210 | 0.225 | 0.015 |
| `QA` | `triviaqa` | `triviaqa_f1` | 0.614 | 0.621 | 0.007 |
| `CR` | `winogrande_xl` | `exact_match` | 0.770 | 0.820 | 0.050 |
| `FA` | `wiki_coreference` | `set_f1` | 0.070 | 0.084 | 0.014 |
| `FA` | `wiki_dependency` | `set_f1` | 0.182 | 0.241 | 0.059 |
| `FA` | `wiki_ner` | `set_f1` | 0.160 | 0.180 | 0.020 |
| `FA` | `wiki_pas` | `set_f1` | 0.071 | 0.079 | 0.008 |
| `FA` | `wiki_reading` | `char_f1` | 0.835 | 0.836 | 0.001 |
| `MT` | `wikicorpus-j-to-e` | `comet_wmt22` | 0.739 | 0.737 | -0.002 |
| `MT` | `wikicorpus-e-to-j` | `comet_wmt22` | 0.812 | 0.810 | -0.002 |
| `SUM` | `xlsum_ja` | `rouge2_scaling` | 0.006 | 0.050 | 0.044 |
| `HE-EN` | `hle` | `hle_exact_match` | 0.030 | 0.030 | 0.000 |
| `HE-JA` | `jhle` | `hle_exact_match` | 0.060 | 0.080 | 0.020 |

## raw data の見方

`result.json` のトップレベルには `config`, `metadata`, `evaluation`, `records`, `scores`, `lang_scores` がある。主に見るのは次のキー。

| Key | 内容 |
|---|---|
| `scores` | ベンチマーク別 metric とカテゴリ集計。`AVG` もここにある |
| `lang_scores` | `JA` / `EN` 別のカテゴリスコアと平均 |
| `evaluation.time_profile` | ベンチマークごとの評価処理時間 |
| `records` | 出力例。設定上 `output_top_n=5` なので、57ベンチマーク x 5件 = 285件 |
| `metadata.generation_config` | vLLM sampling parameter |

各 `<benchmark>.eval-generated.json` には、その benchmark の `samples`、`input`、`prompt`、生成本文 `generated`、正解 `gold`、`reasoning_content` が残る。評価後の抽出済み `pred` / `true` / `exact` などは `result.json` の `records` に入る。詳細な生成内容の確認は `offline/<run_name>/<benchmark>.eval-generated.json`、スコアと抽出後の出力例確認は `results/result.json` を見る。
