# bss-handson

ブラインド音源分離の実験管理を学ぶためのハンズオン用プロジェクトである．
CMU ARCTIC の音声を pyroomacoustics で室内混合し，NumPy で実装した AuxIVA により音源分離を行い，fast-bss-eval で性能評価する．

このプロジェクトは，次の流れを 1 つの Python CLI として実行する．

1. CMU ARCTIC からクリーン音声を読み込む
2. pyroomacoustics で N 音源 N マイクの determined BSS 条件を作る
3. scipy.signal.ShortTimeFFT でマルチチャンネル STFT を計算する
4. NumPy 実装の AuxIVA で分離する
5. projection back により分離信号のスケールを補正する
6. ISTFT で時間波形へ戻す
7. fast-bss-eval で SDR, SIR, SAR を計算する
8. wav, spectrogram, config, metrics を `results/` に保存する

## 前提

この README のコマンドは，すべて `bss-handson/` ディレクトリ内で実行する．
`uv` は実行するディレクトリにある `pyproject.toml` を見て project を判断するため，別ディレクトリで実行すると依存関係や CLI entry point の解決結果が変わる．

```bash
cd bss-handson
```

Python project は `uv init --lib` 相当の構成で作られている．
`src/bss_handson/` 以下が import される package 本体であり，`pyproject.toml` の `[project.scripts]` により `bss-handson` コマンドを登録している．
CLI を `uv run bss-handson ...` として実行するには，`[build-system]` が必要である．

## セットアップ

依存関係は `pyproject.toml` と `uv.lock` で管理する．
初回は次を実行する．

```bash
uv sync
```

主な依存関係は以下である．

| 用途                 | package             |
| -------------------- | ------------------- |
| 行列計算             | NumPy               |
| STFT / ISTFT         | SciPy               |
| 音声ファイル I/O     | soundfile           |
| 音響シミュレーション | pyroomacoustics     |
| 音源分離評価         | fast-bss-eval       |
| 設定管理             | OmegaConf           |
| 可視化               | Matplotlib, seaborn |

## ディレクトリ構成

```txt
bss-handson/
├── configs/
│   ├── bss.yaml
│   ├── bss_model_*.yaml
│   ├── bss_niter_*.yaml
│   ├── bss_refmic_*.yaml
│   └── bss_update_*.yaml
├── data/
├── results/
├── scripts/
│   ├── create_bss_comparison_configs.py
│   ├── evaluate_random_bss.py
│   ├── plot_bss_metrics.py
│   ├── plot_bss_niter_metrics.py
│   ├── run_bss.py
│   └── show_bss_metrics.py
├── src/
│   └── bss_handson/
│       ├── __init__.py
│       ├── auxiva.py
│       ├── cli.py
│       ├── config.py
│       ├── data.py
│       ├── evaluation.py
│       ├── plot.py
│       ├── plot_style.py
│       ├── simulation.py
│       └── stft.py
├── styles/
│   ├── common.mplstyle
│   ├── paper.mplstyle
│   └── slide.mplstyle
├── pyproject.toml
└── uv.lock
```

`data/` には CMU ARCTIC の音声データが保存される．
`results/` には実験結果が保存される．
これらは実験の入出力であり，通常は Git 管理しない．

`src/bss_handson/` には，入力と出力の意味が明確で，他の実験から import して再利用できる処理を置く．
たとえば，STFT，room simulation，AuxIVA，評価指標，spectrogram 保存のような単機能の処理である．
`scripts/` には，それらの処理を組み合わせた実験手順，条件 sweep，集計，可視化を置く．
今回の実験名，`results/` の具体的な出力先，glob pattern，複数条件を回す for loop に依存するコードは `scripts/` に置く．
依存関係は常に `scripts/` から `src/bss_handson/` へ向け，`src/bss_handson/` から `scripts/` は import しない．

## 標準設定

標準設定は `configs/bss.yaml` にある．
この設定では 2 音源 2 マイクの混合シミュレーションを行う．

```yaml
output_dir: results/bss_example
seed: 0

dataset:
  basedir: data/cmu_arctic
  speakers: [bdl, slt]
  utterance_indices: [0, 1]

stft:
  window: hann
  win_length: 1024
  hop: 256

room:
  fs: 16000
  size: [6.0, 4.0]
  rt60: 0.30
  source_positions:
    - [1.5, 1.0]
    - [4.5, 3.0]
  mic_positions:
    - [3.0, 1.975]
    - [3.0, 2.025]

auxiva:
  n_iter: 50
  model: laplace
  update_method: ip
  eps: 1.0e-10
  reference_mic: 0

plot:
  style:
    - styles/common.mplstyle
    - styles/paper.mplstyle
  vmin_db: -80
  vmax_db: 0
```

`auxiva.model` は `laplace` または `gauss` を指定する．
`auxiva.update_method` は `ip` または `iss` を指定する．
`plot.style` は Matplotlib style file のリストであり，共通設定と媒体別設定を順に読み込む．

## 実行

標準設定で 1 回実行する．

```bash
uv run bss-handson --config configs/bss.yaml
```

`uv run bss-handson` は互換性のための entry point であり，内部では `scripts/run_bss.py` を実行する．
このプロジェクトでは，`src/bss_handson/` には単機能の処理を置き，データ取得，シミュレーション，分離，評価，保存を組み合わせる複雑な実験手順は `scripts/run_bss.py` に置く．

初回実行時には，pyroomacoustics の dataset wrapper により CMU ARCTIC が `data/cmu_arctic/` 以下へダウンロードされる．

実行後，`results/bss_example/` に次のようなファイルが保存される．

```txt
results/bss_example/
├── config.yaml
├── estimated_0.wav
├── estimated_1.wav
├── estimated_spectrogram_0.png
├── estimated_spectrogram_1.png
├── iteration_metrics.json
├── metrics.json
├── mixture_0.wav
├── mixture_1.wav
├── mixture_spectrogram_0.png
├── mixture_spectrogram_1.png
├── source_0.wav
└── source_1.wav
```

`config.yaml` は command line override を反映した最終設定である．
`metrics.json` は最終反復後の SDR, SIR, SAR, permutation を保存する．
`iteration_metrics.json` は 0, 10, 20, 30, 40, 50 回目の分離性能を保存する．

## パラメータ変更

OmegaConf の dot-list override により，設定ファイルを書き換えずに一部の値を command line から上書きできる．

```bash
uv run bss-handson --config configs/bss.yaml auxiva.n_iter=10 output_dir=results/bss_niter_10
```

複数条件を比較するときは，条件系列ごとに `output_dir` の prefix をそろえる．

```bash
for n_iter in 10 50 100
do
  uv run bss-handson --config configs/bss.yaml auxiva.n_iter="$n_iter" output_dir="results/bss_niter_${n_iter}"
done

for model in laplace gauss
do
  uv run bss-handson --config configs/bss.yaml auxiva.model="$model" output_dir="results/bss_model_${model}"
done

for update_method in ip iss
do
  uv run bss-handson --config configs/bss.yaml auxiva.update_method="$update_method" output_dir="results/bss_update_${update_method}"
done

for reference_mic in 0 1
do
  uv run bss-handson --config configs/bss.yaml auxiva.reference_mic="$reference_mic" output_dir="results/bss_refmic_${reference_mic}"
done
```

list を上書きする場合は shell に解釈されないように引用符で囲む．

```bash
uv run bss-handson --config configs/bss.yaml dataset.speakers='[bdl,slt]' dataset.utterance_indices='[2,3]' output_dir=results/bss_utt_2_3
```

## 評価結果の確認

`scripts/show_bss_metrics.py` は，指定した glob pattern に一致する `metrics.json` だけを表示する．
何も区別せずに全結果を集計すると比較の意味が曖昧になるため，集計したい条件系列を `--pattern` で明示する．

```bash
uv run python scripts/show_bss_metrics.py --pattern "bss_niter_*/metrics.json"
uv run python scripts/show_bss_metrics.py --pattern "bss_model_*/metrics.json"
uv run python scripts/show_bss_metrics.py --pattern "bss_update_*/metrics.json"
uv run python scripts/show_bss_metrics.py --pattern "bss_refmic_*/metrics.json"
```

IP と ISS の比較結果だけを見る例は次である．

```bash
uv run python scripts/show_bss_metrics.py --pattern "bss_update_*/metrics.json"
```

## 可視化

条件ごとの SDR を棒グラフにする．
ここでも `--pattern` により，比較したい条件系列だけを読み込む．

```bash
uv run python scripts/plot_bss_metrics.py --pattern "bss_model_*/metrics.json" --output results/metrics_model_summary.png
uv run python scripts/plot_bss_metrics.py --pattern "bss_update_*/metrics.json" --output results/metrics_update_method.png
```

プレゼン向けの style を使う場合は，`styles/common.mplstyle` と `styles/slide.mplstyle` を読み込む．

```bash
uv run python scripts/plot_bss_metrics.py \
  --pattern "bss_update_*/metrics.json" \
  --style styles/common.mplstyle styles/slide.mplstyle \
  --output results/metrics_update_method_slide.png
```

反復回数に対する分離性能を見る場合は，`iteration_metrics.json` を使う．
このファイルは 1 回の AuxIVA 実行中に callback で途中評価した結果であり，0, 10, 20, 30, 40, 50 回目の 6 点を含む．

```bash
uv run python scripts/plot_bss_niter_metrics.py \
  --metrics-path results/bss_example/iteration_metrics.json \
  --output results/metrics_by_niter.png
```

## Matplotlib style file

`styles/` には共通設定と媒体別設定を分けて置いている．

| file                     | 用途                                         |
| ------------------------ | -------------------------------------------- |
| `styles/common.mplstyle` | tick を内向きにし，grid を破線にする共通設定 |
| `styles/paper.mplstyle`  | 論文向けの図サイズ，セリフ体，10 pt          |
| `styles/slide.mplstyle`  | スライド向けの図サイズ，サンセリフ体，24 pt  |

`plt.style.use()` は style file のリストを受け取れるため，共通設定と媒体別設定を次のように重ねて使う．

```bash
uv run bss-handson --config configs/bss.yaml \
  plot.style='[styles/common.mplstyle,styles/slide.mplstyle]' \
  output_dir=results/bss_slide_style
```

## Random evaluation

音源信号と音源位置をランダムに変えながら複数 trial の平均性能を見る場合は，`scripts/evaluate_random_bss.py` を使う．

```bash
uv run python scripts/evaluate_random_bss.py \
  --config configs/bss.yaml \
  --output-dir results/random_eval \
  --n-trials 20 \
  --seed 0 \
  --max-utterance-index 99
```

出力は次である．

```txt
results/random_eval/
├── random_average_metrics.csv
└── random_average_metrics.json
```

`random_average_metrics.json` には，各 trial の条件，評価値，SDR/SIR/SAR の平均と標準偏差が保存される．
`random_average_metrics.csv` は tidy data に近い縦長の表であり，pandas や seaborn で集計しやすい．

## AuxIVA 実装

AuxIVA は `src/bss_handson/auxiva.py` に NumPy で実装している．
pyroomacoustics の `pyroomacoustics.bss.auxiva` は呼び出していない．

主な関数は以下である．

| function                | 役割                                          |
| ----------------------- | --------------------------------------------- |
| `demix()`               | 分離行列を観測信号に適用する                  |
| `source_weights()`      | Laplace / Gauss model に基づく重みを計算する  |
| `weighted_covariance()` | IP 更新で使う重み付き共分散行列を計算する     |
| `objective()`           | 補助目的関数を計算する                        |
| `ip_update()`           | iterative projection による 1 回分の更新      |
| `iss_update()`          | iterative source steering による 1 回分の更新 |
| `project_back()`        | 参照マイクに合わせてスケールを補正する        |
| `separate()`            | AuxIVA の反復と projection back をまとめる    |

教育目的のため，実装は数式との対応を追いやすいように `for` 文を多めに使っている．
実験規模が大きい場合は，NumPy の broadcasting や batch matrix multiplication による高速化を検討する．

## Module の役割

| file            | 役割                                               |
| --------------- | -------------------------------------------------- |
| `config.py`     | OmegaConf による設定ファイルと override の読み込み |
| `data.py`       | CMU ARCTIC の取得と音源信号の整形                  |
| `simulation.py` | pyroomacoustics による室内混合                     |
| `stft.py`       | scipy.signal.ShortTimeFFT による STFT / ISTFT      |
| `auxiva.py`     | AuxIVA, IP/ISS 更新, projection back               |
| `evaluation.py` | fast-bss-eval による SDR/SIR/SAR 評価              |
| `plot.py`       | spectrogram の保存                                 |
| `plot_style.py` | Matplotlib axis style の適用                       |
| `cli.py`        | `scripts/run_bss.py` を呼ぶ最小限の entry point    |

`src/bss_handson/` は，1 つの意味のある入力に対して 1 つの結果を返す小さな処理を置く場所である．
`scripts/` は，それらの処理を組み合わせて実験実行，条件 sweep，集計，可視化を行う場所である．
そのため，標準 BSS 実験の本体は `scripts/run_bss.py` に置いている．

| script                             | 役割                                                |
| ---------------------------------- | --------------------------------------------------- |
| `run_bss.py`                       | 標準 BSS 実験の実行と結果保存                       |
| `evaluate_random_bss.py`           | 音源信号と音源位置をランダムに変えた複数 trial 評価 |
| `create_bss_comparison_configs.py` | 比較用 YAML の生成                                  |
| `show_bss_metrics.py`              | 条件系列ごとの `metrics.json` 表示                  |
| `plot_bss_metrics.py`              | 条件系列ごとの SDR 可視化                           |
| `plot_bss_niter_metrics.py`        | AuxIVA 反復回数と SDR の可視化                      |

## 開発時の確認

構文だけを確認する場合は次を実行する．

```bash
uv run python -m py_compile \
  src/bss_handson/auxiva.py \
  src/bss_handson/cli.py \
  scripts/run_bss.py \
  scripts/evaluate_random_bss.py \
  scripts/show_bss_metrics.py \
  scripts/plot_bss_metrics.py \
  scripts/plot_bss_niter_metrics.py
```

最小限の smoke run は，反復回数を小さくして別の出力先に保存する．

```bash
uv run bss-handson --config configs/bss.yaml auxiva.n_iter=1 output_dir=results/smoke
```

## Troubleshooting

### `Failed to spawn: bss-handson`

`uv run bss-handson` で `Failed to spawn: bss-handson` と表示される場合は，`pyproject.toml` のある `bss-handson/` の外で実行している可能性が高い．
`cd bss-handson` してから実行する．

また，`pyproject.toml` に `[project.scripts]` と `[build-system]` が必要である．
この project では次の設定により CLI command を登録している．

```toml
[project.scripts]
bss-handson = "bss_handson.cli:main"

[build-system]
requires = ["uv_build>=0.11.20,<0.12.0"]
build-backend = "uv_build"
```

### `.venv` を移動した場合

`.venv` を別ディレクトリから移動した場合，entry point が壊れることがある．
その場合は `bss-handson/` 内で `.venv` を削除し，`uv sync` で作り直す．

### CMU ARCTIC のダウンロード

初回実行時は `data/cmu_arctic/` にデータをダウンロードするため，ネットワーク接続が必要である．
一度ダウンロードされた後は，同じ `basedir` を使う限り再利用される．

### 結果が上書きされる場合

`output_dir` が同じ場合，既存の `metrics.json` や wav が上書きされる．
条件を変える実験では，`output_dir=results/bss_model_gauss` のように条件名を含める．
