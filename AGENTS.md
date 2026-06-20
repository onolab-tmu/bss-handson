# Repository Guidelines

## プロジェクト構成とモジュール配置

このリポジトリは，音響ブラインド音源分離実験のための `uv` 管理 Pythonパッケージである。
再利用可能な処理は `src/bss_handson/` に置く。CLI，STFT，室内シミュレーション，AuxIVA，評価，描画補助などが対象である。
実験手順，条件 sweep，集計，可視化など，特定の出力先や glob に依存する処理は `scripts/` に置く。
実験設定は `configs/`，Matplotlib style は `styles/`，ダウンロードした CMU ARCTIC データは `data/`，生成結果は `results/` に置く。

## ビルド・テスト・開発コマンド

- `uv sync`: `pyproject.toml` と `uv.lock` に基づいて依存関係を同期する。
- `uv run bss-handson --config configs/bss.yaml`: 標準設定で 2 音源 2 マイクの
  実験を実行し，設定された `results/` 配下へ結果を書き出す。
- `uv run bss-handson --config configs/bss.yaml auxiva.n_iter=10 output_dir=results/bss_niter_10`:
  OmegaConf の command-line override で一部の設定だけを変更して実行する。
- `uv build`: `uv_build` backend でパッケージをビルドする。

## コーディングスタイルと命名規則

Python 3.11 以上を前提とする。import して再利用できる処理は `src/bss_handson/` にまとめ，`scripts/` から利用する一方向の依存にする。
インデントは 4 spaces，module・function・variable は snake_case を使う。
NumPy 配列を扱う関数では，入力と出力の shape が読み取れる名前や docstring を優先する。
スクリプト名は `plot_bss_metrics.py` のように動作が分かる名前にし，設定ファイルは `bss_model_gauss.yaml` や `bss_niter_50.yaml` のように条件名を含める。

## テスト方針

現時点では専用の test suite はない。変更時は，少なくとも小さい反復回数で smoke 実行する。

```bash
uv run bss-handson --config configs/bss.yaml auxiva.n_iter=1 output_dir=results/smoke
```

テストを追加する場合は `tests/` 配下に置き，ファイル名は `test_*.py` とする。
対象は config parsing，STFT/ISTFT の往復，評価補助関数，AuxIVA の shape とfinite value の確認など，決定的に検証できる単位を優先する。

## コミットと Pull Request

このリポジトリにはまだ commit 履歴がないため，既存の convention は推定しない。
commit subject は `Add AuxIVA smoke test` や `Refine BSS comparison plots` のように，短い命令形で書く。
Pull Request には，変更した実験または code path，実行したコマンド，追加・変更した `configs/`，`scripts/`，`styles/` のファイルを記載する。
挙動や結果が変わる場合は，主要な metric や plot も添える。

## データと生成物

`data/` と `results/` はローカル生成物として扱う。ダウンロードした音声データ， WAV，plot，大きな metric dump は，明示的に curated example として扱う場合を除き commit しない。
