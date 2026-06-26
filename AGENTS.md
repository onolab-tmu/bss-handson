# Repository Guidelines

## プロジェクト構成とモジュール配置

このリポジトリは，音響ブラインド音源分離実験のための `uv` 管理 Python パッケージである．
再利用可能な処理は `src/bss_handson/` に置く．
STFT/ISTFT，室内混合，AuxIVA，評価，スペクトログラム保存，JSON/WAV 保存，共通 pipeline などが対象である．
標準実験の入口，条件 sweep，集計，可視化，random evaluation など，具体的な出力先や glob に依存する処理は `scripts/` に置く．
ステップごとの参照実装は `scripts/steps/` に置く．
標準設定は `configs/bss.yaml` に集約し，条件差分は OmegaConf の command-line override と実行後に保存される `results/<experiment>/config.yaml` で管理する．
Matplotlib style は `styles/`，ダウンロードした CMU ARCTIC データは `data/`，生成結果は `results/` に置く．

## ビルド・テスト・開発コマンド

- `uv sync`: `pyproject.toml` と `uv.lock` に基づいて依存関係を同期する．
- `uv run bss-handson --config configs/bss.yaml`: 標準設定で 2 音源 2 マイクの実験を実行し，設定された `results/` 配下へ結果を書き出す．
- `uv run bss-handson --config configs/bss.yaml auxiva.n_iter=10 output_dir=results/bss_niter_10`: OmegaConf の command-line override で一部の設定だけを変更して実行する．
- `uv run python -m py_compile src/bss_handson/*.py scripts/*.py scripts/steps/*.py`: Python ファイルの構文を確認する．
- `uv run ruff check`: Python コードの静的検査を実行する．
- `uv build`: `uv_build` backend でパッケージをビルドする．

## コーディングスタイルと命名規則

Python 3.11 以上を前提とする．
import して再利用できる処理は `src/bss_handson/` にまとめ，`scripts/` から利用する一方向の依存にする．
`src/` には，入力と出力が明確な単機能の処理，または複数の実験で共有する処理を置く．
`scripts/` には，`src/` の関数を組み合わせる実験入口，条件 sweep，集計，可視化を置く．
スクリプト同士の import は避け，共有したい処理は `src/bss_handson/` に切り出す．
インデントは 4 spaces，module・function・variable は snake_case を使う．
NumPy 配列を扱う関数では，入力と出力の shape が読み取れる名前や docstring を優先する．
スクリプト名は `plot_bss_metrics.py` のように動作が分かる名前にする．
設定ファイルは原則として `configs/bss.yaml` の 1 つに保ち，比較条件ごとの YAML を増やさない．

## テスト方針

現時点では専用の test suite はない．
変更時は，少なくとも構文確認と Ruff を実行する．

```bash
uv run python -m py_compile src/bss_handson/*.py scripts/*.py scripts/steps/*.py
uv run ruff check
```

挙動に関わる変更では，小さい反復回数で smoke 実行する．

```bash
uv run bss-handson --config configs/bss.yaml auxiva.n_iter=1 output_dir=results/smoke
```

テストを追加する場合は `tests/` 配下に置き，ファイル名は `test_*.py` とする．
対象は config parsing，STFT/ISTFT の往復，評価補助関数，AuxIVA の shape と finite value の確認など，決定的に検証できる単位を優先する．

## コミットと Pull Request

commit message は commitizen スタイルを使う．
形式は `<type>: <subject>` とし，subject は英語の短い命令形で書く．
主な type は `feat`，`fix`，`docs`，`refactor`，`test`，`chore` を使う．
例は `docs: refine handson workflow`，`refactor: extract separation pipeline`，`fix: handle empty iteration metrics` である．
破壊的変更がある場合は `type!:` を使い，body に移行方法を書く．

Pull Request には，変更した実験または code path，実行したコマンド，追加・変更した `configs/`，`scripts/`，`styles/` のファイルを記載する．
挙動や結果が変わる場合は，主要な metric や plot も添える．

## データと生成物

`data/` と `results/` はローカル生成物として扱う．
ダウンロードした音声データ，WAV，plot，大きな metric dump は，明示的に curated example として扱う場合を除き commit しない．
`__pycache__/`，`.ruff_cache/`，`.venv/`，`.DS_Store` も commit しない．
