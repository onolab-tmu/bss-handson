# これから音源分離の研究を始める人のための実験管理

## 目標

この資料では，Python を用いた音響信号処理実験を，あとから再実行・比較・拡張しやすい形で管理する方法を学ぶ．
研究で使うコードは，とりあえず一度動けばよいというものではない．
数週間後，数ヶ月後，論文締切直前の自分が同じ実験をもう一度実行できる必要がある．
この資料の目標は以下である．

- 研究で使う Python project を，依存関係，設定，実装，データ，結果の役割が分かれる形で設計できる
- notebook での探索と CLI による本実験を使い分け，実験条件を明示して再実行できる
- ブラインド音源分離の実験を，データ取得，音響シミュレーション，分離，評価，保存までの一連の流れとして組み立てられる
- 代表的な音源分離アルゴリズムについて，数式，配列の shape，実装の対応を追いながら理解できる
- 実験結果を機械的に集計できる形式で保存し，条件比較や反復過程を可視化できる
- 論文や発表に使う図を，媒体に合わせて再利用可能な設定として管理できる
- 自分だけでなく，共同研究者や将来の自分が追える実験コードを書くための基準を持てる

本資料では以下のライブラリを前提とする．

- 可視化: `matplotlib`
- 行列計算: `numpy`
- 音声ファイルの読み書き: `soundfile`
- スペクトル解析: `scipy.signal`
- 設定管理: `omegaconf`
- 音響シミュレーションと CMU ARCTIC の取得: `pyroomacoustics`
- 性能評価: `fast-bss-eval`
- 評価結果の集計: `pandas`
- 評価結果の統計可視化: `seaborn`

以降のコードブロックでファイルとして保存するものは，先頭行に `# src/bss_handson/cli.py` のように保存先のパスをコメントとして書く．
この行を目印に，対応するファイルへコードを保存する．

## 実験管理

### 実験管理の必要性

研究では「一度だけ動くコード」よりも，「あとから同じ条件で動かせるコード」の方が重要である．
特に音響信号処理の実験では，サンプリング周波数，窓長，シフト長，データセットの選び方，音源位置，マイク位置，残響条件，乱数 seed，AuxIVA の反復回数など，多数の条件が結果に影響する．
これらをコード中に直接書き散らすと，次のような問題が起きる．

- どの条件で作った図なのかわからない
- 似たような notebook や script が大量に増える
- 実験結果を上書きしてしまう
- 他人の環境で動かない
- 論文提出前に過去の結果を再現できない
- バグ修正によって，過去に正しかった処理が壊れても気づけない

たとえば，実験管理ができていない project では，次のような状態になりやすい．

```txt
research/
├── auxiva.ipynb
├── auxiva_copy.ipynb
├── auxiva_final.ipynb
├── auxiva_final2.ipynb
├── plot.py
├── mixture.wav
├── estimated_1.wav
├── estimated_2.wav
├── result.png
└── result_new.png
```

この状態では，`result_new.png` がどの notebook から作られたのか，どの wav を入力したのか，STFT の窓長や AuxIVA の反復回数が何だったのかをあとから確認しにくい．
さらに，notebook のセルをどの順序で実行したか，途中でどの変数を書き換えたかも残らない．
次のように条件をコードへ直接書き込むと，実験条件を変えるたびにコードそのものが変わり，結果と条件の対応が崩れやすい．

```py
import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
import soundfile as sf
from scipy import signal

# 何度か試した後のファイル名．どの条件の混合音なのかは残っていない．
mixture, fs = sf.read("mixture.wav")
mixture = mixture.T

# STFT 条件を直接書いているため，別条件を試すたびにコードが変わる．
win = signal.get_window("hann", 512)
stft = signal.ShortTimeFFT(win, hop=256, fs=fs)
x = stft.stft(mixture, axis=-1)
x = np.transpose(x, (2, 1, 0))

# 何度か試した後の値．なぜこの値なのかは残っていない．
y = pra.bss.auxiva(
    x,
    n_iter=80,
    proj_back=True,
)

# 出力ファイル名だけでは，入力，STFT 条件，反復回数，参照マイクがわからない．
for k in range(y.shape[2]):
    spec = y[:, :, k].T
    estimate = stft.istft(spec, f_axis=0, t_axis=1)
    sf.write(f"estimated_{k}.wav", estimate, fs)

plt.imshow(
    20 * np.log10(np.maximum(np.abs(y[:, :, 0].T), 1.0e-10)),
    origin="lower",
    aspect="auto",
)
plt.savefig("result_new.png")
```

このようなコードは，その場で図を作るだけなら便利である．
しかし，論文用の結果を作る段階では，入力データ，設定，実行コマンド，出力先を分けて管理する必要がある．

実験管理では **コード，設定，入力，出力を分ける** こと，そして **実行方法をコマンドとして残す** ことが重要である．
科学計算における実践としても，作業を自動化すること，変更履歴を残すこと，依存関係やパラメータを記録することが重要であると整理されている（[Wilson et al., 2014](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1001745)）．
また，研究計算の最小限の実践として，データ，ソフトウェア，プロジェクト構成，変更管理を分けて扱うことが推奨されている（[Wilson et al., 2017](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510)）．

## Pythonプロジェクト管理

### Python パッケージングとプロジェクト構成

Python の研究コードは，単なる script の集まりではなく，import 可能な package として整理しておくと再利用しやすい．
ここでは，Python package としてコードを配置する考え方と，一般的なディレクトリ構成を整理する．

#### パッケージングの基本

Python package では，配布パッケージ名と import package 名を区別する．
たとえば project 名が `bss-handson` の場合，Python の import 名は `bss_handson` のようにハイフンをアンダースコアに変えた名前になる．
このように，プロジェクト全体の名前と，Python の `import` 文で使う名前は一致しないことがある．

プロジェクトの配置には，大きく `flat layout` と `src layout` がある．

#### flat layout

```txt
project-name/
├── package_name/
│   ├── __init__.py
│   └── module.py
└── pyproject.toml
```

`flat layout` は単純で，初学者にもわかりやすい．
小さなスクリプトだけならこれでも問題は少ない．
一方で，プロジェクトのルートディレクトリが Python の import 対象になりやすく，意図しないファイルを偶然 import してしまうことがある．
単一 script や小さな app に近い構成では，flat layout 的な配置でも十分な場合がある．
ただし，研究コードを package として扱い，複数の処理から同じ関数を import するなら，`src layout` にする方がよい．

#### src layout

```txt
project-name/
├── src/
│   └── package_name/
│       ├── __init__.py
│       └── module.py
└── pyproject.toml
```

`src layout` では，実際の package を `src/` の下に置く．
この構成では，プロジェクトのルートディレクトリにある補助ファイルと，import される package の位置が分かれる．
研究コードが少し大きくなり，複数の実験 script やモジュールを持つようになる場合は `src layout` を推奨する．
Python Packaging User Guide でも，src layout はソースコードをプロジェクトルートとは別の `src/` 以下に置く構成であり，開発中のファイルを偶然 import する事故を避ける助けになると説明されている．
この節では配置の考え方だけを押さえ，具体的な作成コマンドは後の `uv` の節で扱う（[Python Packaging User Guide: src layout vs flat layout](https://packaging.python.org/ja/latest/discussions/src-layout-vs-flat-layout/)）．

研究では「最初は小さい実験」でも，すぐに次のような状態になる．

- 前処理，特徴量抽出，学習，評価，作図が別々の処理になる
- 同じ STFT や評価指標を複数の実験で使い回す
- notebook や実験 script から同じ関数を呼びたい
- 複数人でコードを読む

そのため，最初から `src layout` にしておく方が後の移行が少ない．

### README の記述事項

`README.md` は，そのプロジェクトを初めて見る人が最初に読む文書である．
研究コードでは，README は立派な説明文を書く場所というよりも，実験を再実行するための入口である．
半年後の自分や共同研究者が README だけを見て作業を始められるように，最低限必要な情報を書く．
研究計算の実践でも，プロジェクトの概要，実行例，依存関係を README や関連文書として残すことが推奨されている（[Wilson et al., 2017](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510)）．

#### 研究テーマと目的

まず，このリポジトリが何のためのものかを短く書く．
プロジェクトの目的が曖昧であると，後からファイルが増えたときに何を残すべきか判断しにくくなる．
研究テーマ，対象データ，主な手法，最終的に再現したい結果を数行でまとめる．

#### 環境構築

次に，実験を実行するための環境構築手順を書く．
使用する Python のバージョン，依存関係の同期方法，必要な外部ツール，動作確認用の最小コマンドを明記する．
依存関係を lock file で固定している場合は，その lock file を使って環境を再現する手順を書く．

#### データの準備

データの取得方法と配置場所を README に書く．
データセット名，取得元，配置先，前処理の有無，ライセンスや利用条件を明記する．
自動ダウンロードする場合も，どの処理でどこに保存されるのかを書いておく．

#### 実験の実行方法

README には，代表的な実験を 1 つ実行するためのコマンドを書く．
設定ファイルを指定する場合は，どの設定ファイルが標準例なのかも書く．
複数の実験条件を比較する project では，代表的な実行コマンドと，結果の保存先が対応するように書く．

#### 出力される結果

実験結果として何が生成されるのかも README に書く．
ファイル名，保存先，それぞれの役割，どの指標を比較すべきかを明記する．
出力の意味が README に書かれていないと，あとから `results/` を見ても何を比較すべきかわからなくなる．

#### 再現性に関する注意

最後に，再現性に関する注意を書く．
特に音響シミュレーションや反復最適化を含む実験では，乱数 seed，ライブラリのバージョン，データのバージョンが結果に影響する．
依存関係の固定方法，設定ファイルの管理方法，乱数 seed，実行時に保存する metadata，大きなデータや結果を git 管理しない方針を明記する．

### git 管理の方針

git はソースコードと小さなテキストファイルの履歴管理に向いている．
一方で，大きな音声ファイル，学習済みモデル，実験結果の画像やログを何でも git に入れると，リポジトリが巨大化して扱いにくくなる．
研究コードを書く前提として，変更履歴を残す，差分を読む，必要なファイルだけを commit する，`data/` や `results/` を誤って入れない，という操作は身につけておく必要がある．[^git-basic]

[^git-basic]: 情報科学科を卒業しておきながら git を使えないのはこの先つらい思いをすることになるので不安がある場合は，研究コードを書く前によく復習されたい．

#### git 管理すべきもの

- `src/` 以下の Python コード
- `configs/` 以下の設定ファイル
- `pyproject.toml`
- `uv.lock`
- `README.md`
- `.gitignore`
- 小さなサンプルデータ
- 論文や発表に使う最終的な図を再生成するための script

#### git 管理すべきでないもの

- 大きな音声データセット
- 前処理済み特徴量
- 学習済みモデル
- 実験結果の大量の図
- ログファイル
- キャッシュ
- 仮想環境
- Python の生成物

ただし，論文に掲載する最終的な図や，小さな動作確認用 wav などは git 管理してもよい．

#### `.gitignore` の例

```gitignore
# Python
__pycache__/
*.py[cod]

# Environments
.venv/

# Data and experiment outputs
data/
results/*

# Jupyter
.ipynb_checkpoints/

# OS/editor
.DS_Store
```

`.gitignore` を毎回手で書く代わりに，テンプレート生成ツールを使ってもよい．
たとえば `gibo` は，GitHub が公開している `.gitignore` テンプレートをコマンドラインから取得するためのツールである（[simonwhitaker/gibo](https://github.com/simonwhitaker/gibo)）．
Python と macOS のテンプレートを出発点にするなら，`gibo dump Python macOS >> .gitignore` のように実行できる．
ただし，音声データや実験結果の保存先は project ごとに異なるため，`data/` や `results/` などの実験固有の項目は自分で確認して追加する．

音声データセットの置き場所は，研究室サーバ，外付けストレージ，クラウドストレージなどを使い，README に取得方法や配置方法を書く．
「データそのもの」ではなく「どのデータをどう置けば実験が動くか」を git 管理する，という考え方が重要である．

### notebook と Python CLI

Jupyter notebook は非常に便利である．
音声波形を読み込んで確認する，スペクトログラムを描く，パラメータを少し変えながら挙動を見る，論文用の図の案を作る，といった用途では notebook が向いている．
Google Colaboratory も，ブラウザだけで Python を動かせ，環境構築をほとんど意識せずに GPU まで使えるという点で非常に便利である．
一方で，依存関係の固定，データの置き場所，セルの実行順序，実験条件の記録，結果の保存先を明示的に管理しにくい場合がある．
「このセルを上から順に実行すれば再現できるはず」という期待は，実験管理としては弱い．したがって，大規模な実験では Python CLI を推奨する．[^notebook-cli]

[^notebook-cli]: notebook や Colaboratory の便利さは，環境構築だけでなく，依存関係の固定，データの置き場所，セルの実行順序，実験条件の記録，結果の保存先まで，うっかり意識しなくてよい気分にさせてくれる点で危険である．探索とデモには優秀である一方，実験を管理するという観点では，状態を隠し，履歴を曖昧にし，再実行性を静かに削ってくる．

ここでいう CLI とは，次のようにコマンドラインから実行できる Python プログラムである．

```bash
uv run bss-handson --config configs/bss.yaml
```

大規模実験で CLI が向いている理由は次の通りである．

- 実行コマンドをそのまま記録できる
- サーバや GPU マシン上で実行しやすい
- 複数条件を shell script や job scheduler で回しやすい
- notebook のセル実行順序に依存しない
- git diff が読みやすい
- 共通処理を package として再利用しやすい

notebook は「考える場所」，Python module と CLI は「実験を回す場所」と考えるとよい．
notebook で試して，固まった処理を `src/` に移し，CLI から呼び出す，という流れが現実的である．

### `uv` によるプロジェクト管理

`uv` は Python のパッケージ管理，仮想環境作成，依存関係の固定，コマンド実行をまとめて扱えるツールである．
研究コードでは，実験当時のライブラリのバージョンが変わるだけで結果が変わることがあるため，依存関係を明示的に管理する必要がある．
研究向けの Python 開発環境全体については，環境構築，ディレクトリ構成，静的解析，自動整形，実験設定管理などをまとめた資料も参考になる（[研究のための Python 開発環境](https://zenn.dev/zenizeni/books/a64578f98450c2)）．
また，`uv` による Python 環境構築の授業資料として，`uv init`，`uv add`，`uv sync` などの基本操作を扱う講義スライドも公開されている（[uvを使ったPython環境構築 - 人工知能応用特論Ⅰ 第3回](https://www.docswell.com/s/2625216247/Z2Q3YV-2025-10-22-170737#p17)）．

#### プロジェクトの作成

今回は `bss-handson` というプロジェクトを作る．

`uv init` は，project 種別を指定するオプションを付けない場合，application project を作成する．
たとえば `uv init bss-handson` は，`bss-handson/` というディレクトリを作り，その中に application project を初期化する．
project 種別としては，これは `uv init bss-handson --app` に相当する．
デフォルトの application project では，`pyproject.toml`，`README.md`，`.python-version`，`main.py` のようなファイルが作られる．
この構成は，小さな script や単体で実行する application の出発点としては扱いやすい．

一方で，デフォルトの application project は Python package として build される構成ではない．
そのため，`pyproject.toml` に `[build-system]` は含まれず，project 内のコードを package として install して使う前提にもならない．
`uv run main.py` のようにファイルを指定して実行する用途には向いているが，`src/bss_handson/` に処理を分けて置き，`uv run bss-handson` のような entry point から呼び出す教材には不十分である．
この違いは `uv` の公式ドキュメントでも，デフォルトは application，`--lib` は library を作る指定として整理されている（[uv: Creating projects](https://docs.astral.sh/uv/concepts/projects/init/)）．

そのため，このハンズオンでは `--lib` を付けて初期化する．
ただし，`uv init bss-handson --lib` のように project 名を引数に渡すと，そのコマンドを実行したディレクトリの下に `bss-handson/` が作られる．
すでに `bss-handson/` という作業ディレクトリを作ってからその中で実行すると，`bss-handson/bss-handson/` のように project が一段深くなってしまう．

まず，研究用 project を置くための適当な親ディレクトリへ移動する．
勉強会資料やダウンロードした配布資料を置いているディレクトリの中で直接作業を始めると，教材ファイル，生成物，Python package が混ざりやすい．
そのため，実際に実験を動かす作業ディレクトリは，資料置き場とは分けて作る方がよい．

次の例では，`~/codes/` の下に `bss-handson/` を作り，その中で `uv init --lib` を実行する．

```bash
mkdir -p ~/codes
cd ~/codes
mkdir bss-handson
cd bss-handson
uv init --lib
```

以降のコマンドは，必ず `bss-handson/` の中で実行する．
`uv` は実行したディレクトリ，またはその親ディレクトリにある `pyproject.toml` を探して project を決める．
そのため，勉強会資料を置いている親ディレクトリで `uv run ...` を実行した場合と，`bss-handson/` の中で `uv run ...` を実行した場合では，使われる環境や entry point が変わることがある．
たとえば `uv run bss-handson` は，`bss-handson/pyproject.toml` に書かれた `[project.scripts]` をもとに作られるコマンドなので，`bss-handson/` の外で実行すると `Failed to spawn: bss-handson` のようなエラーになることがある．

`--lib` は，ライブラリとして再利用できる Python package を作成するためのオプションである．
このオプションを付けると，`src/bss_handson/` のように `src/` 以下へ import 可能な package が作られ，`pyproject.toml` には package を build するための `[build-system]` も最初から書かれる．
これは `bss_handson` を package として install し，`uv run bss-handson` から project 内の実装を参照するための構成である．
今回のように，CLI からも notebook からも同じ関数を import して使いたい実験コードでは，最初から `--lib` で初期化しておくのがよい．

`uv init` には `--app` や `--script` などの選択肢もある．
`--app` はアプリケーションとして実行することを主に考える構成，`--script` は単一 script を作る構成である．
今回の教材では，`bss_handson.auxiva` や `bss_handson.stft` のように関数を module として分けて再利用するため，`--lib` を使う．

`--package` は，project を Python package として build できるようにするための指定である．
`--lib` は library project を作るための指定であり，実質的に package として扱う構成も含んでいる．
つまり，`--package` は「package として build するか」に注目した指定であり，`--lib` は「再利用可能な library として作るか」に注目した指定である．
今回のように `src/bss_handson/` に関数を分けて置き，CLI や notebook から import する教材では，単に package 化するだけでなく library として扱いたいため `--lib` を選ぶ．

整理すると次のようになる．

| オプション  | 主な用途                             | この教材での扱い                 |
| :---------- | :----------------------------------- | :------------------------------- |
| `--script`  | 1 ファイルの script                  | 使わない                         |
| `--app`     | 実行するアプリケーション             | 今回は使わない                   |
| `--package` | package として build 可能にする      | CLI entry point には必要な考え方 |
| `--lib`     | import して再利用する library を作る | 今回の推奨                       |

#### 依存関係の追加

```bash
uv add numpy scipy matplotlib soundfile omegaconf pyroomacoustics fast-bss-eval pandas seaborn
```

研究用の実行に必要なものは通常の依存関係として追加する．

今回の主な役割は以下の通りである．

| ライブラリ        | 用途                                       |
| :---------------- | :----------------------------------------- |
| `numpy`           | 波形，スペクトル，行列の基本計算           |
| `scipy.signal`    | STFT，フィルタ，スペクトル解析             |
| `matplotlib`      | 図の作成                                   |
| `soundfile`       | wav などの音声ファイル読み書き             |
| `omegaconf`       | YAML 設定ファイルの読み込みと CLI override |
| `pyroomacoustics` | 音声データの取得，音響シミュレーション     |
| `fast-bss-eval`   | 音源分離結果の性能評価                     |
| `pandas`          | 評価結果 JSON の表形式への変換             |
| `seaborn`         | 評価結果の条件比較図の作成                 |

#### コマンドの実行

`uv run` を使うと，プロジェクトの仮想環境内でコマンドを実行できる．

```bash
uv run python --version
uv run python -c "import numpy; print(numpy.__version__)"
```

`python script.py` ではなく `uv run python script.py` と書く習慣をつけると，意図しない Python 環境で実行する事故を減らせる．

#### `pyproject.toml` と `uv.lock`

`uv` では依存関係が主に次の 2 ファイルで管理される．

- `pyproject.toml`: 人間が読むためのプロジェクト設定
- `uv.lock`: 実際に解決された依存関係の固定結果

どちらも git 管理する．
`uv.lock` を管理しておくと，他の人や将来の自分がかなり近い環境を再現できる．

`uv init --lib` で作成した project には，`pyproject.toml` に `[build-system]` が含まれる．
これは，このディレクトリを Python package としてどの backend で build するかを指定する設定である．
このハンズオンで実際に使う `bss-handson/pyproject.toml` では，`uv` の build backend として `uv_build` を次のように指定している．

```toml
[build-system]
requires = ["uv_build>=0.11.20,<0.12.0"]
build-backend = "uv_build"
```

`requires` は build に必要な package を表し，ここでは `uv_build` の `0.11` 系を使うことを指定している．
`build-backend = "uv_build"` は，実際の build 処理を `uv_build` に任せるという意味である．
`[project.scripts]` に CLI コマンドを書いても，project が package として build できなければ entry point は作られない．
そのため，`uv run bss-handson` のようなコマンドを使うには，`[project.scripts]` だけでなく `[build-system]` も必要である．
`uv init --lib` で初期化しておけば，この設定は最初から入る．
`build-system` がない場合は，`uv` が現在の project を install できず，結果として `Failed to spawn: bss-handson` のようなエラーになることがある．

ここまでの操作により，`src/` 以下へ import 可能な package を置く構成と，package build に必要な `pyproject.toml` が用意される．
以降のコマンドは，作業ディレクトリに `pyproject.toml` があることを確認してから実行する．

`uv run` で Python が実行できることも確認しておく．
この確認では，system の Python ではなく，`bss-handson` project の仮想環境内の Python が使われていることを見る．
`numpy` の version が表示されれば，依存関係の解決と import ができている．

```bash
uv run python -c "import numpy; print(numpy.__version__)"
```

### プロジェクト構成

ここからは，ハンズオンで実際に作成する `bss-handson` project の構成を決める．

#### 最初に作成するディレクトリ構成

音源分離の実験では，まず次の最小構成を作る．
研究計算のプロジェクト構成では，`data/`，`results/`，`src/` のように役割ごとにファイルを分けることが推奨されており，本資料の構成もその考え方に合わせる（[Wilson et al., 2017](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510)）．

```txt
bss-handson/
├── configs/
│   └── bss.yaml
├── data/
├── results/
├── scripts/
│   └── run_bss.py
├── src/
│   └── bss_handson/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       └── stft.py
├── .gitignore
├── pyproject.toml
├── README.md
└── uv.lock
```

#### 各ディレクトリの役割

各ディレクトリの役割は次の通りである．

| パス       | 役割                                                          |
| :--------- | :------------------------------------------------------------ |
| `configs/` | 実験条件を記述する設定ファイル                                |
| `data/`    | 音声データなどの入力データ                                    |
| `results/` | 実験結果，図，ログ，モデル重み                                |
| `scripts/` | `src/` の関数を組み合わせる実験実行，設定生成，集計，可視化   |
| `src/`     | 1 つの意味のある入力に対して 1 つの結果を返す再利用可能な処理 |

`src/` と `scripts/` の違いは，コードの長さではなく責務で決める．
`src/` には，入力と出力の意味が明確で，他の実験から import して再利用できる処理を置く．
たとえば，音声波形を STFT に変換する，音源信号と部屋条件から混合信号を作る，観測スペクトログラムから分離信号を推定する，参照信号と推定信号から評価値を返す，という処理は `src/` に置く．
これらは「何を入力すると何が返るか」を関数の引数と戻り値で説明できる．

一方で，`scripts/` には，`src/` の関数を組み合わせて特定の実験を実行する処理を置く．
たとえば，設定ファイルを読み，CMU ARCTIC を取得し，room simulation を行い，STFT，AuxIVA，ISTFT，評価，保存を順番に実行する処理は `scripts/run_bss.py` に置く．
複数条件を作る，複数の `metrics.json` を集計する，図を保存する，random trial を回す，という処理も `scripts/` に置く．
これらは単一の数値計算関数ではなく，実験の手順そのものだからである．

判断に迷う場合は，そのコードを「別の実験でもそのまま import して使いたいか」で決める．
使いたいなら `src/`，今回の実験条件や出力ディレクトリに強く依存するなら `scripts/` に置く．
また，`src/` から `scripts/` を import してはいけない．
依存関係は常に `scripts/` から `src/` へ向ける．

`bss-handson/` の中に，ハンズオンで使うディレクトリを作成せよ．
この段階では中身をすべて埋める必要はなく，コード，設定，データ，結果を置く場所を先に決めることが目的である．
最初に置き場所を決めておくと，あとから `results/` に wav や図が増えても，ソースコードと混ざりにくい．
このコマンドを実行する前に，`pwd` や `ls` で現在の作業ディレクトリが `bss-handson/` であり，`pyproject.toml` が見えていることを確認する．

```bash
mkdir -p configs data results scripts src/bss_handson
```

`tree` コマンドなどで適宜ディレクトリ構成を確認せよ．
後続の節で AuxIVA，評価，可視化，style file，補助 script を追加し，完成形の構成へ広げていく．

## 音源分離実験ハンズオン

### README の作成

`bss-handson` では，次の内容を `README.md` に書く．
この README は，ハンズオン後に自分で実験を再実行するための入口になる．

README には，少なくとも次の項目を含める．

- プロジェクトの目的: CMU ARCTIC の音声を室内混合し，NumPy 実装の AuxIVA で分離して評価すること
- セットアップ: `uv sync` により `pyproject.toml` と `uv.lock` から環境を再現すること
- データ: 初回実行時に CMU ARCTIC が `data/cmu_arctic/` 以下へダウンロードされること
- 実行方法: 標準設定を `uv run bss-handson --config configs/bss.yaml` で実行すること
- 出力: `results/<experiment>/` に wav，spectrogram，`config.yaml`，`metrics.json`，`iteration_metrics.json` を保存すること
- 再現性: 実験条件は `configs/*.yaml` と command line override で管理し，`data/` と `results/` は通常 git 管理しないこと

この時点では README に詳細な作図コマンドや発展的な評価手順をすべて書く必要はない．
標準実験を再実行する入口と，出力を読むための最小限の情報があればよい．

### 音源分離実験パイプライン

ここからは，より研究に近い題材として，音声データの取得，音響シミュレーション，ブラインド音源分離，性能評価，可視化までを 1 つの CLI として構成する．
扱う問題は N 音源 N マイクの determined BSS である．
標準設定では 2 音源 2 マイクを使うが，`speakers`，`utterance_indices`，`source_positions`，`mic_positions` を同じ数だけ増やせば，3 音源 3 マイク以上の設定にも拡張できる．
入力音声には CMU ARCTIC，部屋のシミュレーションには `pyroomacoustics`，評価には `fast-bss-eval` を使う．
ただし，AuxIVA は既存ライブラリの実装を呼び出さず，このハンズオンで NumPy による最小実装を書く．

#### `configs/bss.yaml` の作成

実験条件を Python コードに直接書くと，条件を変えるたびにコードを書き換えることになり，バグの温床になる．
そのため，実験条件は TOML, YAML, JSON など外部の設定ファイルに分けた方が管理しやすい．
ここでは YAML を使う．

設定ファイルには，次のような情報を書く．

- 入力ファイル
- 出力ディレクトリ
- サンプリング周波数
- STFT の `window`, `win_length`, `hop`
- 乱数 seed
- 部屋の大きさ，残響条件，音源位置，マイク位置
- AuxIVA の反復回数や数値安定化のための微小値
- AuxIVA の音源モデル
- AuxIVA の更新則
- projection back の参照マイク
- 図の style file と色範囲

`configs/` ディレクトリを作成せよ．
設定ファイルは，実験条件を Python コードから切り離すための入口である．
このディレクトリに YAML を置くことで，コードを変更せずに STFT 条件，部屋条件，AuxIVA の反復回数を変えられる．
このコマンドも `bss-handson/` の中で実行する．
実行前に，現在のディレクトリに `pyproject.toml` があることを確認する．

```bash
mkdir -p configs
```

標準設定として `configs/bss.yaml` を作成せよ．
このファイルは，ハンズオン全体で最初に実行する基準条件である．
以降で条件を変えるときは，この YAML をコピーして `output_dir` や一部のパラメータだけを変える．

```yaml
# configs/bss.yaml
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

`output_dir` は，この実行で作る wav，spectrogram，設定ファイル，評価値の保存先である．
条件を変える実験では，標準設定の結果を上書きしないように `results/bss_niter_10` のような別名に変える．

`seed` は，乱数を使う処理の再現性を保つための値である．
標準 BSS 実験では主に設定の一部として保存される．
音源位置や発話をランダムに選ぶ実験へ拡張するときは，同じ考え方で乱数 seed を記録する．

`dataset.basedir` は，CMU ARCTIC を保存するディレクトリである．
初回実行時に `pyroomacoustics` の dataset wrapper がこの下へ音声をダウンロードする．
`dataset.speakers` は使う話者 ID，`dataset.utterance_indices` は各話者から使う発話 index である．
この 2 つの list は同じ長さにし，1 つの話者と 1 つの発話 index が 1 つの音源に対応する．

`stft.window`，`stft.win_length`，`stft.hop` は，時間波形を時間周波数領域へ変換する条件である．
標準設定では Hann 窓，窓長 1024 samples，hop 256 samples を使う．
AuxIVA は STFT 後の複素スペクトログラムを入力にするため，この条件は分離性能と ISTFT 後の波形に影響する．

`room.fs` は room simulation と保存 wav のサンプリング周波数である．
`room.size` は 2 次元 shoebox room の幅と奥行き，`room.rt60` は残響時間である．
`simulate_room()` では，この RT60 から `pyroomacoustics.inverse_sabine()` により壁の吸音率と image source method の反射次数を計算する．
`room.source_positions` は各音源の位置，`room.mic_positions` は各マイクの位置であり，どちらも `[x, y]` のリストで指定する．
この標準設定では，マイク配置は 5 cm 間隔のまま固定し，音源をマイクアレイの左右上下に離して配置している．
この教材では determined BSS を扱うため，音源数とマイク数は一致させる．

`auxiva.n_iter` は AuxIVA の反復回数である．
`auxiva.model` は音源モデルであり，`laplace` または `gauss` を指定する．
`auxiva.update_method` は分離行列の更新則であり，`ip` または `iss` を指定する．
`auxiva.eps` はゼロ除算や極端な重みを避けるための微小値である．
`auxiva.reference_mic` は projection back で分離信号のスケールを合わせる参照マイク index である．

`plot.style` は Matplotlib style file のリストである．
前から順に読み込まれ，後の style が前の style を上書きする．
標準設定では共通設定 `styles/common.mplstyle` と論文向け設定 `styles/paper.mplstyle` を使う．
`plot.vmin_db` と `plot.vmax_db` は，スペクトログラムを保存するときの dB 表示の色範囲である．
色範囲を固定すると，条件を変えた図を比較するときにカラースケールの違いを性能差と誤解しにくい．

#### 設定ファイルの読み込み

`src/bss_handson/config.py` を作る．

```py
# src/bss_handson/config.py
from pathlib import Path

from omegaconf import OmegaConf


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    base_config = OmegaConf.load(path)
    override_config = OmegaConf.from_dotlist(overrides or [])
    config = OmegaConf.merge(base_config, override_config)
    return OmegaConf.to_container(config, resolve=True)


def save_config(config: dict, path: str | Path) -> None:
    OmegaConf.save(config=OmegaConf.create(config), f=path)


def get_output_dir(config: dict) -> Path:
    return Path(config["output_dir"])


def get_dataset_config(config: dict) -> dict:
    return dict(config["dataset"])


def get_room_config(config: dict) -> dict:
    return dict(config["room"])


def get_stft_config(config: dict) -> dict:
    return dict(config["stft"])


def get_auxiva_config(config: dict) -> dict:
    return dict(config["auxiva"])


def get_plot_config(config: dict) -> dict:
    return dict(config["plot"])
```

ここでは `os.path` ではなく `pathlib.Path` を使う．
`os.path.join()` や `os.path.exists()` は文字列として path を組み立てる関数であり，ファイルを読む処理，ディレクトリを作る処理，glob で探す処理は別の関数や module に分かれやすい．
一方で，`Path` は path を表す object であり，`Path("results") / "bss_example" / "metrics.json"` のように path を組み立てられる．
さらに，`path.exists()`，`path.mkdir(parents=True, exist_ok=True)`，`path.glob("bss_*/metrics.json")`，`path.read_text()` のように，path に対する操作を同じ object から呼び出せる．
研究コードでは，設定ファイル，音声ファイル，結果ディレクトリ，評価 JSON など多くの path を扱うため，`Path` に統一しておくと，文字列連結によるミスを減らし，処理の意図も読みやすくなる．

`OmegaConf.load()` は YAML を階層的な設定 object として読み込む．
`OmegaConf.from_dotlist()` は，`auxiva.n_iter=10` や `output_dir=results/bss_niter_10` のような command line override を設定 object に変換する．
`OmegaConf.merge()` は，設定ファイルで読んだ標準設定に command line override を重ねる．
OmegaConf では，YAML ファイル，dot-list，command line 引数から作った設定を merge できるため，実験設定の標準値と一時的な変更を分けて扱いやすい（[OmegaConf: Usage](https://omegaconf.readthedocs.io/en/latest/usage.html)）．
そのため，標準設定を `configs/bss.yaml` に残したまま，実行時に一部のパラメータだけを変更できる．
最後に `OmegaConf.to_container(..., resolve=True)` で通常の Python `dict` に変換している．
これは，後段の `load_cmu_arctic_sources(**config["dataset"])` のような `**` 展開に渡しやすくするためである．

`save_config()` は，override まで反映した最終的な設定を `results/<experiment>/config.yaml` として保存するための関数である．
元の `configs/bss.yaml` を単純にコピーすると，command line で変更した `auxiva.n_iter` や `output_dir` が結果ディレクトリに残らない．
実験結果をあとから再確認するには，実際に使った最終設定を保存する必要がある．

一方で，実験が大きくなると，設定項目の typo や型の間違いに気づきにくくなる．
たとえば `win_length` と書くべきところを `window_length` と書いた場合，その設定が使われないまま実行されたり，実行途中で `KeyError` になったりする．
また，`n_iter` に整数ではなく文字列を書いてしまうような間違いも起こる．
このような問題が増えてきたら，`dataclasses` や `pydantic` を使って，設定ファイルの構造と型を明示することを検討する．

#### CMU ARCTIC の取得

`pyroomacoustics` には CMU ARCTIC の dataset wrapper[^wrapper] が用意されている．
公式ドキュメントでは，dataset wrapper は音声 sample と metadata をまとめて扱うための仕組みであり，`pra.datasets.CMUArcticCorpus(download=True, speaker=['bdl'])` のように書くと，指定した話者の corpus が手元になければ自動的にダウンロードされると説明されている（[pyroomacoustics: Dataset Wrappers](https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.datasets.html)）．
CMU ARCTIC は，Carnegie Mellon University の Language Technologies Institute で作成された音声合成研究用の英語音声 corpus である（[CMU_ARCTIC Databases](https://www.festvox.org/cmu_arctic/)）．
単一話者ごとに収録された phonetically balanced な発話から構成され，Kominek and Black (2004) では，スタジオ環境で収録された約 1200 発話からなる音声合成用データベースとして説明されている（[Kominek and Black, 2004](https://www.isca-archive.org/ssw_2004/kominek04b_ssw.html)）．
代表的な話者として，米語男性話者 `bdl`，米語女性話者 `slt` などが用意されている．
このハンズオンでは，発話数が多すぎず，複数話者を簡単に選べ，`pyroomacoustics` から取得できるため，2 音源以上の音響シミュレーション用のクリーン音声として利用する．

`src/bss_handson/data.py` を作る．

```py
# src/bss_handson/data.py
from pathlib import Path

import numpy as np
import pyroomacoustics as pra


def load_cmu_arctic_sources(
    basedir: str | Path,
    speakers: list[str],
    utterance_indices: list[int],
) -> tuple[np.ndarray, int]:
    basedir = Path(basedir)
    basedir.mkdir(parents=True, exist_ok=True)
    signals = []
    sample_rates = []

    for speaker, index in zip(speakers, utterance_indices, strict=True):
        corpus = pra.datasets.CMUArcticCorpus(
            basedir=str(basedir),
            download=True,
            speaker=[speaker],
        )
        sentence = corpus[index]
        signal = np.asarray(sentence.data, dtype=np.float64)
        signal = signal / max(np.max(np.abs(signal)), 1.0e-12)
        signals.append(signal)
        sample_rates.append(sentence.fs)

    if len(set(sample_rates)) != 1:
        raise ValueError(f"sample rates must be identical: {sample_rates}")

    length = min(len(signal) for signal in signals)
    sources = np.stack([signal[:length] for signal in signals], axis=0)
    return sources, sample_rates[0]
```

ここでは `sources` の shape を `(n_sources, n_samples)` とする．
実験コードでは，このように配列の shape を最初に決めておくことが重要である．
`speakers` と `utterance_indices` は同じ長さの list として扱う．
`zip(..., strict=True)` を使っているため，話者数と発話 index 数が一致しない場合はその場で例外が出る．
これは，意図せず一部の音源だけを読み込んで実験してしまう事故を防ぐためである．

各発話は `sentence.data` から NumPy 配列として取り出し，`float64` に変換する．
その後，最大絶対値で割って振幅をおおよそ `[-1, 1]` の範囲に正規化する．
複数の発話は長さが異なることがあるため，最も短い発話に合わせて切り詰めてから `np.stack` する．
この処理により，返り値 `sources` は常に 2 次元配列になり，後段のシミュレーションや評価で扱いやすくなる．
サンプリング周波数が話者間で異なる場合は，STFT や room simulation の条件が崩れるため，`ValueError` で停止する．

#### N 音源 N マイクの音響シミュレーション

次に，複数の音源信号を同じ本数のマイクで観測する混合信号を作る．
ここでは 2 次元の shoebox room を使う．
実際の部屋録音を用意しなくても，部屋サイズ，吸音率，音源位置，マイク位置を変えながら実験できるのが音響シミュレーションの利点である．
このハンズオンの AuxIVA 実装は determined condition を仮定するため，音源数とマイク数は一致させる．

`src/bss_handson/simulation.py` を作る．

```py
# src/bss_handson/simulation.py
import numpy as np
import pyroomacoustics as pra


def simulate_room(
    sources: np.ndarray,
    fs: int,
    size: list[float],
    rt60: float,
    source_positions: list[list[float]],
    mic_positions: list[list[float]],
) -> np.ndarray:
    if len(source_positions) != sources.shape[0]:
        raise ValueError(
            f"source_positions must have one position per source: "
            f"{len(source_positions)} != {sources.shape[0]}"
        )
    if len(mic_positions) != sources.shape[0]:
        raise ValueError(
            f"this handson assumes determined BSS, so the number of microphones "
            f"must match the number of sources: {len(mic_positions)} != {sources.shape[0]}"
        )

    absorption, max_order = pra.inverse_sabine(rt60, size)
    room = pra.ShoeBox(
        size,
        fs=fs,
        materials=pra.Material(absorption),
        max_order=max_order,
    )

    for source, position in zip(sources, source_positions, strict=True):
        room.add_source(position, signal=source)

    mic_array = np.asarray(mic_positions, dtype=np.float64).T
    room.add_microphone_array(pra.MicrophoneArray(mic_array, fs=fs))
    room.simulate()

    mixture = np.asarray(room.mic_array.signals, dtype=np.float64)
    mixture = mixture / max(np.max(np.abs(mixture)), 1.0e-12)
    return mixture
```

`mixture` の shape は `(n_mics, n_samples)` である．
determined BSS では `sources.shape[0] == mixture.shape[0]` となる．
標準設定ではこの値が 2 であるが，設定ファイルの音源数とマイク数を同じ数にすれば 3 以上にも対応できる．

`simulate_room()` では，まず設定ファイルで指定した音源位置とマイク位置の数を確認する．
音源位置が足りない場合，どの信号をどこに置いたのかが曖昧になるため，`source_positions` の長さを `sources.shape[0]` と一致させる．
また，この教材の AuxIVA は determined condition の実装なので，マイク数も音源数と一致させる．
この確認を room simulation の前に行うことで，後段の STFT や AuxIVA の内部で shape error になる前に，設定ファイルの問題として原因を特定できる．

`pra.ShoeBox` は，矩形の部屋を作る pyroomacoustics の基本的なクラスである．
`size` は部屋の大きさ，`rt60` は残響時間を表す．
`pra.inverse_sabine(rt60, size)` は，指定した残響時間に近づくように，壁の吸音率 `absorption` と image source method で考慮する反射次数 `max_order` を計算する．
`room.add_source()` で各音源を部屋に置き，`MicrophoneArray` でマイク位置を登録してから `room.simulate()` を呼ぶと，マイクで観測される混合信号が `room.mic_array.signals` に保存される．
最後に混合信号全体を最大絶対値で正規化し，音声ファイルとして保存しやすいスケールにしている．

#### STFT と ISTFT

AuxIVA は時間周波数領域で動作するため，時間波形を STFT に変換する必要がある．
ここでは `scipy.signal.ShortTimeFFT` の wrapper 関数を用意する．

`src/bss_handson/stft.py` を作る．

```py
# src/bss_handson/stft.py
import numpy as np
from scipy import signal


def create_stft(
    fs: int,
    window: str,
    win_length: int,
    hop: int,
) -> signal.ShortTimeFFT:
    win = signal.get_window(window, win_length)
    return signal.ShortTimeFFT(
        win,
        hop=hop,
        fs=fs,
    )


def stft_channels(
    signals: np.ndarray,
    fs: int,
    window: str,
    win_length: int,
    hop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stft = create_stft(fs, window, win_length, hop)
    spectra = stft.stft(signals, axis=-1)
    x = spectra.transpose(2, 1, 0)
    return x, stft.f, stft.t(signals.shape[1])


def istft_sources(
    spectra: np.ndarray,
    fs: int,
    window: str,
    win_length: int,
    hop: int,
    n_samples: int,
) -> np.ndarray:
    stft = create_stft(fs, window, win_length, hop)
    spectra_sources_first = spectra.transpose(2, 1, 0)
    return stft.istft(
        spectra_sources_first,
        k1=n_samples,
        f_axis=-2,
        t_axis=-1,
    )
```

`stft_channels` の出力 `x` の shape は `(n_frames, n_freq, n_mics)` である．
`ShortTimeFFT.stft` は `axis` で指定した軸に沿って STFT を計算するため，`signals` の shape が `(n_mics, n_samples)` の場合は `axis=-1` を指定すれば全マイクをまとめて処理できる．
このとき出力は `(n_mics, n_freq, n_frames)` になるため，AuxIVA で使いやすいように `(n_frames, n_freq, n_mics)` へ転置する．
逆変換では `spectra` を `(n_sources, n_freq, n_frames)` に戻し，`f_axis=-2`, `t_axis=-1` を指定して `istft` を呼ぶ．
以降では，時間 frame を $t$ ，周波数 bin を $f$ ，マイク index を $m$ とする．
NumPy の ndarray は，標準では最後の軸が連続したメモリに配置される C-order である．
音響信号処理の実験では frame 数が最も大きくなることが多いため，大きな次元である frame を先頭に置き，最後の小さな channel 軸に対して行列演算を行うと扱いやすい．
また，大きな次元を先に置くと，`x[t, f]` のように小さな channel ベクトルを取り出しながら反復する実装で余計な転置を減らしやすく，実験コード全体として速くしやすい．
この資料では，観測信号を `frame, freq, chan` の順で実装する．

#### 補助関数型独立ベクトル分析

ブラインド音源分離は観測信号だけから混合前の音源信号を推定する．補助関数型独立ベクトル分析
(auxilary-function-based independent vector analysis, AuxIVA) は代表的な周波数領域 BSS 手法である．
AuxIVA は，独立ベクトル分析 (independent vector analysis, IVA) に補助関数法を用いた更新則であり，重み付き共分散行列の更新と分離行列の更新を交互に行う（[Ono, 2011](https://doi.org/10.1109/ASPAA.2011.6082320)）．
IVA は，周波数ごとに独立に ICA を解くのではなく，周波数間の依存を利用して周波数領域 BSS の permutation 問題を抑える考え方に基づく（[Kim et al., 2007](https://doi.org/10.1109/TASL.2006.872618)）．

このハンズオンでは determined BSS を仮定し，観測チャンネル数と分離チャンネル数はどちらも $K$ とする．
観測信号を $\mathbfit{x} _ {f,t} \in \mathbb{C}^{K}$ ，分離信号を $\mathbfit{y} _ {f,t} = W _ {f} \mathbfit{x} _ {f,t}$ とする．
$W _ {f} \in \mathbb{C}^{K \times K}$ は周波数 $f$ の分離行列であり， $W _ {f}$ の $k$ 番目の行ベクトルを $\mathbfit{w} _ {k,f}^{\mathsf{H}}$ と書く．
したがって， $k$ 番目の分離信号は $y _ {k,f,t} = \mathbfit{w} _ {k,f}^{\mathsf{H}} \mathbfit{x} _ {f,t}$ である．
AuxIVA では，現在の分離信号から音源モデルの重みを計算し，その重みを固定して $W _ {f}$ に関する補助目的関数を最小化する．
このとき，周波数方向に沿った二乗和の平方根を

$$
r _ {k,t} = \sqrt{\sum _ {f=1}^{F} \lvert y _ {k,f,t}\rvert ^2}
$$

と定義し，事前に仮定した音源モデルによって重み関数 $\varphi(r _ {k,t})$ の実現が変わる．
球対称 Laplace 分布では $\varphi(r _ {k,t}) = 1 / (2 r _ {k,t})$ を使う．
一方，ゼロ平均時変分散複素 Gauss 分布では，各時間 frame の時変分散を周波数方向の平均二乗振幅として推定するため， $\varphi(r _ {k,t}) = F / r _ {k,t}^{2}$ を使う．
単に $1 / r _ {k,t}^{2}$ とすると，周波数方向の平均ではなく二乗和を分散として扱うことになる．
`pyroomacoustics.bss.auxiva` でも，`model="laplace"` と `model="gauss"` によってこの 2 種類の統計モデルを切り替えられる．
Laplace model は一般にロバストであり，time-varying Gauss model は音源数が少なくノイズが小さい条件で高い性能を示すことがある（[pyroomacoustics: Independent Vector Analysis (AuxIVA)](https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.bss.auxiva.html)）．
このとき，重み付き共分散行列 $V _ {k,f}$ を用いると， $W _ {f}$ の最適化問題は

$$
\min _ {\lbrace W _ {f}\rbrace  _ {f=1}^{F}} \mathcal{J} = \sum _ {f=1}^{F} \left( \sum _ {k=1}^{K} \mathbfit{w} _ {k,f}^{\mathsf{H}} V _ {k,f} \mathbfit{w} _ {k,f} - \log \lvert \det W _ {f}\rvert ^2 \right)
$$

と書ける．
第 1 項は $\mathbfit{w} _ {k,f}^{\mathsf{H}}\mathbfit{x} _ {f,t}$ の二乗形式から得られる項であり，第 2 項はヤコビアンに由来する項である．

AuxIVA では補助関数法により，次の反復更新を行う．
まず，現在の分離信号から音源モデルに応じた重み $\varphi(r _ {k,t})$ を計算し，重み付き共分散行列を計算する．

$$
V _ {k,f} = \frac{1}{T} \sum _ {t=1}^{T} \varphi(r _ {k,t}) \mathbfit{x} _ {f,t} \mathbfit{x} _ {f,t}^{\mathsf{H}}
$$

次に， $k$ 番目の基底ベクトルを $\mathbfit{e} _ {k}$ として，分離ベクトルを

$$
\mathbfit{w} _ {k,f} \gets \left(W _ {f} V _ {k,f}\right)^{-1} \mathbfit{e} _ {k}
$$

で更新する．
最後にスケールを正規化する．

$$
\mathbfit{w} _ {k,f} \gets \frac{\mathbfit{w} _ {k,f}} {\sqrt{\mathbfit{w} _ {k,f}^{\mathsf{H}} V _ {k,f} \mathbfit{w} _ {k,f}}}
$$

実装上は， $W _ {f}$ の $k$ 番目の行が $\mathbfit{w} _ {k,f}^{\mathsf{H}}$ になるように格納する．

上の更新は iterative projection (IP) と呼ばれる．
IP は $W _ {f} V _ {k,f}$ に対する線形方程式を解くため，式の意味を追いやすく，AuxIVA の基本形として理解しやすい．
一方で，同じ目的関数を最小化する別の更新則として iterative source steering (ISS) がある（[Scheibler and Ono, 2020](https://github.com/onolab-tmu/code_2020ICASSP_iss)）．
ISS は分離信号の 1 成分を使って他の成分を逐次的に打ち消す rank-1 update として書けるため，行列の逆行列計算を必要としない．

ISS では，現在の $y _ {j,f,t}$ と $\varphi(r _ {j,t})$ を固定し，source index $k$ を 1 つ選んで $y _ {j,f,t} \leftarrow y _ {j,f,t} - v _ {j,k,f} y _ {k,f,t}$ と更新する．
係数は，

$$
v _ {j,k,f} =
\begin{cases}
  \dfrac{\sum _ {t} \varphi(r _ {j,t}) y _ {j,f,t} y _ {k,f,t}^{\ast}}{\sum _ {t} \varphi(r _ {j,t}) \lvert y _ {k,f,t}\rvert ^2} & (j \neq k) \newline
  1 - \left(\frac{1}{T}\sum _ {t = 1} ^{T} \varphi(r _ {k,t}) \lvert y _ {k,f,t}\rvert ^2\right)^{-\frac{1}{2}} & (j = k)
\end{cases}
$$

とする．
分離信号の更新 $y _ {j,f,t} \gets y _ {j,f,t} - v _ {j,k,f} y _ {k,f,t}$ は，分離行列の行更新 $\mathbfit{w} _ {j,f}^{\mathsf{H}} \gets \mathbfit{w} _ {j,f}^{\mathsf{H}} - v _ {j,k,f}\mathbfit{w} _ {k,f}^{\mathsf{H}}$ と同じ意味を持つ．
このハンズオンでは，`auxiva.update_method` を `ip` または `iss` に変えることで，2 つの更新則を同じ pipeline で比較する．

ただし，ブラインド音源分離では分離信号のスケールが不定である．
そのため，AuxIVA で得られた分離信号をそのまま保存すると，音量が不自然になることがある．
このスケール不定性を補正する代表的な後処理が projection back である．
参照マイクを $m$ とし，分離信号 $y _ {k,f,t}$ が参照マイクの観測信号 $x _ {m,f,t}$ に近づくように，周波数ごと，音源ごとの複素スケール $a _ {k,f}$ を推定する．
具体的には， $f$ と $k$ を固定し， $a _ {k,f} y _ {k,f,t}$ が参照マイクの観測信号 $x _ {m,f,t}$ を最小二乗の意味で近似するようにする．

$$
a _ {k,f} = \operatorname*{argmin} _ {a \in \mathbb{C}} \sum _ {t=1}^{T} \lvert x _ {m,f,t} - a y _ {k,f,t} \rvert^2
$$

この目的関数を $a^{\ast}$ で微分して 0 とおくと，

$$
\sum _ {t=1}^{T} y _ {k,f,t}^{\ast} \left( x _ {m,f,t} - a _ {k,f} y _ {k,f,t} \right) = 0
$$

となる．
したがって，projection back の係数は

$$
a _ {k,f} = \frac{ \sum _ {t=1}^{T} x _ {m,f,t} y _ {k,f,t}^{\ast} }{ \sum _ {t=1}^{T} \lvert y _ {k,f,t}\rvert ^2 }
$$

で与えられる．
分母が非常に小さいと数値的に不安定になるため，実装では $\sum _ {t} \lvert y _ {k,f,t}\rvert ^2$ と `eps` の大きい方を分母として使う．
最終的な分離信号は $\tilde{y} _ {k,f,t} = a _ {k,f} y _ {k,f,t}$ とする．
この処理は分離行列 $W _ {f}$ を更新する処理ではなく，AuxIVA の反復が終わった後に分離信号のスケールだけを参照マイクに合わせる後処理である．

実装では `y` と `x` の shape はどちらも `(n_frames, n_freq, n_channels)` である．
したがって，周波数 index `f` と音源 index `k` を固定した `separated = y[:, f, k]` と，参照マイクの観測信号 `reference = x[:, f, reference_mic]` から `scale` を計算し，`y_scaled[:, f, k] = scale * separated` とする．

AuxIVA の入力と出力の仕様を疑似コードとして書くと，次のようになる．
ここでは，実装で使う配列 shape に合わせて `x` と `y` を frame, freq, channel の順に並べる．
分離行列 `w` は周波数ごとに持つため，shape は `(n_freq, n_channels, n_channels)` である．

```txt
input:
  x: complex ndarray with shape (n_frames, n_freq, n_channels)
     observed mixture spectrogram
  n_iter: int
     number of AuxIVA iterations
  model: "laplace" or "gauss"
     source model used to compute weights
  update_method: "ip" or "iss"
     update rule used for the demixing matrix
  eps: float
     small positive value for numerical stability
  reference_mic: int
     microphone index used for projection back

output:
  y: complex ndarray with shape (n_frames, n_freq, n_channels)
     separated spectrogram after projection back
  w: complex ndarray with shape (n_freq, n_channels, n_channels)
     demixing matrices

procedure:
  initialize w[f] as identity matrix for all frequency bins f

  repeat n_iter times:
    y <- demix x by w
    weights[t, k] <- source model weights from y[t, :, k]
    v[k, f] <- weighted covariance of x using weights[:, k]

    for each source k:
      for each frequency f:
        solve (w[f] v[k, f]) u = e[k]
        normalize u by sqrt(u^\mathsf{H} v[k, f] u)
        set row k of w[f] to u^\mathsf{H}

  y <- demix x by final w
  y <- project y back to reference_mic
  return y, w
```

#### AuxIVA の NumPy 実装

まずは設定ファイルや CLI を気にせず，数式に対応する NumPy 関数を書く．
ここでは，分離行列を適用する処理を `demix()`，音源モデルの重みを作る処理を `source_weights()`，IP 更新を `ip_update()`，ISS 更新を `iss_update()`，後処理を `project_back()`，一連の分離処理を `separate()` と名付ける．
`src/bss_handson/auxiva.py` を作る．

```py
# src/bss_handson/auxiva.py
from collections.abc import Callable

import numpy as np


def demix(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    n_frames, n_freq, n_channels = x.shape
    y = np.empty((n_frames, n_freq, n_channels), dtype=np.complex128)

    for t in range(n_frames):
        for f in range(n_freq):
            y[t, f] = w[f] @ x[t, f]

    return y


def source_weights(
    y: np.ndarray,
    model: str = "laplace",
    eps: float = 1.0e-10,
) -> np.ndarray:
    power_sum = np.sum(np.abs(y) ** 2, axis=1)
    r = np.sqrt(np.maximum(power_sum, eps))

    if model == "laplace":
        varphi = 1.0 / (2.0 * r)
    elif model == "gauss":
        varphi = y.shape[1] / (r**2)
    else:
        raise ValueError(f"model must be 'laplace' or 'gauss': {model}")

    return varphi


def weighted_covariance(
    x: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    n_frames, n_freq, n_channels = x.shape

    v = np.empty((n_channels, n_freq, n_channels, n_channels), dtype=np.complex128)
    for k in range(n_channels):
        for f in range(n_freq):
            v_kf = np.zeros((n_channels, n_channels), dtype=np.complex128)
            for t in range(n_frames):
                x_tf = x[t, f, :, None]
                v_kf += weights[t, k] * (x_tf @ x_tf.conj().T)
            v[k, f] = v_kf / n_frames

    return v


def objective(w: np.ndarray, v: np.ndarray) -> float:
    n_channels, n_freq = v.shape[:2]
    quadratic = 0.0

    for k in range(n_channels):
        for f in range(n_freq):
            wk = w[f, k].conj()
            quadratic += float(np.real(wk.conj() @ v[k, f] @ wk))

    logdet = 0.0
    for w_f in w:
        sign, logabsdet = np.linalg.slogdet(w_f)
        if sign == 0:
            return float("inf")
        logdet += logabsdet

    return float(quadratic - 2.0 * logdet)


def ip_update(
    x: np.ndarray,
    w: np.ndarray,
    model: str = "laplace",
    eps: float = 1.0e-10,
) -> np.ndarray:
    _, n_freq, n_channels = x.shape
    y = demix(x, w)
    weights = source_weights(y, model=model, eps=eps)
    v = weighted_covariance(x, weights)

    w_new = w.copy()

    for k in range(n_channels):
        for f in range(n_freq):
            eye_k = np.zeros(n_channels, dtype=np.complex128)
            eye_k[k] = 1.0
            wk = np.linalg.solve(w_new[f] @ v[k, f], eye_k)
            denom_sq = max(float(np.real(wk.conj() @ v[k, f] @ wk)), eps)
            w_new[f, k, :] = (wk / np.sqrt(denom_sq)).conj()

    return w_new


def iss_update(
    x: np.ndarray,
    w: np.ndarray,
    model: str = "laplace",
    eps: float = 1.0e-10,
) -> np.ndarray:
    _, n_freq, n_channels = x.shape
    y = demix(x, w)
    weights = source_weights(y, model=model, eps=eps)

    w_new = w.copy()
    y_new = y.copy()

    for k in range(n_channels):
        for f in range(n_freq):
            y_k = y_new[:, f, k].copy()
            w_k = w_new[f, k, :].copy()
            power_k = np.abs(y_k) ** 2

            for j in range(n_channels):
                denom = float(np.mean(weights[:, j] * power_k))
                denom = max(denom, eps)

                if j == k:
                    coeff = 1.0 - 1.0 / np.sqrt(denom)
                else:
                    numerator = np.mean(weights[:, j] * y_new[:, f, j] * y_k.conj())
                    coeff = numerator / denom

                y_new[:, f, j] -= coeff * y_k
                w_new[f, j, :] -= coeff * w_k

    return w_new


def project_back(
    y: np.ndarray,
    x: np.ndarray,
    reference_mic: int = 0,
    eps: float = 1.0e-10,
) -> np.ndarray:
    n_frames, n_freq, n_channels = y.shape
    y_scaled = np.empty_like(y)

    for f in range(n_freq):
        reference = x[:, f, reference_mic]
        for k in range(n_channels):
            separated = y[:, f, k]
            numerator = np.sum(reference * separated.conj())
            denominator = np.sum(np.abs(separated) ** 2)
            scale = numerator / max(float(denominator), eps)
            y_scaled[:, f, k] = scale * separated

    return y_scaled


def separate(
    x: np.ndarray,
    n_iter: int = 50,
    model: str = "laplace",
    update_method: str = "ip",
    eps: float = 1.0e-10,
    reference_mic: int = 0,
    callback: Callable[[int, np.ndarray], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    _, n_freq, n_channels = x.shape
    w = np.tile(np.eye(n_channels, dtype=np.complex128), (n_freq, 1, 1))
    update = {
        "ip": ip_update,
        "iss": iss_update,
    }.get(update_method)
    if update is None:
        raise ValueError(f"update_method must be 'ip' or 'iss': {update_method}")

    if callback is not None:
        callback(0, w.copy())

    for iteration in range(1, n_iter + 1):
        w = update(x, w, model=model, eps=eps)
        if callback is not None:
            callback(iteration, w.copy())

    y = demix(x, w)
    y = project_back(y, x, reference_mic=reference_mic, eps=eps)
    return y, w
```

この実装では，数式との対応を見やすくするために意図的に `for` 文を多く使っている．
たとえば `demix()` の `y[t, f] = w[f] @ x[t, f]` は，数式の $\mathbfit{y} _ {f,t} = W _ {f} \mathbfit{x} _ {f,t}$ と 1 対 1 に対応している．
同様に，`weighted_covariance()` の 3 重 loop は， $V _ {k,f} = \frac{1}{T}\sum _ {t} \varphi(r _ {k,t})\mathbfit{x} _ {f,t}\mathbfit{x} _ {f,t}^{\mathsf{H}}$ の和をそのまま書いている．

ただし，実行速度という観点では，Python の `for` 文を大量に回す実装は遅くなる．
実験規模が大きくなった段階では，発展編で述べるように，frame や周波数 bin の loop を NumPy のベクトル演算，broadcasting，batch matrix multiplication に置き換えることも検討する．

この実装では determined condition を仮定し，`x` と `y` の shape をどちらも `(n_frames, n_freq, n_channels)` としている．
数式では $y _ {k,f,t}$ と書いたが，実装では frame, freq, channel の順に並べる．
`source_weights` は AuxIVA の音源モデルに対応する重みを計算する関数であり，`model="laplace"` と `model="gauss"` を切り替えられる．
`weighted_covariance` は与えられた重みを使って $V _ {k,f}$ だけを計算する関数である．
このように分けておくと，たとえば別の音源モデルを試す場合でも，重みの計算部分だけを差し替えやすい．
`demix`, `source_weights`, `weighted_covariance`, `objective`, `ip_update`, `iss_update`, `project_back` を分けておくと，アルゴリズムの各部分を個別に確認できる．

`demix()` は，現在の分離行列 `w` を観測信号 `x` に適用して分離信号 `y` を作る．
ここで `w[f]` は周波数 bin `f` の分離行列であり，`x[t, f]` はその時刻 frame と周波数 bin におけるマイク方向のベクトルである．
したがって，`w[f] @ x[t, f]` は数式の $W _ {f} \mathbfit{x} _ {f,t}$ に対応する．
この関数を独立させると，AuxIVA の反復中だけでなく，最終的な分離信号の計算や重み計算の前処理にも同じ実装を使える．

`source_weights()` は，分離信号 `y` から時間 frame ごとの周波数方向の二乗和 `power_sum` を計算し，その平方根 `r` を音源モデルに応じた重み `varphi` へ変換する．
`power_sum = np.sum(np.abs(y) ** 2, axis=1)` と `r = np.sqrt(power_sum)` の shape はどちらも `(n_frames, n_channels)` である．
`model="laplace"` では `varphi = 1.0 / (2.0 * r)` を返す．
`model="gauss"` では，時変分散を `power_sum / n_freq` とみなすため，`varphi = n_freq / (r**2)` を返す．
`np.maximum(power_sum, eps)` は，無音に近い frame で重みが極端に大きくなることを避けるための数値安定化である．

`weighted_covariance()` は，重み `weights[t, k]` を使って，音源 index `k` と周波数 bin `f` ごとの重み付き共分散行列 `v[k, f]` を計算する．
内側の `x_tf @ x_tf.conj().T` は $\mathbfit{x} _ {f,t}\mathbfit{x} _ {f,t}^{\mathsf{H}}$ に対応する外積である．
この行列を frame 方向に足し合わせ，最後に `n_frames` で割ることで， $V _ {k,f}$ の時間平均を得る．

`ip_update()` は，1 回分の iterative projection 更新を行う．
まず現在の `w` で `y` を計算し，そこから重みと重み付き共分散行列を作る．
その後，音源 index `k` と周波数 bin `f` ごとに `np.linalg.solve(w_new[f] @ v[k, f], eye_k)` を解く．
これは明示的に逆行列を作るのではなく，線形方程式として $\left(W _ {f} V _ {k,f}\right)^{-1}\mathbfit{e} _ {k}$ を計算するための書き方である．
最後に `denom_sq` で正規化し，`w_new[f, k, :]` に分離行列の行ベクトルとして格納する．

`iss_update()` は，1 回分の iterative source steering 更新を行う．
IP と同じく現在の `w` から `y` と重みを計算するが，重み付き共分散行列を作らず，分離信号 `y_new[:, f, k]` を使って他の分離信号成分を逐次的に更新する．
`coeff` は数式の $v _ {j,k,f}$ に対応し，`y_new[:, f, j] -= coeff * y_k` で分離信号を更新し，同時に `w_new[f, j, :] -= coeff * w_k` で分離行列の行も同じ変換に合わせる．
`y_k` と `w_k` は更新前の値を使う必要があるため，内側の loop に入る前に `.copy()` している．
この実装により，`separate()` は `update_method="ip"` では `ip_update()`，`update_method="iss"` では `iss_update()` を呼び出す．

`project_back()` は，分離後のスケールを参照マイクに合わせる後処理である．
BSS では分離信号の順番とスケールは不定であるため，分離音を wav として保存すると音量が不自然になることがある．
ここでは，参照マイクの観測信号 `x[:, f, reference_mic]` を，分離信号 `y[:, f, k]` の複素スカラー倍で最小二乗近似する係数を求めている．
`separate()` はこれらの部品をまとめ，単位行列で初期化した分離行列を `n_iter` 回更新し，最後に projection back した分離信号と分離行列を返す．
`update_method` は `ip` または `iss` を受け取り，設定ファイルや command line override から更新則を切り替えるための引数である．
`callback` を渡した場合は，0 回目の初期値と各更新後の分離行列を呼び出し側へ渡す．
AuxIVA の中では評価指標や保存先を知らなくてよいため，分離性能の途中評価は `scripts/run_bss.py` 側で行う．

#### 関数の配置と呼び出し

ここまでに実装した関数は，役割ごとにファイルを分けて配置する．
研究コードでは，1 つの巨大な `main.py` にすべてを書くと，どこまでがデータ取得で，どこからがアルゴリズム本体で，どこが評価なのかわからなくなる．
関数の責務ごとにファイルを分けておくと，処理を個別に確認しやすく，後から別の実験で再利用しやすい．

今回の配置は次の通りである．

```txt
src/
└── bss_handson/
    ├── __init__.py
    ├── auxiva.py
    ├── cli.py
    ├── config.py
    ├── data.py
    ├── evaluation.py
    ├── plot.py
    ├── plot_style.py
    ├── simulation.py
    └── stft.py
```

各ファイルの役割は以下である．

| ファイル        | 役割                                               |
| :-------------- | :------------------------------------------------- |
| `config.py`     | OmegaConf による設定ファイルと override の読み込み |
| `data.py`       | CMU ARCTIC の取得と音源信号の整形                  |
| `simulation.py` | pyroomacoustics による混合信号の生成               |
| `stft.py`       | STFT / ISTFT                                       |
| `auxiva.py`     | AuxIVA，IP/ISS 更新，目的関数，projection back     |
| `evaluation.py` | fast-bss-eval による性能評価                       |
| `plot.py`       | matplotlib による可視化                            |
| `plot_style.py` | matplotlib の axis style 適用                      |
| `cli.py`        | `scripts/run_bss.py` を呼ぶ最小限の entry point    |

`src/` に置く関数は，できるだけ小さく保つ．
たとえば `simulate_room()` は音源信号と部屋条件から混合信号を返し，`separate()` は観測スペクトログラムから分離スペクトログラムと分離行列を返す．
一方で，データ取得，シミュレーション，STFT，分離，評価，保存を一気に実行するような複雑な手順は `scripts/run_bss.py` に置く．
この分け方により，`src/` は再利用しやすい処理の集まりになり，`scripts/` は実験ごとの組み合わせを表す場所になる．

配置の基準を具体化すると次のようになる．

| 置き場所   | 置くコード                          | 例                                                                          |
| :--------- | :---------------------------------- | :-------------------------------------------------------------------------- |
| `src/`     | 入力と出力が明確な単機能の処理      | `stft_channels()`, `simulate_room()`, `separate()`, `evaluate_separation()` |
| `src/`     | 複数の実験で import して使う処理    | `source_weights()`, `project_back()`, `save_spectrograms()`                 |
| `scripts/` | `src/` の処理を組み合わせた実験手順 | `run_bss.py`, `evaluate_random_bss.py`                                      |
| `scripts/` | 結果ファイルの集計や図の作成        | `show_bss_metrics.py`, `plot_bss_metrics.py`                                |
| `scripts/` | 設定ファイルや実験条件の生成        | `create_bss_comparison_configs.py`                                          |

`src/` の関数は，できるだけ `results/` の具体的なディレクトリ名や，今回だけの実験名を知らないようにする．
逆に `scripts/` は，`results/bss_update_ip` のような出力先，`bss_update_*/metrics.json` のような glob pattern，複数条件を回す for loop を持ってよい．
この境界を守ると，研究が進んで実験 script が増えても，信号処理や評価の基本部品を壊さずに使い回せる．

#### `__init__.py`

`__init__.py` は，そのディレクトリを Python package として扱うためのファイルである．
古い Python では必須であり，現在でも明示的に置いておく方がわかりやすい．
最小限であれば空ファイルでもよい．

一方で，package の外からよく使う関数だけを `__init__.py` で公開しておくと，notebook や別の script から短く import できる．
たとえば今回の `bss_handson` では，AuxIVA に関する主要な関数だけを公開 API として並べる．

```py
# src/bss_handson/__init__.py
"""Blind source separation demo package."""

from bss_handson.auxiva import (
    separate,
    demix,
    project_back,
    source_weights,
    weighted_covariance,
    objective,
    ip_update,
    iss_update,
)

__all__ = [
    "separate",
    "demix",
    "project_back",
    "source_weights",
    "weighted_covariance",
    "objective",
    "ip_update",
    "iss_update",
]
```

`__all__` は，この package から外部に公開する名前を明示するためのリストである．
何でも `__init__.py` に import すればよいわけではない．
重い処理，ファイル I/O，データダウンロード，matplotlib の設定などを `__init__.py` で実行してしまうと，`import bss_handson` しただけで余計な処理が走る．
そのため，`__init__.py` には軽い import と package のメタ情報だけを書くのがよい．

このように書いておくと，次のように import できる．

```py
from bss_handson import separate
```

ただし，コードの所在を明確にしたい場合は，これまで通り `from bss_handson.auxiva import separate` と書いてもよい．

依存関係は，基本的に `scripts/run_bss.py` が `src/` の小さな module を呼び出す形にする．
たとえば `auxiva.py` から `plot.py` を import したり，`stft.py` から `data.py` を import したりしない．
数値計算の中心部分はできるだけ独立させ，ファイル I/O や設定ファイルの読み込みは script 側に寄せる．

呼び出しの流れは次の通りである．

```txt
configs/bss.yaml
  ↓ load_config()
load_cmu_arctic_sources()
  ↓
simulate_room()
  ↓
stft_channels()
  ↓
separate()
  ├─ ip_update()
  ├─ iss_update()
  ├─ demix()
  └─ project_back()
  ↓
istft_sources()
  ↓
evaluate_separation()
  ↓
save_spectrograms()
```

このような流れにすると，たとえば AuxIVA だけを確認したい場合は `auxiva.py` だけを import すればよい．
部屋シミュレーションだけを差し替えたい場合も，`simulation.py` の関数を変えればよい．

Python コード上では，`scripts/run_bss.py` から次のように呼び出す．

```py
from bss_handson.auxiva import separate
from bss_handson.config import load_config
from bss_handson.data import load_cmu_arctic_sources
from bss_handson.evaluation import evaluate_separation
from bss_handson.plot import save_spectrograms
from bss_handson.simulation import simulate_room
from bss_handson.stft import istft_sources, stft_channels
```

notebook から一部の関数だけを使いたい場合も同様である．
たとえば，混合信号を作らずに人工的な複素スペクトログラムで AuxIVA の挙動だけを調べたい場合は次のようにする．

```py
import numpy as np

from bss_handson.auxiva import (
    ip_update,
    iss_update,
    objective,
    source_weights,
    demix,
    weighted_covariance,
)

rng = np.random.default_rng(0)
x = rng.normal(size=(40, 17, 2)) + 1j * rng.normal(size=(40, 17, 2))
w = np.tile(np.eye(2, dtype=np.complex128), (17, 1, 1))

y = demix(x, w)
weights = source_weights(y, model="laplace")
v = weighted_covariance(x, weights)
before = objective(w, v)
w = ip_update(x, w, model="laplace")
after = objective(w, v)
print(before, after)

w = iss_update(x, w, model="laplace")
print(w.shape)
```

この例のように，アルゴリズム本体を CLI から切り離しておくと，notebook での確認や実験 CLI から同じ関数を呼び出せる．
これが `src layout` で package として実装する大きな利点である．

#### 性能評価

分離結果の評価には `fast-bss-eval` を使う．
BSS では，分離信号の順番とスケールは不定である．
そのため，評価時には permutation を許して，参照信号と推定信号の対応を決める必要がある．

`src/bss_handson/evaluation.py` を作る．

```py
# src/bss_handson/evaluation.py
import numpy as np
from fast_bss_eval import bss_eval_sources


def evaluate_separation(
    references: np.ndarray,
    estimates: np.ndarray,
) -> dict[str, list[float]]:
    length = min(references.shape[1], estimates.shape[1])
    references = references[:, :length]
    estimates = estimates[:, :length]

    sdr, sir, sar, perm = bss_eval_sources(
        references,
        estimates,
        compute_permutation=True,
    )
    return {
        "sdr": np.asarray(sdr).tolist(),
        "sir": np.asarray(sir).tolist(),
        "sar": np.asarray(sar).tolist(),
        "perm": np.asarray(perm).tolist(),
    }
```

`evaluate_separation()` は，参照信号 `references` と推定信号 `estimates` を受け取り，`fast_bss_eval.bss_eval_sources()` を本資料の実験コードから呼びやすい形に wrap する関数である．
シミュレーションや ISTFT の都合で参照信号と推定信号の長さが完全には一致しないことがあるため，まず短い方の長さに合わせて切り詰める．
`compute_permutation=True` にしているのは，BSS の出力順序が不定だからである．
たとえば `estimated_0.wav` が必ず `source_0.wav` に対応するとは限らないため，評価時には最も対応がよい permutation を選んで SDR, SIR, SAR を計算する．
返り値は JSON に保存しやすいように，NumPy 配列ではなく Python の `list` に変換している．

`metrics.json` には，たとえば次のような内容が保存される．
次の数値は，上記の標準設定を実際に実行した `results/bss_example/metrics.json` の内容である．

```json
{
  "sdr": [6.827044740214696, 8.735539422581354],
  "sir": [10.334476721757897, 13.65016346195846],
  "sar": [9.775881045127534, 10.609952728138726],
  "perm": [1, 0]
}
```

JSON で保存しておくと，あとから Python で読み直して条件間の比較表を作りやすい．
また，テキストファイルなので git diff やエディタで内容を確認しやすく，`config.yaml` と同じ出力ディレクトリに置くことで，どの実験条件から得られた評価値なのかを追跡しやすい．
大量の実験結果を集計する場合も，`results/*/metrics.json` を順番に読み込めば，CSV や図に変換できる．
このとき，評価結果は tidy data の形に変換してから扱うとよい．
tidy data は，1 つの変数を 1 つの列，1 つの観測を 1 つの行，1 種類の観測単位を 1 つの表として整理する考え方である（[Wickham, 2014](https://www.jstatsoft.org/v59/i10/)，[私たちのR: 整然データ構造](https://www.jaysong.net/RBook/tidydata.html)）．
特に後者は，日本語で messy data と tidy data を対比し，wide 型から long 型への変換を説明しているため，ここで扱う `metrics.json` の集計表を理解する補助資料として読みやすい．

たとえば，AuxIVA の分離行列更新手法として IP と ISS を比較する場合を考える．
SDR を次のような表にすると，人間には読みやすいが，作図コードでは扱いにくい．

| update_method | source_0_sdr | source_1_sdr |
| ------------- | -----------: | -----------: |
| ip            |         6.83 |         8.74 |
| iss           |         6.54 |         8.58 |

この表では，`source_0_sdr` や `source_1_sdr` のように，音源 index と評価指標が列名に埋め込まれている．
音源数が増えると列が増え，`seaborn` で「音源 index を色で分ける」といった指定がしにくくなる．
tidy data として整理すると，次のようになる．

| update_method | source |  sdr |
| ------------- | -----: | ---: |
| ip            |      0 | 6.83 |
| ip            |      1 | 8.74 |
| iss           |      0 | 6.54 |
| iss           |      1 | 8.58 |

この表では，`update_method`，`source`，`sdr` がそれぞれ列になっている．
1 行は「1 つの更新手法における 1 つの音源の SDR」を表す．
この形にしておくと，`seaborn` で `x="update_method"`，`y="sdr"`，`hue="source"` のように列名を指定するだけで，IP と ISS の SDR 比較図を作れる．

tidy data にしておくと，`pandas` の `groupby()` でも同じ列名を使って集計できる．
たとえば，更新手法ごとの平均 SDR を見るなら，`update_method` で group by して `sdr` の平均を取る．

```py
df.groupby("update_method")["sdr"].mean()
```

この結果は，概念的には次のような表になる．

| update_method | mean_sdr |
| ------------- | -------: |
| ip            |     7.78 |
| iss           |     7.56 |

音源ごとの差も残したい場合は，`update_method` と `source` の 2 つを group by の key にする．

```py
df.groupby(["update_method", "source"])["sdr"].mean()
```

この場合は，次のように「更新手法と音源 index の組」に対して平均 SDR が対応する．

| update_method | source | mean_sdr |
| ------------- | -----: | -------: |
| ip            |      0 |     6.83 |
| ip            |      1 |     8.74 |
| iss           |      0 |     6.54 |
| iss           |      1 |     8.58 |

`seaborn.catplot(data=df, x="update_method", y="sdr", hue="source")` は，この group by の考え方とよく対応している．
`x="update_method"` は横軸方向の group，`hue="source"` は色分けする group，`y="sdr"` は描画する値である．
つまり，tidy data では，`pandas` で集計するときの key と，`seaborn` で軸や色に割り当てる列が同じになる．
この対応があるため，実験条件や音源数が増えても，列名を指定するだけで集計と作図を同じ考え方で扱える．

#### 可視化

分離結果は音を聴くことも重要だが，波形やスペクトログラムを確認するとバグに気づきやすい．
ここでは混合信号と分離信号のスペクトログラムを保存する．

`src/bss_handson/plot.py` を作る．

```py
# src/bss_handson/plot.py
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bss_handson.stft import create_stft


def _indexed_path(path: str | Path, index: int) -> Path:
    path = Path(path)
    return path.with_name(f"{path.stem}_{index}{path.suffix}")


def save_spectrograms(
    signals: np.ndarray,
    fs: int,
    output_path: str | Path,
    title: str,
    style: str | list[str],
    window: str,
    win_length: int,
    hop: int,
    vmin_db: float,
    vmax_db: float,
) -> None:
    plt.style.use(style)
    stft = create_stft(fs=fs, window=window, win_length=win_length, hop=hop)
    freqs = stft.f
    times = stft.t(signals.shape[1])
    spectra = stft.stft(signals, axis=-1)

    for index, spectrum in enumerate(spectra):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        power_db = 20.0 * np.log10(np.maximum(np.abs(spectrum), 1.0e-10))
        image = ax.imshow(
            power_db,
            origin="lower",
            aspect="auto",
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            vmin=vmin_db,
            vmax=vmax_db,
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"{title} {index}")
        fig.colorbar(image, ax=ax, label="Magnitude (dB)")
        fig.tight_layout()
        fig.savefig(_indexed_path(output_path, index))
        plt.close(fig)
```

`save_spectrograms()` は，複数チャンネルまたは複数音源の信号を，信号ごとに別々のスペクトログラム画像として保存する関数である．
`signals` の shape は `(n_signals, n_samples)` を想定し，先頭軸を保存する図の index として使う．
たとえば `output_path` に `mixture_spectrogram.png` を渡すと，`mixture_spectrogram_0.png`，`mixture_spectrogram_1.png` のように index 付きのファイル名で保存する．
`ShortTimeFFT.stft(signals, axis=-1)` と書くことで，各信号の時間軸に沿って STFT をまとめて計算できる．
可視化では `20 * log10(abs(spectrum))` によって振幅スペクトログラムを dB 表示にし，`vmin_db` と `vmax_db` で色の範囲を固定する．
色の範囲を固定しておくと，条件を変えた複数の図を見比べるときに，単なるカラースケールの違いを性能差と誤解しにくい．
`imshow` には `extent` を指定し，横軸を秒，縦軸を Hz として読めるようにしている．
`plt.style.use(style)` は，Matplotlib の style file を読み込み，フォントサイズ，線幅，グリッド，colormap，保存時 dpi などの既定値をまとめて変更する．
`plt.style.use()` には style file のリストも渡せるため，共通設定と個別設定を分けて，`plt.style.use(["styles/common.mplstyle", "styles/paper.mplstyle"])` のように読み込める．
後に指定した style が前の style の設定を上書きする．
論文向けの図とスライド向けの図では，必要な文字サイズや線幅が異なるため，コード中に `plt.rcParams[...] = ...` を散らすのではなく，style file として分けておくと管理しやすい．
図を載せる媒体の大きさを先に決め，Matplotlib 側の `figure.figsize` と `font.size` をそれに合わせることが重要である（[科学技術論文に用いる図表のための matplotlib 設定](https://qiita.com/n-taishi/items/042039847601f264236d)）．
Matplotlib のデフォルト設定は，汎用的な確認には使えるが，論文や発表の図として最適化されているわけではない．
図の媒体，読者，主張に合わせて，図のサイズ，フォント，線幅，grid，色を調整する（[Rougier et al., 2014](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003833)）．[^matplotlib-default]
最後に `plt.close(fig)` を呼ぶのは，大量の実験を連続実行したときに figure がメモリ上に残り続けることを避けるためである．

[^matplotlib-default]: Matplotlib のデフォルトを何も工夫せずに論文や発表に貼り付けるのは怠慢である．意外にもプロの研究者でもできていないことが多いが，媒体と読者に合わせた調整は当然の作業である．

`styles/` ディレクトリを作成し，論文向けとスライド向けの style file を作る．
`.mplstyle` は `key: value` の形式で Matplotlib の rcParams を書くファイルである．
style file で指定できる項目は Matplotlib の rcParams と対応しており，公式ドキュメントでは style sheet の使い方と組み込み style の一覧が説明されている（[Customizing Matplotlib with style sheets and rcParams](https://matplotlib.org/stable/users/explain/customizing.html)，[Style sheets reference](https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html)）．
ここでは共通設定として，x 軸と y 軸の tick を内向きにし，グリッドを破線にする．
論文向けはセリフ体，図の大きさ 80 mm x 50 mm，フォントサイズ 10 pt とする．
プレゼン向けはサンセリフ体，図の大きさ 200 mm x 120 mm，フォントサイズ 24 pt とする．
Matplotlib の `figure.figsize` は inch 単位なので，mm を 25.4 で割った値を書く．
`axes.titlesize`，`axes.labelsize`，`xtick.labelsize`，`ytick.labelsize`，`legend.fontsize` などは，Matplotlib のデフォルトでは `font.size` に対する相対的な指定になっている．
そのため，ここでは個別に同じ値を書かず，`font.size` だけを媒体に合わせて変更する．
自分で style file を書く代わりに，既存の style package を出発点にする方法もある．
たとえば SciencePlots は，科学論文，発表，学位論文向けの Matplotlib style を提供する package である（[SciencePlots](https://github.com/garrettj403/SciencePlots)）．
SciencePlots のような外部 style は便利だが，投稿先の規定，使用フォント，日本語表示，LaTeX 依存の有無を確認した上で使う．
この資料では，どの設定が図に効いているかを学ぶため，最小限の `.mplstyle` を自分で書く．

```bash
mkdir -p styles
```

```txt
# styles/common.mplstyle
axes.grid: True
grid.linestyle: --
grid.linewidth: 0.6
grid.alpha: 0.4
xtick.direction: in
ytick.direction: in
pdf.fonttype: 42
ps.fonttype: 42
```

```txt
# styles/paper.mplstyle
figure.figsize: 3.149606, 1.968504
figure.dpi: 120
savefig.dpi: 300
font.family: serif
font.size: 10
lines.linewidth: 1.0
image.cmap: viridis
```

```txt
# styles/slide.mplstyle
figure.figsize: 7.874016, 4.724409
figure.dpi: 120
savefig.dpi: 200
font.family: sans-serif
font.size: 24
lines.linewidth: 2.4
image.cmap: magma
```

`seaborn` で作った図では，style file の設定が axis に十分反映されない場合がある．
そのため，axis に対する共通の見た目調整を `src/bss_handson/plot_style.py` にまとめておく．

```py
# src/bss_handson/plot_style.py
import matplotlib.pyplot as plt


def apply_axis_style(ax) -> None:
    ax.tick_params(direction=plt.rcParams["xtick.direction"])
    ax.grid(
        visible=plt.rcParams["axes.grid"],
        linestyle=plt.rcParams["grid.linestyle"],
        linewidth=plt.rcParams["grid.linewidth"],
        alpha=plt.rcParams["grid.alpha"],
    )
```

標準設定では `plot.style` に `styles/common.mplstyle` と `styles/paper.mplstyle` をこの順に書き，論文向けの図を出す．
スライド向けに大きな文字の図を出したい場合は，設定ファイルを書き換えずに command line override で style file だけを差し替える．

```bash
uv run bss-handson --config configs/bss.yaml plot.style='[styles/common.mplstyle,styles/slide.mplstyle]' output_dir=results/bss_slide_style
```

#### パラメータ直書きから設定ファイル駆動への変換

最初にアルゴリズムを実装するときは，次のようにパラメータを直書きしてもよい．

```py
y, w = separate(x, n_iter=50, model="laplace", eps=1.0e-10)
```

しかし，研究では `n_iter=20`, `50`, `100` のような比較を頻繁に行う．
同様に，音源モデルを `laplace` と `gauss` で切り替えたい場合もある．
そのたびに Python コードを書き換えると，実験条件と実装変更が混ざってしまう．
そこで，関数自体は引数で parametrized にしておき，CLI では設定ファイルから値を読む．

Python では，辞書を `**` で展開して keyword arguments として渡せる．
たとえば `params = {"n_iter": 50, "model": "laplace", "eps": 1.0e-10}` のとき，`separate(x, **params)` は `separate(x, n_iter=50, model="laplace", eps=1.0e-10)` と同じ意味になる．
これは設定ファイルの key と関数の引数名が完全に対応している場合に便利である．
一方で，余計な key が入っていると `TypeError` になるため，設定ファイルの key と関数の引数名を対応させておく必要がある．
このハンズオンでは，`dataset`, `room`, `stft`, `auxiva`, `plot` の key を関数引数名に合わせ，CLI ではできるだけ `**` 展開で渡す．
ただし，`sources`，`mixture`，`fs`，`output_path`，`title` のように実行中に決まる値や，複数の設定を合成して渡す値は明示的に渡す．

```py
y, w = separate(x, **config["auxiva"])
```

この変換は非常に重要である．
アルゴリズムの実装は `src/` に置き，実験条件は `configs/` に置く．
そうすれば，同じ実装に対して設定ファイルだけを変えて実験を回せる．

#### BSS pipeline と CLI

最後に，すべての処理を 1 つの CLI から実行できるようにする．
ただし，`src/` には単純な関数を置き，複雑な実験手順は `scripts/` に置く．
この方針にすると，`src/` の関数は別の実験でも再利用しやすく，`scripts/` はその実験で何をどう組み合わせたのかを表す場所になる．

`scripts/run_bss.py` は，設定を受け取り，`src/` の関数を組み合わせて標準 BSS 実験を実行する script である．
この script は，データ取得，音響シミュレーション，STFT，AuxIVA，ISTFT，評価，保存を順番に接続する．
このような複雑な手順は，`src/` ではなく `scripts/` に置く．

```py
# scripts/run_bss.py
import argparse
import json
from pathlib import Path

import soundfile as sf

from bss_handson.auxiva import demix, project_back, separate
from bss_handson.config import (
    get_auxiva_config,
    get_dataset_config,
    get_output_dir,
    get_plot_config,
    get_room_config,
    get_stft_config,
    load_config,
    save_config,
)
from bss_handson.data import load_cmu_arctic_sources
from bss_handson.evaluation import evaluate_separation
from bss_handson.plot import save_spectrograms
from bss_handson.simulation import simulate_room
from bss_handson.stft import istft_sources, stft_channels


def run_separation_pipeline(config: dict, evaluation_iterations: set[int] | None = None) -> dict:
    dataset_config = get_dataset_config(config)
    room_config = get_room_config(config)
    stft_config = get_stft_config(config)
    auxiva_config = get_auxiva_config(config)

    sources, source_fs = load_cmu_arctic_sources(**dataset_config)
    fs = room_config["fs"]
    if source_fs != fs:
        raise ValueError(f"source fs and room fs must match: {source_fs} != {fs}")

    mixture = simulate_room(sources=sources, **room_config)
    x, _, _ = stft_channels(mixture, fs=fs, **stft_config)
    y, w = separate(x, **auxiva_config)
    estimates = istft_sources(y, fs=fs, n_samples=mixture.shape[1], **stft_config)
    metrics = evaluate_separation(sources, estimates)

    return {
        "fs": fs,
        "sources": sources,
        "mixture": mixture,
        "spectra": x,
        "separated_spectra": y,
        "demixing_matrix": w,
        "estimates": estimates,
        "metrics": metrics,
        "iteration_metrics": [],
    }


def save_json(data, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_indexed_audio(output_dir: Path, prefix: str, signals, fs: int) -> None:
    for index, signal in enumerate(signals):
        sf.write(output_dir / f"{prefix}_{index}.wav", signal, fs)


def save_run_outputs(output_dir: Path, config: dict, result: dict) -> None:
    fs = result["fs"]
    save_indexed_audio(output_dir, "source", result["sources"], fs)
    save_indexed_audio(output_dir, "mixture", result["mixture"], fs)
    save_indexed_audio(output_dir, "estimated", result["estimates"], fs)

    spectrogram_config = {**get_stft_config(config), **get_plot_config(config)}
    save_spectrograms(
        result["mixture"],
        fs=fs,
        output_path=output_dir / "mixture_spectrogram.png",
        title="Mixture",
        **spectrogram_config,
    )
    save_spectrograms(
        result["estimates"],
        fs=fs,
        output_path=output_dir / "estimated_spectrogram.png",
        title="Estimated sources",
        **spectrogram_config,
    )

    save_config(config, output_dir / "config.yaml")
    save_json(result["metrics"], output_dir / "metrics.json")
    save_json(result["iteration_metrics"], output_dir / "iteration_metrics.json")


def run_bss(config_path: str | Path, overrides: list[str] | None = None) -> None:
    config = load_config(config_path, overrides)
    output_dir = get_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    auxiva_config = get_auxiva_config(config)
    evaluation_iterations = set(range(0, auxiva_config["n_iter"] + 1, 10))
    result = run_separation_pipeline(
        config,
        evaluation_iterations=evaluation_iterations,
    )
    save_run_outputs(output_dir, config, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    run_bss(args.config, args.overrides)
```

上のコードブロックは要点だけを示している．
実際の `scripts/run_bss.py` では，`evaluation_iterations` が指定された場合に，0, 10, 20, 30, 40, 50 回目の分離行列から途中評価を計算し，`iteration_metrics` として返す．
反復回数ごとの性能を調べるために `uv run bss-handson ... auxiva.n_iter=10` などを何度も実行する必要はなく，1 回の実行中に callback で途中結果を評価して保存する．

`src/bss_handson/cli.py` は，互換性のために `uv run bss-handson` から `scripts/run_bss.py` を呼ぶ中継用 wrapper にする．
この wrapper は実験本体を持たない．

```py
# src/bss_handson/cli.py
from pathlib import Path
import runpy


def main() -> None:
    script_path = Path.cwd() / "scripts" / "run_bss.py"
    if not script_path.exists():
        raise RuntimeError(
            "scripts/run_bss.py が見つかりません．"
            "pyproject.toml のある bss-handson/ で実行してください．"
        )
    runpy.run_path(str(script_path), run_name="__main__")
```

この構成では，実験本体は `scripts/run_bss.py` にあり，`src/bss_handson/cli.py` は command line entry point を維持するための wrapper にすぎない．
`evaluate_random_bss.py` のような別 script も `scripts/run_bss.py` の `run_separation_pipeline()` を再利用できるため，標準実行と random evaluation で処理内容がずれにくい．
また，`parser.add_argument("overrides", nargs="*")` は，`--config` で指定した設定ファイルの後ろに並ぶ任意個数の `key=value` を受け取るための指定である．
この引数を `load_config(config_path, overrides)` に渡すことで，`auxiva.n_iter=10` のような変更を Python コードを書き換えずに反映できる．

#### CLI コマンドの登録

`pyproject.toml` に次のような設定を追加する．
`[build-system]` は `uv init --lib` の時点で作成されているはずなので，消さずに残しておく．

```toml
# pyproject.toml
[project.scripts]
bss-handson = "bss_handson.cli:main"

[build-system]
requires = ["uv_build>=0.11.20,<0.12.0"]
build-backend = "uv_build"
```

左辺の `bss-handson` はコマンド名であり，右辺の `bss_handson.cli:main` は `bss_handson/cli.py` の `main()` 関数を指している．
つまり，配布パッケージ名が `bss-handson` の場合，通常の Python import 名はハイフンをアンダースコアに変えた `bss_handson` になる．
この教材では，`bss_handson.cli:main` は `scripts/run_bss.py` を呼ぶ wrapper であり，実験手順そのものは `scripts/run_bss.py` に置く．
`src/bss_handson/__init__.py` が存在しない場合や，右辺の module 名を間違えた場合，`uv run bss-handson` は実行できない．
また，`[build-system]` がない場合，`uv` はこの project を package として install できないため，`[project.scripts]` に書いたコマンドも生成されない．
このため，CLI を持つ教材プロジェクトは `uv init --lib` で初期化し，`[build-system]` を含む package として扱う．

これで次のように実行できる．

```bash
uv run bss-handson --config configs/bss.yaml
```

実行後，`results/bss_example/` に次のようなファイルが保存される．

```txt
results/bss_example/
├── config.yaml
├── estimated_<k>.wav
├── estimated_spectrogram_<k>.png
├── metrics.json
├── mixture_<m>.wav
├── mixture_spectrogram_<m>.png
└── source_<k>.wav
```

音声を聴き，スペクトログラムを見て，さらに `metrics.json` の SDR などを確認する．
`metrics.json` は機械的に読み込める評価結果なので，複数条件の結果をあとからまとめて比較する入口になる．
このようにして，実験の入力，条件，出力，評価を 1 つのディレクトリにまとめる．
条件を変える実験では，出力先も条件ごとに変える．
たとえば標準設定と反復回数を変えた実験は，次のように別ディレクトリへ保存する．

```txt
results/
├── bss_example/
│   ├── config.yaml
│   ├── metrics.json
│   └── ...
└── bss_niter_10/
    ├── config.yaml
    ├── metrics.json
    └── ...
```

実行時に最終的な設定を出力ディレクトリへ保存しておくと，後から音声，図，評価値を見たときに，どの条件で作られた結果なのか確認しやすい．
さらに余裕があれば，git commit hash，実行日時，実行コマンドもログとして残す．
ただし，実験結果をすべて git に入れる必要はない．
大量の結果は `results/` に保存し，必要なものだけ論文用ディレクトリや共有ストレージに移す．
git は「結果を再生成するためのコードと設定」を管理する場所である．

CLI コマンドを登録したら，まず標準設定で BSS パイプラインを 1 回実行する．
この実行は，entry point，設定ファイルの読み込み，データ取得，音響シミュレーション，AuxIVA，評価，可視化が一通り接続できているかを確認するための最小確認である．
実行前に，現在の作業ディレクトリが `bss-handson/` であり，`configs/bss.yaml` が存在することを確認する．

```bash
uv run bss-handson --config configs/bss.yaml
```

標準設定だけでは，パラメータを変えたときに結果がどう変わるかを比較できない．
OmegaConf の dot-list override を使うと，設定ファイルを増やさずに一部の値だけを command line から変更できる．
たとえば `auxiva.n_iter=10` は，`configs/bss.yaml` の `auxiva` の下にある `n_iter` だけを 10 に上書きする．
同時に `output_dir=results/bss_niter_10` も指定しておくと，標準設定の結果を上書きせず，条件ごとに別ディレクトリへ保存できる．

```bash
uv run bss-handson --config configs/bss.yaml auxiva.n_iter=10 output_dir=results/bss_niter_10
```

複数条件をまとめて実行する場合も，同じ `configs/bss.yaml` に対して override だけを変えればよい．
ここでは，AuxIVA の反復回数，音源モデル，更新則，projection back の参照マイクを変更して実行する．

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

実行後に結果を確認するときも，すべての `metrics.json` をまとめて読むのではなく，今比較したい系列だけを読む．
たとえば反復回数の比較なら `bss_niter_*`，音源モデルの比較なら `bss_model_*`，更新則の比較なら `bss_update_*` に限定する．

```bash
uv run python scripts/show_bss_metrics.py --pattern "bss_niter_*/metrics.json"
uv run python scripts/show_bss_metrics.py --pattern "bss_model_*/metrics.json"
uv run python scripts/show_bss_metrics.py --pattern "bss_update_*/metrics.json"
uv run python scripts/show_bss_metrics.py --pattern "bss_refmic_*/metrics.json"
```

この書き方では，実装も標準設定ファイルも変更しない．
実験条件の差分は command line に現れ，実行後には override 済みの最終設定が `results/<experiment>/config.yaml` に保存される．
そのため，あとから結果ディレクトリだけを見ても，標準設定から何を変えた実験なのか確認できる．
list を override する場合は shell に解釈されないように引用符で囲み，たとえば `dataset.speakers='[bdl,slt]'` のように指定する．

この資料では，仕組みが見えるように shell の `for` 文と OmegaConf の override だけで parameter sweep を行っている．
実験条件の数が増えた場合は，Hydra の `--multirun` や `hydra.mode=MULTIRUN` により，複数の設定値の組合せを command line から sweep できる（[Hydra: Multi-run](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/)）．
実験結果を UI で比較したい場合は，MLflow を使う選択肢もある（[MLflow: Tracking Hyperparameter Tuning](https://mlflow.org/docs/latest/ml/getting-started/hyperparameter-tuning/)）．
詳しくは発展編で扱う．
ただし，最初からこれらを導入すると，設定管理や実験追跡の仕組みそのものを学ぶ負担が増える．
このハンズオンでは，まず `configs/bss.yaml`，command line override，`results/<experiment>/config.yaml`，`metrics.json` という最小構成を理解することを優先する．

各実験の `metrics.json` を一覧表示する補助 script を作る．
結果ディレクトリを開いて 1 つずつ確認してもよいが，同じ形式の `metrics.json` をまとめて表示する script があると，条件間の比較がしやすい．

```py
# scripts/show_bss_metrics.py
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--pattern", required=True)
    args = parser.parse_args()

    paths = sorted(Path(args.results_dir).glob(args.pattern))
    if not paths:
        raise RuntimeError(f"{args.results_dir}/{args.pattern} が見つかりません")

    for path in paths:
        metrics = json.loads(path.read_text())
        print(path.parent.name, metrics)


if __name__ == "__main__":
    main()
```

`show_bss_metrics.py` は，`results/` 以下から `--pattern` に一致する `metrics.json` だけを探して表示する．
実験結果を集計するときは，何を比較しているのかを明示する必要がある．
たとえば，反復回数の比較，音源モデルの比較，更新則の比較を同じ一覧に混ぜると，数字は表示できても比較の意味が曖昧になる．
そのため，この script では `--pattern` を必須にし，集計したい条件を command line で明示する．
`Path.glob()` を使うと，shell の wildcard に近い書き方でファイルを集められる．
`sorted()` をかけているのは，表示順を毎回安定させるためである．
この script は評価値を整形しているだけで，実験結果そのものは変更しない．
このような確認用 script は，何度実行しても同じ結果を読むだけの処理にしておくと扱いやすい．

```bash
uv run python scripts/show_bss_metrics.py --pattern "bss_niter_*/metrics.json"
uv run python scripts/show_bss_metrics.py --pattern "bss_model_*/metrics.json"
uv run python scripts/show_bss_metrics.py --pattern "bss_update_*/metrics.json"
uv run python scripts/show_bss_metrics.py --pattern "bss_refmic_*/metrics.json"
```

評価結果を図として比較する補助 script を作る．
`metrics.json` は実験ごとに 1 ファイルとして保存されているが，そのままでは `seaborn` に渡しにくい．
ここでは異なる評価指標を横に並べるのではなく，条件が異なる実験を横軸に並べ，SDR だけを比較する．
たとえば AuxIVA の音源モデルを変えた条件，STFT パラメータを変えた条件，参照マイクを変えた条件を同じ図に並べる．
SIR や SAR は `metrics.json` には残すが，1 つの図に複数指標を詰め込むと，何を比較しているのかが見えにくくなる．

```py
# scripts/plot_bss_metrics.py
from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from bss_handson.plot_style import apply_axis_style


def load_metrics(
    results_dir: str | Path = "results",
    pattern: str = "bss_*/metrics.json",
) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(results_dir).glob(pattern)):
        metrics = json.loads(path.read_text())
        experiment = path.parent.name
        for source_index, value in enumerate(metrics["sdr"]):
            rows.append(
                {
                    "experiment": experiment,
                    "source": str(source_index),
                    "sdr": value,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--pattern", default="bss_*/metrics.json")
    parser.add_argument("--output", default="results/metrics_summary.png")
    parser.add_argument(
        "--style",
        nargs="+",
        default=["styles/common.mplstyle", "styles/paper.mplstyle"],
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_metrics(args.results_dir, args.pattern)
    if df.empty:
        raise RuntimeError(f"{args.results_dir}/{args.pattern} が見つかりません")

    plt.style.use(args.style)
    grid = sns.catplot(
        data=df,
        x="experiment",
        y="sdr",
        hue="source",
        kind="bar",
        errorbar=None,
        height=3.0,
        aspect=1.1,
    )
    plt.style.use(args.style)
    grid.figure.set_size_inches(plt.rcParams["figure.figsize"])
    grid.set_axis_labels("Experiment", "SDR (dB)")
    for ax in grid.axes.flat:
        ax.tick_params(axis="x", rotation=30)
        apply_axis_style(ax)
    grid.figure.subplots_adjust(left=0.12, right=0.98, bottom=0.25, top=0.86, wspace=0.45)
    grid.figure.savefig(output_path)
    plt.close(grid.figure)


if __name__ == "__main__":
    main()
```

`load_metrics()` は，`results/bss_*/metrics.json` を順番に読み込み，`pandas.DataFrame` に変換する．
`--pattern` を指定すると，たとえば `bss_update_*/metrics.json` のように，特定の条件だけを抜き出して図にできる．
この script では `metrics["sdr"]` だけを取り出し，`experiment`，`source`，`sdr` の 3 列に整形する．
この表では，`sdr` が評価値であり，`experiment` は比較する実験条件，`source` は音源 index を表す．
この形にしておくと，`seaborn.catplot()` の `x` に実験条件，`y` に SDR，`hue` に音源 index を指定するだけで，条件ごとの SDR を比較できる．
ここでは実験条件を横軸，SDR を縦軸，音源 index を色で分けた棒グラフを `results/metrics_summary.png` に保存している．
`seaborn` は figure を作るときに内部で Matplotlib の style に関わる設定を変更する場合がある．
そのため，`sns.catplot()` の前に一度 `plt.style.use(args.style)` を呼んで図の作成時の既定値を与え，`sns.catplot()` の後でもう一度同じ style file を読み込む．
さらに，既に作成された axis には，`apply_axis_style()` で tick の向きや grid の設定を明示的に反映している．
このようにしておくと，`styles/common.mplstyle` に書いた内向き tick と破線 grid が，`seaborn` の既定 style によって上書きされたままになることを避けられる．

```bash
uv run python scripts/plot_bss_metrics.py --pattern "bss_model_*/metrics.json" --output results/metrics_model_summary.png
```

集計図も同じ style file で切り替えられる．

```bash
uv run python scripts/plot_bss_metrics.py --pattern "bss_model_*/metrics.json" --style styles/common.mplstyle styles/slide.mplstyle --output results/metrics_model_summary_slide.png
```

更新則ごとの分離性能を比較する場合は，`auxiva.update_method` だけを変えて IP と ISS を実行する．
`output_dir` に更新則の名前を入れておくと，後で `seaborn` の横軸としてそのまま使える．

```bash
uv run bss-handson --config configs/bss.yaml auxiva.update_method=ip output_dir=results/bss_update_ip
uv run bss-handson --config configs/bss.yaml auxiva.update_method=iss output_dir=results/bss_update_iss
uv run python scripts/plot_bss_metrics.py --pattern "bss_update_*/metrics.json" --output results/metrics_update_method.png
```

この図では，横軸が `bss_update_ip` と `bss_update_iss`，縦軸が SDR になる．
同じ混合条件に対して更新則だけを変えるため，IP と ISS の収束や最終性能の違いを見やすい．

AuxIVA の反復回数と分離性能の関係を見る図も作る．
この図は，`n_iter=10`，`n_iter=20` のように実験全体を何度も回して作るのではなく，1 回の AuxIVA 実行中に callback で途中の分離行列を評価して作る．
`scripts/run_bss.py` では `separate()` に `evaluate_iteration()` を渡し，0, 10, 20, 30, 40, 50 回目の分離行列から分離信号を作り，`iteration_metrics.json` に保存している．
これにより，横軸が反復回数，縦軸が分離性能となる 6 点のグラフを，1 回の実験実行から作れる．

```py
# scripts/plot_bss_niter_metrics.py
from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from bss_handson.plot_style import apply_axis_style


def load_iteration_metrics(metrics_path: str | Path = "results/bss_example/iteration_metrics.json") -> pd.DataFrame:
    rows = []
    records = json.loads(Path(metrics_path).read_text())
    for record in records:
        iteration = int(record["iteration"])
        metrics = record["metrics"]
        for source_index, value in enumerate(metrics["sdr"]):
            rows.append(
                {
                    "iteration": iteration,
                    "source": str(source_index),
                    "sdr": value,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-path", default="results/bss_example/iteration_metrics.json")
    parser.add_argument("--output", default="results/metrics_by_niter.png")
    parser.add_argument(
        "--style",
        nargs="+",
        default=["styles/common.mplstyle", "styles/paper.mplstyle"],
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_iteration_metrics(args.metrics_path)
    if df.empty:
        raise RuntimeError(f"{args.metrics_path} に反復回数ごとの評価結果がありません")

    plt.style.use(args.style)
    grid = sns.relplot(
        data=df,
        x="iteration",
        y="sdr",
        hue="source",
        kind="line",
        marker="o",
        errorbar=None,
        height=3.0,
        aspect=1.1,
    )
    plt.style.use(args.style)
    grid.figure.set_size_inches(plt.rcParams["figure.figsize"])
    grid.set_axis_labels("Number of iterations", "SDR (dB)")
    for ax in grid.axes.flat:
        ax.set_xlim(0, 50)
        ax.set_xticks(range(0, 51, 10))
        apply_axis_style(ax)
    grid.figure.subplots_adjust(left=0.12, right=0.98, bottom=0.22, top=0.86, wspace=0.45)
    grid.figure.savefig(output_path)
    plt.close(grid.figure)


if __name__ == "__main__":
    main()
```

`load_iteration_metrics()` は，`results/bss_example/iteration_metrics.json` を読み，反復回数 `iteration` と SDR を 1 つの表にまとめる．
横軸を `iteration`，縦軸を `sdr` として `seaborn.relplot()` に渡すことで，AuxIVA の反復に伴って SDR がどのように変化するかを確認できる．
ここでは 0, 10, 20, 30, 40, 50 回目の 6 点を表示するため，横軸の表示範囲を 0 から 50 に固定し，tick を 10 刻みにしている．
音源ごとの差は `source` を `hue` に指定して色分けする．

```bash
uv run python scripts/plot_bss_niter_metrics.py
```

スライド向けの style file を使う場合も，同じように `--style` で切り替えられる．

```bash
uv run python scripts/plot_bss_niter_metrics.py --style styles/common.mplstyle styles/slide.mplstyle --output results/metrics_by_niter_slide.png
```

単一の音源信号と単一の音源位置だけで性能を見ると，たまたま分離しやすい条件や分離しにくい条件に引っ張られる．
そこで，音源信号と音源位置をランダムに変えながら複数回評価し，平均性能を見るための補助 script を作る．
ここでは CMU ARCTIC の utterance index をランダムに選び，音源位置を部屋の内側からランダムにサンプリングする．
マイク配置，部屋サイズ，RT60，STFT 条件，AuxIVA の設定は `configs/bss.yaml` の値を使う．

```py
# scripts/evaluate_random_bss.py
from pathlib import Path
import argparse
import csv
import json

import numpy as np
from omegaconf import OmegaConf

from bss_handson.config import get_dataset_config, get_room_config
from run_bss import run_separation_pipeline, save_json


def sample_source_positions(
    rng: np.random.Generator,
    room_size: list[float],
    n_sources: int,
    margin: float,
    min_distance: float,
) -> list[list[float]]:
    positions: list[list[float]] = []
    room_width, room_depth = room_size
    while len(positions) < n_sources:
        candidate = np.array(
            [
                rng.uniform(margin, room_width - margin),
                rng.uniform(margin, room_depth - margin),
            ]
        )
        if all(np.linalg.norm(candidate - np.array(position)) >= min_distance for position in positions):
            positions.append(candidate.tolist())
    return positions


def run_trial(config: dict, rng: np.random.Generator, max_utterance_index: int) -> dict:
    dataset_config = get_dataset_config(config)
    room_config = get_room_config(config)
    n_sources = len(dataset_config["speakers"])
    utterance_indices = rng.integers(
        0,
        max_utterance_index + 1,
        size=n_sources,
    ).tolist()
    source_positions = sample_source_positions(
        rng=rng,
        room_size=room_config["size"],
        n_sources=n_sources,
        margin=config["random_eval"]["position_margin"],
        min_distance=config["random_eval"]["min_source_distance"],
    )

    trial_config = {
        **config,
        "dataset": {**dataset_config, "utterance_indices": utterance_indices},
        "room": {**room_config, "source_positions": source_positions},
    }
    result = run_separation_pipeline(trial_config)
    return {
        "utterance_indices": utterance_indices,
        "source_positions": source_positions,
        "metrics": result["metrics"],
    }


def summarize_trials(trials: list[dict]) -> dict:
    summary = {}
    for metric_name in ["sdr", "sir", "sar"]:
        values = np.asarray([trial["metrics"][metric_name] for trial in trials], dtype=np.float64)
        summary[metric_name] = {
            "mean_per_source": values.mean(axis=0).tolist(),
            "std_per_source": values.std(axis=0).tolist(),
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    return summary


def save_trial_csv(trials: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trial",
                "source",
                "utterance_index",
                "source_position",
                "sdr",
                "sir",
                "sar",
                "perm",
            ],
        )
        writer.writeheader()
        for trial_index, trial in enumerate(trials):
            metrics = trial["metrics"]
            for source_index, utterance_index in enumerate(trial["utterance_indices"]):
                writer.writerow(
                    {
                        "trial": trial_index,
                        "source": source_index,
                        "utterance_index": utterance_index,
                        "source_position": json.dumps(trial["source_positions"][source_index]),
                        "sdr": metrics["sdr"][source_index],
                        "sir": metrics["sir"][source_index],
                        "sar": metrics["sar"][source_index],
                        "perm": metrics["perm"][source_index],
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bss.yaml")
    parser.add_argument("--output-dir", default="results/random_eval")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-utterance-index", type=int, default=99)
    args = parser.parse_args()

    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    config["random_eval"] = {
        "position_margin": 0.6,
        "min_source_distance": 1.0,
    }
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = [
        run_trial(
            config=config,
            rng=rng,
            max_utterance_index=args.max_utterance_index,
        )
        for _ in range(args.n_trials)
    ]
    result = {
        "n_trials": args.n_trials,
        "seed": args.seed,
        "config": config,
        "summary": summarize_trials(trials),
        "trials": trials,
    }

    save_json(result, output_dir / "random_average_metrics.json")
    save_trial_csv(trials, output_dir / "random_average_metrics.csv")

    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
```

`sample_source_positions()` は，部屋の壁から `margin` 以上離れた範囲で音源位置を一様乱数により選ぶ．
また，音源同士が近すぎる条件を避けるため，既に選んだ音源位置から `min_source_distance` 以上離れている場合だけ採用する．
`run_trial()` は 1 回分の評価であり，utterance index と音源位置を乱数で決めてから，標準実験と同じ `run_separation_pipeline()` に渡す．
データ取得，音響シミュレーション，STFT，AuxIVA，ISTFT，評価の順序は `scripts/run_bss.py` と共有するため，標準実行と random evaluation で処理内容がずれにくい．
この script では音声ファイルやスペクトログラムを trial ごとに保存しない．
20 パターン平均を見る目的では，すべての中間ファイルを保存すると `results/` が大きくなりすぎるためである．

`random_average_metrics.json` には，全 trial の条件と評価値，および SDR, SIR, SAR の平均と標準偏差を保存する．
`random_average_metrics.csv` には，1 行が「1 trial，1 音源」の評価値になるように保存する．
CSV も保存しておくと，あとから `pandas` や表計算ソフトで分布を確認しやすい．

20 パターンの平均を計算するには，次のコマンドを実行する．

```bash
uv run python scripts/evaluate_random_bss.py --n-trials 20 --seed 0
```

動作確認だけを短く行う場合は，trial 数を減らす．

```bash
uv run python scripts/evaluate_random_bss.py --n-trials 2 --output-dir results/random_eval_smoke
```

## 発展編

### テストコードの作成

研究コードでは，結果の良し悪しを最終的な SDR や図だけで判断しがちである．
しかし，最終結果だけを見ても，STFT の shape 変換，音源位置の生成，評価指標の並べ替え，設定の読み込みのどこで壊れたのかはわかりにくい．
そのため，実験全体を重い end-to-end 実行だけで確認するのではなく，小さな関数ごとにテストを書くとよい．

テストしやすいコードにする第一歩は，`main()` にすべての処理を書かないことである．[^strong-main]
たとえば，次のような処理が 1 つの script に直書きされていると，音声ファイル，設定ファイル，乱数，出力先，作図が一体になり，部分的な確認が難しい．

```py
def main():
    config = load_config()
    mixture = simulate_room(config)
    estimate = separate(mixture, config)
    metrics = evaluate(estimate)
    save_results(metrics)
```

これを，入力と出力が明確な小さな関数に分けると，テストで確認できる単位が増える．

```py
def run_bss(config, sources, source_positions):
    mixture = simulate_room(config, sources, source_positions)
    estimate = separate(mixture, config.method)
    return evaluate(estimate, sources)
```

このようにしておくと，短いダミー信号を渡して「shape が期待通りか」「乱数 seed を固定すると同じ条件になるか」「評価結果の key がそろっているか」を確認できる．
音響シミュレーションや AuxIVA の完全な性能を毎回テストする必要はない．
最初は，配列 shape，設定値の反映，保存する JSON/CSV の列名，例外を出すべき不正設定など，壊れると後で追いにくい部分からテストするだけでも効果がある．

```py
def test_run_bss_returns_metrics():
    metrics = run_bss(config, sources, source_positions)
    assert set(metrics) >= {"sdr", "sir", "sar"}
```

この程度のテストでも，関数の入口と出口を意識するきっかけになる．
テストを書くためには，巨大な `main()` から処理を切り出す必要があり，結果として実験コードの責務も自然に分かれる．
機械学習や信号処理の研究コードでは，完全な正解値を毎回用意するのが難しい場合も多い．
その場合でも，「落ちないこと」だけでなく，「shape が変わっていないこと」「seed 固定時に同じ条件が再現されること」「保存形式が後段の集計 script と合っていること」をテストにしておくと，実験を増やすときの事故を減らせる．

[^strong-main]: 冒頭で bad example として示されるような，何でも `main.py` に詰め込んだコードは「オレオレ最強 main.py」と呼ばれ揶揄されている（[テストを書かない研究者に送る、最初にテストを書く実験コード入門](https://speakerdeck.com/shunk031/lets-start-your-ml-project-by-writing-tests)）．

### ベクトル演算による高速化

本資料の AuxIVA 実装では，数式との対応を見やすくするために意図的に `for` 文を多く使っている．
しかし，実行速度という観点では，Python の `for` 文を大量に回す実装は遅い．
NumPy は C や Fortran で実装された配列演算をまとめて呼び出すことで高速になるため，本格的な実験では frame や周波数 bin の loop をできるだけ NumPy のベクトル演算，broadcasting，batch matrix multiplication に寄せる方がよい．
特に STFT frame 数や周波数 bin 数が大きい音響信号処理では，この差は非常に大きくなる．
ただし，broadcasting は常に速くなるわけではなく，一時配列のコピー，メモリアクセスパターン，cache に載るデータ量にも注意が必要である（[Performance Tips of NumPy ndarray](https://shihchinw.github.io/2019/03/performance-tips-of-numpy-ndarray.html)）．

たとえば，`demix()` は次のように周波数方向と frame 方向をまとめて計算できる．
`w[None, :, :, :]` は `w` に frame 軸を追加し，`x[:, :, :, None]` は channel ベクトルを列ベクトルとして扱うための軸を追加している．
`@` 演算子は，両辺が 3 次元以上の場合，先頭側の軸を batch 次元として扱う．
そのため，shape `(n_frames, n_freq, n_channels, n_channels)` と `(n_frames, n_freq, n_channels, 1)` の行列積をまとめて計算できる．

```py
def demix_fast(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    y = w[None, :, :, :] @ x[:, :, :, None]
    return y[..., 0]
```

重み付き共分散行列も，少なくとも frame 方向と周波数方向の loop はまとめられる．
次の例では，`x_fct` を shape `(1, n_freq, n_channels, n_frames)` にし，`weighted_x` を shape `(n_channels, n_freq, n_frames, n_channels)` にする．
これにより，`x_fct @ weighted_x.conj()` は，音源 index，周波数 bin ごとに $\sum _ {t} \varphi(r _ {k,t})\mathbfit{x} _ {f,t}\mathbfit{x} _ {f,t}^{\mathsf{H}}$ をまとめて計算する．

```py
def weighted_covariance_fast(
    x: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    n_frames = x.shape[0]
    x_fct = x.transpose(1, 2, 0)[None, :, :, :]
    weighted_x = weights.T[:, None, :, None] * x.transpose(1, 0, 2)[None, :, :, :]
    return (x_fct @ weighted_x.conj()) / n_frames
```

`ip_update()` も，音源 index `k` の更新順序は残しつつ，周波数 bin 方向の `np.linalg.solve()` をまとめて実行できる．
`np.linalg.solve()` は stack された行列にも対応するため，`a` の shape が `(n_freq, n_channels, n_channels)`，`e` の shape が `(n_freq, n_channels, 1)` であれば，各周波数 bin の線形方程式をまとめて解ける．

```py
def ip_update_fast(
    x: np.ndarray,
    w: np.ndarray,
    model: str = "laplace",
    eps: float = 1.0e-10,
) -> np.ndarray:
    _, n_freq, n_channels = x.shape
    y = demix_fast(x, w)
    weights = source_weights(y, model=model, eps=eps)
    v = weighted_covariance_fast(x, weights)

    w_new = w.copy()
    for k in range(n_channels):
        a = w_new @ v[k]
        e = np.zeros((n_freq, n_channels, 1), dtype=np.complex128)
        e[:, k, 0] = 1.0
        wk = np.linalg.solve(a, e)[:, :, 0]
        v_wk = v[k] @ wk[:, :, None]
        denom_sq = np.real((wk.conj()[:, None, :] @ v_wk)[:, 0, 0])
        denom = np.sqrt(np.maximum(denom_sq, eps))
        w_new[:, k, :] = (wk / denom[:, None]).conj()

    return w_new
```

このような実装は高速だが，最初に読むには少し難しい．
`None` による軸追加，`transpose()` による軸順序の変更，`@` 演算子の batch 次元の扱いを理解していないと，どの軸に対する計算なのか追いにくい．
そのため，本資料の本体では，まず `for` 文で数式に近い形を示した．
実験規模が大きくなり，実行時間が問題になった段階で，このようなベクトル演算に置き換えるのが現実的である．

### クラスを用いた設計

この教材の `bss-handson` は，関数を中心にした小さい実装として書いている．
関数中心の設計は，処理の流れが見えやすく，初学者が数式と実装を対応づけやすい．
一方で，手法が増え，手法ごとの状態や設定が増えてくると，すべてを関数引数として渡し続けるより，class にまとめた方が見通しがよくなる場合がある．

たとえば，AuxIVA では反復回数，更新手法，音源モデル，微小値，参照マイクなどの設定が一組で使われる．
これらを毎回 `separate(x, n_iter=..., update_method=..., eps=...)` のように渡すと，実験 script 側が手法固有の細部を知りすぎる．
代わりに，手法ごとの設定を保持する `AuxIVASeparator` のような class を考えることができる．

```py
class AuxIVASeparator:
    def __init__(self, n_iter, update_method, model, eps, reference_mic):
        self.n_iter = n_iter
        self.update_method = update_method
        self.model = model
        self.eps = eps
        self.reference_mic = reference_mic

    def separate(self, x):
        y, w = demix(
            x,
            n_iter=self.n_iter,
            update_method=self.update_method,
            model=self.model,
            eps=self.eps,
        )
        return project_back(y, x, self.reference_mic), w
```

この形にすると，実験 script は AuxIVA の細かい引数ではなく，「分離器に `x` を渡す」という入口だけを見ればよくなる．
また，`OnlineAuxIVASeparator` や `ILRMASeparator` を追加するときも，外側からは同じように `separator.separate(x)` を呼べる設計に寄せられる．
手法固有の初期化や内部状態を class の中へ置けるため，online 処理のように frame ごとの統計量を持つ手法も扱いやすくなる．

さらに，AuxIVA と ILRMA のように一見別の手法に見えるものでも，すべての処理が別物になるわけではない．
determined BSS として見ると，観測スペクトログラム `x` に分離行列 `w` をかけて `y` を得る `separate()` の処理や，重み付き共分散行列から IP により分離行列を更新する処理は共通化しやすい．
本質的に変わるのは，分離信号から音源モデルの重みをどう更新するかである．
AuxIVA では，Laplace model や time-varying Gauss model に基づく重みを計算する．
ILRMA では，音源スペクトログラムの分散を NMF の基底とアクティベーションで表し，その NMF パラメータを更新して重みを作る．
つまり，分離行列更新を「空間モデルの更新」，音源モデル重みの更新を「音源モデルの更新」と分けておくと，手法間で再利用できる部分が見えやすくなる．

```py
class DeterminedBSS:
    def separate(self, x, w):
        return apply_demixing_filter(x, w)

    def update_spatial_model(self, x, y, w):
        weights = self.update_source_model(y)
        covariance = weighted_covariance(x, weights)
        return ip_update(w, covariance)

    def update_source_model(self, y):
        raise NotImplementedError
```

この設計では，`DeterminedBSS` が `separate()` と IP 更新の流れを持ち，`update_source_model()` だけを子 class で実装する．
`AuxIVASeparator` は AuxIVA の重み計算を実装し，`ILRMASeparator` は NMF パラメータ更新とそこから得られる重み計算を実装する．
すると，IP の式を修正したときに AuxIVA 用と ILRMA 用の両方を直す必要がなくなる．
また，新しい音源モデルを試すときも，分離行列更新や `separate()` を再利用しながら，差分だけを小さく実装できる．

このような設計の実例として，`ssspy` も参考になる（[tky823/ssspy](https://github.com/tky823/ssspy)）．
`ssspy` は sound source separation のための Python toolkit であり ICA，FDICA，IVA，ILRMA，IPSDTA，MNMF，PDS-BSS，ADMM-BSS，HVA，cACGMM など，多数の BSS 手法と notebook が整理されている．
実装も，`ssspy/bss/iva.py`，`ssspy/bss/ilrma.py` のような手法別 module と，`ssspy/bss/_update_spatial_model.py` のような IP/ISS/IPA 系の空間モデル更新 module に分かれている．
教育用の `bss-handson` にそのまま持ち込むには大きすぎるが，多数の手法を扱う場合に「共通の反復処理」「空間モデル更新」「音源モデル更新」を分ける設計例として読む価値がある．

実験全体も class にすることができる．
たとえば，データ生成，分離，評価，保存をまとめる `BSSExperiment` を考える．

```py
class BSSExperiment:
    def __init__(self, config, separator, evaluator):
        self.config = config
        self.separator = separator
        self.evaluator = evaluator

    def run(self, sources, source_positions):
        mixture = simulate_room(self.config, sources, source_positions)
        estimate, _ = self.separator.separate(mixture)
        return self.evaluator.evaluate(estimate, sources)
```

この例では，`BSSExperiment` は「実験を 1 回実行する流れ」を持ち，`separator` は「混合信号を分離する方法」を持つ．
責務が分かれているため，AuxIVA から ILRMA に変えるときは `separator` だけを差し替えればよい．
評価方法を変えるときも，`evaluator` を差し替えればよい．
実験の流れを読む人は，`run()` を見れば処理順を追える．
手法の詳細を読む人は，`AuxIVASeparator` や `ILRMASeparator` を見ればよい．

ただし，class を使えば必ず良くなるわけではない．
状態を持たない単純な処理まで class にすると，かえって読む場所が増える．
このコードベースで class を導入するなら，まずは「手法ごとの設定と状態を持つ分離器」「実験全体の流れをまとめる実行器」のように，状態や差し替え対象が明確な箇所に限るのがよい．
関数で十分に表せる STFT の shape 変換や評価値の小さな整形まで，無理に class にする必要はない．

### 依存性注入

依存性注入は，ある処理が内部で使う部品を，関数や class の外側から渡せるようにする考え方である．
研究コードでは，乱数生成器，音響シミュレータ，分離手法，評価関数，保存先などが依存性になりやすい．
これらを関数の中で直接作ると，実験条件の差し替えやテストが難しくなる．
機械学習 project におけるコードと設定の分離や依存性注入の考え方は，実験管理の講義資料にも整理されている（[機械学習プロジェクトにおける実験管理](https://www.docswell.com/s/2625216247/ZQXY9J-2026-02-02-185832#p1)）．

たとえば，`run_bss()` の内部で常に AuxIVA を直接呼ぶと，ILRMA やテスト用の軽い分離器へ差し替えるには `run_bss()` 自体を書き換える必要がある．

```py
def run_bss(config, mixture):
    estimate = auxiva.separate(mixture, **config.auxiva)
    return evaluate(estimate)
```

分離処理を外から渡す形にすると，実験全体の流れと手法の実装を分けられる．

```py
def run_bss(config, mixture, separator):
    estimate = separator(mixture, config.method)
    return evaluate(estimate)
```

この形にしておくと，本実験では `auxiva.separate` を渡し，動作確認では入力をそのまま返す簡単な関数を渡せる．
重い音響シミュレーションや反復最適化を避けたテストも書きやすくなる．

```py
def identity_separator(mixture, method_config):
    return mixture
```

依存性注入は，何でも抽象化するための技法ではない．
差し替えたい理由がある部品だけを外から渡せるようにするのがよい．
この資料の範囲では，まず `separator`，`evaluator`，`rng`，`output_dir` のように，実験条件やテストで変わりやすいものを関数引数に出すだけで十分である．
こうしておくと，設定ファイルで `method.name` を変えたときに呼び出す手法を差し替えやすくなり，テストでは軽い代替関数を渡して保存形式や集計処理だけを確認できる．

### 音源分離手法の追加

このハンズオンでは AuxIVA の IP/ISS 更新だけを扱ったが，実際の研究では online AuxIVA，ILRMA，FastMNMF，DNN を使う手法など，複数の音源分離手法を同じ実験基盤で比較したくなる．
このとき重要なのは，新しい手法を追加するたびに `scripts/run_bss.py` や既存の `auxiva.py` を巨大化させないことである．
手法が増えたときほど，`src/` には単機能の処理を置き，`scripts/` には実験の組み合わせを書く，という境界を守る必要がある．

まず，音源分離に関する実装を `bss/` という subpackage に分けることを考える．
現在の教材ではファイル数を抑えるために `auxiva.py` を `src/bss_handson/` 直下に置いているが，online AuxIVA や ILRMA を追加する段階では，BSS 関連の module をまとめた方が見通しがよい．
たとえば次のような構成にする．

```txt
src/
└── bss_handson/
    ├── bss/
    │   ├── __init__.py
    │   ├── auxiva.py
    │   ├── online_auxiva.py
    │   ├── ilrma.py
    │   ├── source_models.py
    │   └── update_methods.py
    ├── stft.py
    ├── evaluation.py
    └── ...
```

ここでは，BSS 手法本体，分離行列更新，音源モデルに依存する処理を別 module に置く．
AuxIVA と ILRMA の共通化の考え方は「クラスを用いた設計」で述べたので，この節ではファイル配置，設定，結果整理の観点に絞る．
分離行列更新は，`bss/update_methods.py` のような module に切り出す方がよい．

たとえば，`bss/update_methods.py` には，重み付き共分散行列と現在の分離行列を受け取って更新後の分離行列を返す関数を置く．

```py
def ip_update(w, covariance, eps):
    ...


def iss_update(y, w, weights, eps):
    ...
```

`bss/source_models.py` には，Laplace model，time-varying Gauss model，ILRMA の NMF 音源モデルなど，音源モデルに依存する処理を置く．

BSS 手法の module は，できるだけ同じ入口を持つようにする．
たとえば，バッチ処理の手法であれば，観測スペクトログラム `x` を受け取り，分離スペクトログラム `y`，分離行列 `w`，必要なら手法固有の状態を返す関数にそろえる．

```py
result = separate(x, **method_config)
```

`result` は，最低限 `estimate` と `demixing_matrix` を持つようにする．
ILRMA なら NMF の基底やアクティベーション，online AuxIVA なら最終状態や frame ごとの統計量も含める．
戻り値を tuple だけにすると，手法ごとに返すものが増えたときに意味が崩れやすい．
手法が増える段階では，`dict` や dataclass で名前付きの戻り値にすることも検討する．

online AuxIVA のように frame を逐次処理する手法では，内部状態を持つ必要がある．
その場合でも，外側から見た責務を明確にする．
たとえば，低レベルには `initialize()`, `update(frame)`, `separate_frame(frame)` のような状態更新関数を置き，実験から使う入口として `separate_online(x, **method_config)` を用意する．
ここでいう wrapper は，低レベルの逐次更新関数を実験で使いやすい形にまとめるための関数である．
低レベルの更新式と，実験全体を回す処理を同じ関数に混ぜないことが重要である．

手法を切り替えるための dispatch は，`src/` の数値計算 module に埋め込まない．
たとえば `bss/auxiva.py` の中で `if method == "ilrma"` と書くのではなく，`scripts/run_bss.py` 側で設定ファイルの `method.name` を見て，どの module の `separate()` を呼ぶかを決める．
これは，`auxiva.py` が ILRMA の存在を知らなくてよい状態を保つためである．

設定ファイルも，手法が増えることを見越して分けるとよい．
現在は `auxiva:` という key を使っているが，複数手法を扱う段階では，次のように `method:` の下に BSS 手法名と，その内部で使う更新手法名を分けて書く構成が扱いやすい．

```yaml
method:
  name: auxiva
  n_iter: 50
  model: laplace
  update_method: ip
  eps: 1.0e-10
  reference_mic: 0
```

この設定では，`method.name` は BSS 手法そのものを表し，`method.update_method` は分離行列更新手法を表す．
つまり，`auxiva` と `ilrma` は同じ階層の比較対象であり，`ip` と `iss` はその内部で使う更新手法の比較対象である．
この 2 つを混ぜて `method=ip` のように書くと，何を比較しているのか曖昧になる．

ILRMA では，たとえば `n_basis` や NMF 更新回数が必要になる．
online AuxIVA では，忘却係数，block size，初期化に使う frame 数などが必要になる．
これらをすべて `auxiva:` の下に押し込むと，設定ファイルの意味が崩れる．
手法ごとに必要なパラメータが違う場合は，`configs/bss_auxiva.yaml`，`configs/bss_online_auxiva.yaml`，`configs/bss_ilrma.yaml` のように設定ファイルを分ける方が読みやすい．

結果を可視化するときも，この階層を保つ必要がある．
IP と ISS だけを比較する図であれば，横軸は `update_method` でよい．
しかし AuxIVA と ILRMA を比較する図では，横軸は `method` または `bss_method` にするべきである．
さらに，AuxIVA-IP，AuxIVA-ISS，ILRMA-IP のように BSS 手法と更新手法の組合せを比較するなら，tidy data は次のような列を持つのが自然である．

| bss_method | update_method | source |  sdr |
| :--------- | :------------ | -----: | ---: |
| auxiva     | ip            |      0 | 6.83 |
| auxiva     | ip            |      1 | 8.74 |
| auxiva     | iss           |      0 | 6.54 |
| auxiva     | iss           |      1 | 8.58 |
| ilrma      | ip            |      0 | 7.20 |
| ilrma      | ip            |      1 | 9.10 |

この形にしておくと，`seaborn.catplot(data=df, x="bss_method", y="sdr", hue="update_method")` のように BSS 手法と更新手法を分けて可視化できる．
音源ごとの差も見るなら，`col="source"` を使うか，音源ごとに別の図を作る．
重要なのは，`ip` と `auxiva` を同じ列に入れないことである．
一方は更新手法であり，他方は BSS 手法なので，tidy data でも別の変数として扱う．

このような拡張の例として，`taishi-n/oobss` も参考になる（[taishi-n/oobss](https://github.com/taishi-n/oobss)）．
`oobss` は筆者の作成した Open Online Blind Source Separation toolkit であり，classical and online BSS algorithms に加えて，設定，logging，documentation のための utilities を含んでいる．
batch 処理では `fit_transform_tf(..., request=BatchRequest(...))`，stream 処理では `process_stream_tf(..., request=StreamRequest(...))` のように，処理形態ごとの request object を使う unified separator contract が示されている．

本資料の `bss-handson` は教育用の最小実装なので，最初から `oobss` のような class 設計や benchmark engine を入れる必要はない．
一方で，研究が進み，AuxIVA，online AuxIVA，ILRMA を同じ実験基盤で比較する段階では，`oobss` のように「手法本体」「実験実行」「benchmark」「reporting」を分ける設計が有効になる．
特に，`oobss.benchmark` には method sweep，結果集計，report 生成のための entry point が用意されており，この資料で扱った `scripts/` による sweep と可視化を大きく発展させた設計例として読める．
拡張時の配置基準をまとめると，次のようになる．

| 追加したいもの               | 置き場所                                | 理由                                            |
| :--------------------------- | :-------------------------------------- | :---------------------------------------------- |
| 新しい BSS 手法              | `src/bss_handson/bss/<method>.py`       | 手法単位で仮定と状態を分けるため                |
| IP/ISS などの更新手法        | `src/bss_handson/bss/update_methods.py` | AuxIVA や ILRMA から共通利用できるため          |
| 音源モデル                   | `src/bss_handson/bss/source_models.py`  | 手法と統計モデルの依存を分けるため              |
| 手法固有の目的関数           | `src/bss_handson/bss/<method>.py`       | 手法の検証や debug に使うため                   |
| 手法を切り替えて実験する処理 | `scripts/run_bss.py` または別 script    | 実験条件に依存する組み合わせであるため          |
| 複数手法の一括実行           | `scripts/`                              | parameter sweep や結果保存に依存するため        |
| 複数手法の結果集計           | `scripts/`                              | `results/` の構造や glob pattern に依存するため |
| 論文用の図作成               | `scripts/`                              | どの条件を比較するかが実験ごとに変わるため      |

この方針にすると，online AuxIVA を追加しても，ILRMA を追加しても，STFT，room simulation，評価，可視化の基本部品はそのまま使える．
また，手法固有の実装と実験管理の処理が分かれるため，分離性能が変わったときに，アルゴリズムの変更によるものなのか，実験条件や保存処理の変更によるものなのかを切り分けやすくなる．

### MLflow による実験トラッキング

この資料では，実験結果を `results/` 以下のディレクトリ，`config.yaml`，`metrics.json`，CSV，図として保存する方針を扱った．
この方法は小さな研究 project では十分に有効であり，ファイルを直接読めるため仕組みも追いやすい．
一方で，手法数，パラメータ数，乱数 seed，データ条件が増えると，どの run がどの条件で，どの評価値を出したのかをファイル名と手元の集計 script だけで追うのが難しくなる．

この段階では，実験管理ツールとして MLflow を使うことも検討できる（[MLflow: Tracking Hyperparameter Tuning](https://mlflow.org/docs/latest/ml/getting-started/hyperparameter-tuning/)）．
MLflow Tracking では，各 run に対して parameter，metric，artifact を記録できる．
BSS の実験に対応させるなら，`method.name`，`method.update_method`，`stft.win_length`，`room.rt60`，`seed` などを parameter として記録し，SDR，SIR，SAR，実行時間などを metric として記録する．
分離音，スペクトログラム，設定ファイル，評価 CSV は artifact として残せる．

```py
with mlflow.start_run(run_name=output_dir.name):
    mlflow.log_params(flatten_config(config))
    mlflow.log_metrics({"sdr_mean": sdr_mean, "sir_mean": sir_mean})
    mlflow.log_artifacts(output_dir)
```

このようにしておくと，複数の実験 run を UI 上で検索，比較しやすくなる．
たとえば AuxIVA-IP，AuxIVA-ISS，ILRMA-IP のような条件を横断して，`method.name` や `method.update_method` で filter しながら評価値を確認できる．
共同研究で同じ tracking server を使えば，誰がどの条件を実行したのかも共有しやすい．
ただし，MLflow を入れると依存関係と運用対象が増えるため，最初から必須にする必要はない．
まずはこの資料で扱ったようにファイルとして再現可能に保存し，run の数が増えて比較や共有がつらくなった段階で導入するのが現実的である．

### Coding agentの活用

発展編で扱ったような拡張，たとえばテスト追加，NumPy ベクトル化，class 設計への移行，依存性注入，ILRMA の追加，MLflow tracking は，Claude Code や Codex をはじめとする coding agent に依頼できる．
実際に筆者も最近のコードベースは coding agent に多くの部分を頼っている．
ただし，coding agent に丸投げすれば研究コードがよくなるわけではない．
生成 AI の出力に責任を持つには，人間側が出力の良し悪しを判断できる必要がある（[生成AIとの付き合い方](https://speakerdeck.com/kaityo256/generative-ai-and-us)）．
発展的な拡張を依頼する場面では，coding agent は実装作業を速くする相手であり，研究上の判断を代わりに引き受ける相手ではない．

依頼する側が最低限わかっているべきことは，まず「何を変えたいのか」である．
たとえば「ILRMA を追加して」では曖昧である．
AuxIVA と ILRMA で共通化できる部分は `separate()` や IP による空間モデル更新であり，差分は NMF に基づく音源モデル更新である，という程度の切り分けを人間が説明できる必要がある．
この理解がないと，coding agent が見た目だけ似た別実装を作っても，何が再利用され，何が新しく追加されたのか判断できない．

次に，コードベースの境界を説明できる必要がある．
`src/` に置く再利用可能な処理と，`scripts/` に置く実験実行処理の違い，`configs/` に置く標準設定と command line override の関係，`data/` と `results/` を git 管理しない理由を理解しておく．
class 設計や依存性注入を依頼する場合も，どの依存を差し替えたいのかを説明できなければ，抽象化だけが増えて読みにくくなる．
たとえば，差し替えたいのは `separator`，`evaluator`，`rng`，`output_dir` なのか，それとも単なる局所変数なのかを区別する．

また，検収条件を自分で決められる必要がある．
coding agent が「実装しました」と言っても，それが研究上正しいとは限らない．
少なくとも，`uv run python -m py_compile src/bss_handson/*.py`，smoke test，既存条件での metrics の形式確認，追加した関数の単体テスト，保存される artifact の確認を指定する．
性能改善を依頼する場合は，速くなったかどうかだけでなく，ベクトル化前後で shape と数値が一致するかも確認する．
MLflow 対応を依頼する場合は，parameter，metric，artifact のどれとして記録すべきかを人間が決める．

そのために必要な勉強は，coding agent の使い方そのものよりも，まず対象分野とコードの読み方である．
本資料の範囲では，次の項目を学んでおくと依頼の質が上がる．

- BSS の基礎: STFT，determined BSS，AuxIVA，IP/ISS，projection back，ILRMA の音源モデル
- 配列計算: `(n_frames, n_freq, n_channels)` の shape，broadcasting，batch matrix multiplication，数値誤差
- Python project: `src layout`，`pyproject.toml`，entry point，`uv run`，依存関係管理
- 実験管理: 設定ファイル，dot-list override，出力ディレクトリ，JSON/CSV，MLflow の parameter/metric/artifact
- テスト: 小さい入力での単体テスト，smoke test，乱数 seed，保存形式の確認
- git: 差分を読む，commit 単位を分ける，生成物を混ぜない，変更を戻す前に内容を確認する
- 既存実装の読解: `ssspy` や `oobss` のような repository で，手法本体，共通更新，実験実行がどう分かれているかを見る

依頼文では，目的，変更範囲，守るべき設計，検収条件を明示する．
たとえば，ILRMA 追加を依頼するなら次のように書ける．

```txt
発展編の設計方針に沿って，AuxIVA と ILRMA を同じ実験基盤で比較できるようにしたい．
既存の STFT，評価，保存処理は変更しない．
`src/bss_handson/bss/` を作り，IP 更新と `separate()` は共通部品として切り出す．
AuxIVA と ILRMA の差分は音源モデル更新として分ける．
設定は `method.name` と `method.update_method` を使う．
動作確認として既存の AuxIVA smoke test が通ること，新しい ILRMA の最小 smoke test が通ること，
metrics.json の形式が既存の集計 script と互換であることを確認する．
```

このように書くと，coding agent は設計判断を勝手に埋める必要が少なくなる．
一方で，すべての行の実装方法まで指定する必要はない．
重要なのは，研究上の目的，既存設計との接続，変更してよい範囲，確認方法を人間が握ることである．
coding agent の出力に付加価値を付けられるかどうかは，プロンプトの技巧だけでなく，自分がどこまでコードと実験を理解しているかに依存する．

## まとめ

ブラインド音源分離の研究では，分離手法そのものだけでなく，実験を再実行し，比較し，論文や発表に使える形で残すための基盤が必要である．
この資料では，その基盤を `uv` による project 管理，役割の分かれたディレクトリ構成，設定ファイルを中心とした実験管理，結果の可視化という観点から扱った．

`uv` は，Python version，依存関係，entry point，lock file を project 単位で扱うための入口である．
実験コードを単なる script の集合として置くのではなく，CLI として実行できる project にしておくと，実行環境と実行方法が明確になる．
どの directory で `uv run ...` を実行するのか，どの設定ファイルを読み込むのかを揃えておくことで，環境差や実行場所の違いによる事故を減らせる．

ディレクトリ構成では，実装，設定，補助 script，データ，結果を混ぜないことが重要である．
`src/` には再利用する処理，`configs/` には実験条件，`scripts/` には集計や作図などの補助処理，`data/` と `results/` には再生成可能な入力と出力を置く．
git では，結果そのものではなく，結果を再生成するためのコード，設定，README，lock file を管理する．

実験管理では，標準設定を 1 つ決め，条件差分を command line override や別設定として明示する．
実行後には，最終的な設定，評価指標，途中経過を出力ディレクトリに保存する．
この形にしておくと，あとから結果だけを見ても，どの条件で実行したのか，何を比較すべきなのかを追える．
単一条件の結果だけで判断せず，反復回数，音源信号，音源位置などを変えた結果を集計できる形式で残すことも重要である．

可視化では，確認用の図と論文・発表用の図を分けて考える．
評価値は JSON や CSV から読み込み，条件比較や反復過程を再現可能な作図 script で描く．
Matplotlib の style file を使って，共通設定，論文向け設定，スライド向け設定を分けておくと，同じ結果から用途に応じた図を作り直せる．
研究が進むほど実験条件と結果は増えるため，早い段階でこのような実験管理と可視化の型を作っておくことが重要である．

[^wrapper]: wrapper は，既存の関数，class，外部ライブラリなどを，目的の実験コードから使いやすい形に wrap するために作る関数や class を指す．たとえば，データのダウンロード，path の指定，shape の整形，既定値の設定などを wrapper 側に置くと，呼び出し側のコードを簡潔にできる．

## 参考資料

- [Wilson et al. (2014) Best Practices for Scientific Computing](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1001745)
  - 科学計算におけるコードの読みやすさ，自動化，version control，重複排除，documentation，collaboration を扱う論文である．本資料の git 管理や CLI 化の背景として参照できる．
- [Wilson et al. (2017) Good enough practices in scientific computing](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510)
  - 研究計算のために採用しやすい実践として，data management，software，collaboration，project organization，change tracking を整理した論文である．本資料の README，`data/`，`results/`，`src/`，依存関係管理の背景として参照できる．
- [Python Packaging User Guide: src レイアウト対フラットレイアウト](https://packaging.python.org/ja/latest/discussions/src-layout-vs-flat-layout/)
  - `src layout` と `flat layout` の違い，src layout では package を install して使う必要があること，カレントディレクトリ由来の意図しない import を避けやすいことが説明されている．
- [simonwhitaker/gibo](https://github.com/simonwhitaker/gibo)
  - GitHub が公開している `.gitignore` template を command line から取得するための tool である．本資料では，Python project 用の `.gitignore` を用意する補助として参照できる．
- [研究のための Python 開発環境](https://zenn.dev/zenizeni/books/a64578f98450c2)
  - 研究用 Python 環境構築，ディレクトリ構成，実験設定管理などを幅広く扱う資料である．本資料の `uv` やディレクトリ構成の説明と合わせて読むとよい．
- [uvを使ったPython環境構築 - 人工知能応用特論Ⅰ 第3回](https://www.docswell.com/s/2625216247/Z2Q3YV-2025-10-22-170737#p17)
  - `uv init`，`uv add`，`uv sync` など，`uv` を使った Python 環境構築の基本操作を講義資料としてまとめたもの．本資料のハンズオンで使う `uv` コマンドの補足資料として参照できる．
- [uv: Creating projects](https://docs.astral.sh/uv/concepts/projects/init/)
  - `uv init` による project 作成と，application と library の違いを説明している公式資料である．本資料では，`uv init --lib` を使う理由の背景として参照できる．
- [OmegaConf: Usage](https://omegaconf.readthedocs.io/en/latest/usage.html)
  - YAML ファイルの読み込み，dot-list からの設定作成，command line 引数からの設定作成，複数設定の merge などを説明している．本資料では，`configs/bss.yaml` と `auxiva.n_iter=10` のような command line override を統合するために参照できる．
- [pyroomacoustics: Dataset Wrappers](https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.datasets.html)
  - CMU ARCTIC を含む dataset wrapper の仕様と，`CMUArcticCorpus(download=True, speaker=[...])` による自動ダウンロード例を説明している．本資料の CMU ARCTIC 取得処理の背景として参照できる．
- [CMU_ARCTIC Databases](https://www.festvox.org/cmu_arctic/)
  - CMU ARCTIC の公式ページであり，話者ごとの音声データベースの概要と配布ファイルへのリンクを確認できる．本資料では，音響シミュレーションに使うクリーン音声データの出典として参照する．
- [Kominek and Black (2004) The CMU Arctic speech databases](https://www.isca-archive.org/ssw_2004/kominek04b_ssw.html)
  - CMU ARCTIC の設計意図，収録条件，発話数，音声合成研究での利用を説明した論文である．本資料で CMU ARCTIC を実験用音声として使う背景として参照できる．
- [Ono (2011) Stable and fast update rules for independent vector analysis based on auxiliary function technique](https://doi.org/10.1109/ASPAA.2011.6082320)
  - AuxIVA の代表的な原典であり，補助関数法に基づく IVA の安定で高速な更新則を提案している．本資料の重み付き共分散行列と IP 更新の背景として参照できる．
- [Kim et al. (2007) Blind source separation exploiting higher-order frequency dependencies](https://doi.org/10.1109/TASL.2006.872618)
  - IVA の背景となる周波数間依存を利用したブラインド音源分離を扱う論文である．周波数領域 BSS における permutation 問題と，IVA の考え方を理解するための参考文献である．
- [pyroomacoustics: Independent Vector Analysis (AuxIVA)](https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.bss.auxiva.html)
  - `pyroomacoustics.bss.auxiva` の API と，`model="laplace"`，`model="gauss"` による音源モデルの切り替えを説明している．本資料の AuxIVA 実装で重み計算を分ける背景として参照できる．
- [Scheibler and Ono (2020) Fast and stable blind source separation with rank-1 updates](https://github.com/onolab-tmu/code_2020ICASSP_iss)
  - AuxIVA の目的関数を rank-1 update で最適化する ISS を提案した研究である．本資料では，IP と ISS を `auxiva.update_method` で切り替えて比較する背景として参照できる．
- [Wickham (2014) Tidy Data](https://www.jstatsoft.org/v59/i10/)
  - 1 つの変数を 1 つの列，1 つの観測を 1 つの行として整理する tidy data の考え方を説明した論文である．本資料では，IP と ISS のような更新手法の違いを `update_method`，`source`，`sdr` の縦長の表へ変換し，`seaborn` で可視化する背景として参照できる．
- [私たちのR: 整然データ構造](https://www.jaysong.net/RBook/tidydata.html)
  - tidy data の 4 条件，すなわち 1 つの列は 1 つの変数，1 つの行は 1 つの観測，1 つのセルは 1 つの値，1 つの表は 1 つの観測単位を持つという整理を日本語で説明している．本資料の wide table と tidy table の比較，および `seaborn` に渡す縦長データの理解に対応する参考資料である．
- [科学技術論文に用いる図表のための matplotlib 設定](https://qiita.com/n-taishi/items/042039847601f264236d)
  - 論文や発表スライドに載せる媒体の大きさに合わせて，Matplotlib 側の図サイズ，フォントサイズ，style file を設計する考え方を説明している．本資料では，`styles/common.mplstyle`，`styles/paper.mplstyle`，`styles/slide.mplstyle` を分ける背景として参照できる．
- [Rougier et al. (2014) Ten Simple Rules for Better Figures](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003833)
  - 科学図の設計について，読者とメッセージの明確化，媒体に合わせた図の調整，デフォルト設定への依存を避けること，色や不要な装飾の扱いなどを整理した論文である．本資料では，論文向けとプレゼン向けで Matplotlib style file を分ける背景として参照できる．
- [Matplotlib: Customizing Matplotlib with style sheets and rcParams](https://matplotlib.org/stable/users/explain/customizing.html)
  - Matplotlib の style sheet と rcParams の仕組み，`.mplstyle` の配置方法，`plt.style.use()` による読み込み方が説明されている．本資料で `styles/common.mplstyle` などを作る際の公式資料である．
- [Matplotlib: Style sheets reference](https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html)
  - Matplotlib に組み込まれている style sheet の一覧と見た目を確認できる．自分で style file を作る前に，既存 style の考え方を確認する参考になる．
- [SciencePlots](https://github.com/garrettj403/SciencePlots)
  - 科学論文，発表，学位論文向けの Matplotlib style を提供する package である．外部 style を使う場合の選択肢として参照できる．
- [Hydra: Multi-run](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/)
  - command line から複数の設定値を指定し，parameter sweep を実行する multi-run の仕組みを説明している．本資料の shell loop による複数条件実行を，より大きな設定管理へ拡張する際に参照できる．
- [MLflow: Tracking Hyperparameter Tuning](https://mlflow.org/docs/latest/ml/getting-started/hyperparameter-tuning/)
  - parameter sweep や hyperparameter tuning の各 run について，parameter，metric，artifact を記録し比較する流れを説明している．複数条件の実験結果を UI や tracking server で管理したい場合に参照できる．
- [テストを書かない研究者に送る、最初にテストを書く実験コード入門](https://speakerdeck.com/shunk031/lets-start-your-ml-project-by-writing-tests)
  - pytest と最初にテストを書く考え方を，機械学習の実験コードに適用するための資料である．本資料では，巨大な `main.py` から処理を切り出し，小さな関数単位で確認する背景として参照できる．
- [Performance Tips of NumPy ndarray](https://shihchinw.github.io/2019/03/performance-tips-of-numpy-ndarray.html)
  - NumPy ndarray の view と copy，メモリアクセスパターン，cache，vectorization，broadcasting の性能上の注意を実験例とともに説明している．本資料では，`for` 文をベクトル演算へ置き換えるときの利点と注意点の参考資料である．
- [tky823/ssspy: A Python toolkit for sound source separation](https://github.com/tky823/ssspy)
  - ICA，FDICA，IVA，ILRMA，IPSDTA，MNMF，PDS-BSS，ADMM-BSS，HVA，cACGMM など，多数の BSS 手法の実装と notebook を公開している toolkit である．手法別 module と IP/ISS/IPA 系の空間モデル更新 module が分かれており，BSS 手法を class と共通部品で整理する設計例として参照できる．
- [機械学習プロジェクトにおける実験管理](https://www.docswell.com/s/2625216247/ZQXY9J-2026-02-02-185832#p1)
  - 機械学習 project の再現性，拡張性の高いコード設計，依存性注入，コードと設定の分離を扱う講義資料である．本資料の発展編で扱う依存性注入と実験設計の参考資料である．
- [taishi-n/oobss: Open Online Blind Source Separation toolkit](https://github.com/taishi-n/oobss)
  - classical and online BSS algorithms，configuration，logging，documentation utilities を含む toolkit である．Batch separators として `AuxIVA`，`ILRMA`，online separators として `OnlineAuxIVA`，`OnlineILRMA`，`OnlineISNMF` が整理されており，手法拡張時の module 分割，共通 interface，benchmark 設計の参考になる．
- [生成AIとの付き合い方 / Generative AI and us](https://speakerdeck.com/kaityo256/generative-ai-and-us)
  - 生成 AI の活用では，出力に責任を持つこと，出力に付加価値を付けるために専門知識が必要であることを説明している講義資料である．本資料では，coding agent に研究コードの拡張を依頼するときの人間側の責任と学習内容の背景として参照できる．
