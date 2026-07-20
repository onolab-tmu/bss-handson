# NumPy broadcasting図の仕様と計算例

## 1. 目的

scalar，vector，matrix，tensorの要素ごとの加算を，NumPy broadcastingの有無と概念上の展開が分かる独立SVGとして生成する．
基本となる13個の加算に加え，broadcastできないshape，`None`の挿入位置，`np.dot`と`@`の動作，暗黙的な先頭次元の追加を示す9個の補足図を生成する．

## 2. 図の共通仕様

図の最上部に全体タイトルを置かない．
全ての図は描画エリアの幅を180 mmに固定し，縦方向には上下7 ptの余白を設ける．
図中のセル値，変数名，shape，演算子，注記は12 ptで表示する．
各データの直上に，変数名とshapeを次の二行形式で表示する．

```text
a
shape=(3,)
```

ラベルにはLaTeXの`\\ttfamily`を使い，同じ演算段にある変数名とshapeの垂直位置を最も背の高い配列へ揃える．
入力変数は`a`または`A`と`b`または`B`，出力は`c`または`C`とする．
broadcast後の概念上の配列にも，broadcast前と同じ変数名を付ける．

scalarとvectorは小文字，matrixとtensorは大文字で表記する．
図中の`@`はタイプライタ体で表示する．

broadcastが必要な図は，上段に元の演算，下段にbroadcast後の概念上の配列を使った等価な演算を置く．
下段の配列の大きさに合わせて上段の配列を配置し，上下で配列の中心を揃える．
broadcast矢印は，上段の配列から対応する下段の配列へ真下に向ける．

## 3. 基本となる加算パターン

順序を区別した次の13パターンを生成する．

1. scalar + vector
2. vector + scalar
3. scalar + matrix
4. matrix + scalar
5. vector + matrix
6. matrix + vector
7. scalar + tensor
8. tensor + scalar
9. vector + tensor
10. tensor + vector
11. matrix + tensor
12. tensor + matrix
13. tensor + tensor

同一shape同士の13はbroadcasting不要で，演算を一段だけ表示する．
01から12までは上段に元の演算，下段にbroadcast後の概念上の等価表現を表示する．

## 4. 補足図

基本となる13個の加算とは別に，次の9個の図を生成する．

14. shape `(3,3)` のmatrixとshape `(2,)` のvectorを加算できない例
15. `a[None, :]` によってshape `(1,3)` を作り，shape `(3,)` のvectorと加算する例
16. `a[:, None]` によってshape `(3,1)` を作り，shape `(3,)` のvectorと加算する例
17. 1次元vectorに対する `np.dot(a,b)` と `a @ b` の内積
18. 2次元matrixに対する `np.dot(A,B)` と `A @ B` の行列積
19. `@` が先頭側のバッチ軸をbroadcastして行列積を計算する例
20. `None` で横vectorと縦vectorを作り，`@` で内積を計算する例
21. `None` で縦vectorと横vectorを作り，`@` で外積を計算する例
22. shape `(3,)`を`(2,3,3)`へbroadcastするとき，先頭に長さ1の軸が暗黙的に追加される例

14は，末尾の次元 `3` と `2` が一致せず，どちらも `1` ではないため，図中には結果配列の代わりに`Error`を表示する．
15は一段の演算として表示する．
16は上段に元の演算，下段に二つの入力をshape `(3,3)` へbroadcastした等価な演算を表示する．
17と18は，1次元と2次元では `np.dot` と `@` が同じ結果を返すことを示す．
19は，`@`が先頭側のバッチ軸をbroadcastすることを示す．
20と21は，長さ1の軸を置く側によって内積と外積が切り替わることを示す．
22は，shapeを右端から揃え，不足する先頭軸を長さ1として補ってから相手の次元へ広げる過程を示す．

## 5. Broadcasting 規則

NumPyのtrailing-axis ruleを用いる．

- `() -> (3,)`
- `() -> (3,3)`
- `() -> (3,3,3)`
- `(3,) -> (1,3) -> (3,3)`
- `(3,) -> (1,1,3) -> (3,3,3)`
- `(3,3) -> (1,3,3) -> (3,3,3)`

軸数が少ないshapeは右端から揃え，不足する先頭軸を長さ1として扱う．
22では，shape `(3,)`を`(1,1,3)`として扱い，二つの長さ1の軸を広げて`(2,3,3)`にする．

補足図では次のshapeを扱う．

- `(3,3) + (2,)` は末尾の `3` と `2` が不一致であり，broadcastできない．
- `a[None, :]` は `(3,)` を `(1,3)` に変換し，`(1,3) + (3,)` の結果は `(1,3)` になる．
- `a[:, None]` は `(3,)` を `(3,1)` に変換し，`(3,1) + (3,)` の結果は `(3,3)` になる．

`np.dot` と `@` では次の規則を扱う．

- 二つの1次元配列に対する `np.dot` と `@` は，どちらも要素積の総和をscalarとして返す．
- 二つの2次元配列に対する `np.dot` と `@` は，どちらも通常の行列積を返す．
- `@` は最後の二つの軸をmatrixとして扱い，それより前のバッチ軸をbroadcastする．
- `(1,3) @ (3,1)` はshape `(1,1)` の内積となり，`(3,1) @ (1,3)` はshape `(3,3)` の外積となる．

vectorはmatrixの各行へ反復される．
tensorに対しては，vectorが各行と各サブ行列へ，matrixが各サブ行列へ反復される．

broadcast後の配列では，中央に配置した実線のセル群がbroadcast前の元配列に対応する．
破線セルは，その元配列から論理的に反復された成分を示し，物理コピーを意味しない．
NumPyは通常，strideが`0`のviewなどを使い，実データを複製せずに等価な計算を行う．

## 6. 配列の表示

- scalar: 単一セル
- vector: 横一列の3セル
- shape `(2,)`のvector: 横一列の2セル
- shape `(1,3)`: 横一列の3セル
- shape `(3,1)`: 縦一列の3セル
- shape `(2,2)`: 2行2列のセル
- shape `(2,2,2)`: 2個の2行2列matrixを右上方向へずらして積層
- matrix: 3行3列の平面グリッド
- tensor: 3行3列のmatrixを右上方向へずらして3枚積層
- shapeの概念図: 各次元を独立したセルで示し，暗黙的に追加または拡張される次元を破線と灰色で表示

tensorとバッチ配列には，奥行き方向の添字ラベルを表示しない．
matrix自体は平面であり，立方体の側面は描かない．

## 7. 視覚符号

- 実線と黒文字のセル: 元入力，結果，またはbroadcast後も元配列に対応する成分
- 破線と灰色の文字のセル: broadcastによって反復された成分
- 垂直な下向き矢印: 元データからbroadcast後データへの変換
- 矢印ラベル: 説明語を付けず，shapeの変化だけを表示
- タイプライタ体の`@`: NumPyの行列積演算子

破線セルには縦横とも1 mmの線分と1 mmの間隔を使う．
元配列に対応する中央のセル群は実線で囲み，それ以外の外枠と内部境界を破線にする．
共有境界は重ね描きせず，一度だけ描く．
結果配列は常に実線とする．

## 8. ファイル構成

```text
docs/
└── img/
    ├── SPECIFICATION.md
    ├── add-scalar-vector.svg
    ├── ...
    ├── leading-axis.svg
    └── memory-order.svg
```

`broadcast-common.tex`は，セル，ラベル，矢印，scalar，vector，matrix，tensor，バッチ配列の描画マクロを定義する．
各`.tex`は単独でコンパイル可能な`standalone`文書である．

## 9. 出力

各`.tex`を`pdflatex`でPDF化し，Inkscapeで独立SVGへ変換する．
PDFとSVGの描画エリア幅は180 mmとする．
SVGへ変換するときはPDFのページ全体を出力範囲とし，余白を図形の外接範囲へ切り詰めない．
SVGは透明背景，外部画像なし，文字欠落なし，クリッピングなしとする．

## 10. 各図の計算例

以下のコードブロックは，`import numpy as np`を実行した後に個別に実行できる．

### add-scalar-vector

scalar `2`を3要素へ反復し，shape `(3,)`のvectorへ加算する図である．

```python
a = np.array(2)
b = np.arange(3)
c = a + b

# c == [2, 3, 4]
# c.shape == (3,)
```

### add-vector-scalar

vectorとscalarの順序を入れ替え，scalar `2`を3要素へ反復して加算する図である．

```python
a = np.arange(3)
b = np.array(2)
c = a + b

# c == [2, 3, 4]
# c.shape == (3,)
```

### add-scalar-matrix

scalar `2`を3行3列へ反復し，shape `(3,3)`のmatrixへ加算する図である．

```python
a = np.array(2)
B = np.arange(9).reshape(3, 3)
C = a + B

# C == [[2, 3, 4],
#       [5, 6, 7],
#       [8, 9, 10]]
# C.shape == (3, 3)
```

### add-matrix-scalar

matrixとscalarの順序を入れ替え，scalar `2`を3行3列へ反復して加算する図である．

```python
A = np.arange(9).reshape(3, 3)
b = np.array(2)
C = A + b

# C == [[2, 3, 4],
#       [5, 6, 7],
#       [8, 9, 10]]
# C.shape == (3, 3)
```

### add-vector-matrix

shape `(3,)`のvectorをmatrixの各行へ反復し，shape `(3,3)`のmatrixへ加算する図である．

```python
a = np.arange(3)
B = np.arange(9).reshape(3, 3)
C = a + B

# C == [[0, 2, 4],
#       [3, 5, 7],
#       [6, 8, 10]]
# C.shape == (3, 3)
```

### add-matrix-vector

matrixとvectorの順序を入れ替え，vectorをmatrixの各行へ反復して加算する図である．

```python
A = np.arange(9).reshape(3, 3)
b = np.arange(3)
C = A + b

# C == [[0, 2, 4],
#       [3, 5, 7],
#       [6, 8, 10]]
# C.shape == (3, 3)
```

### add-scalar-tensor

scalar `2`を3枚のサブ行列全体へ反復し，shape `(3,3,3)`のtensorへ加算する図である．

```python
a = np.array(2)
B = np.arange(27).reshape(3, 3, 3)
C = a + B

# C[k, i, j] == B[k, i, j] + 2
# C.shape == (3, 3, 3)
```

### add-tensor-scalar

tensorとscalarの順序を入れ替え，scalar `2`をtensorの全要素へ反復して加算する図である．

```python
A = np.arange(27).reshape(3, 3, 3)
b = np.array(2)
C = A + b

# C[k, i, j] == A[k, i, j] + 2
# C.shape == (3, 3, 3)
```

### add-vector-tensor

shape `(3,)`のvectorをtensorの各行と各サブ行列へ反復して加算する図である．

```python
a = np.arange(3)
B = np.arange(27).reshape(3, 3, 3)
C = a + B

# C[k, i, :] == B[k, i, :] + [0, 1, 2]
# C.shape == (3, 3, 3)
```

### add-tensor-vector

tensorとvectorの順序を入れ替え，vectorをtensorの各行と各サブ行列へ反復して加算する図である．

```python
A = np.arange(27).reshape(3, 3, 3)
b = np.arange(3)
C = A + b

# C[k, i, :] == A[k, i, :] + [0, 1, 2]
# C.shape == (3, 3, 3)
```

### add-matrix-tensor

shape `(3,3)`のmatrixをtensorの3枚のサブ行列へ反復して加算する図である．

```python
A = np.arange(9).reshape(3, 3)
B = np.arange(27).reshape(3, 3, 3)
C = A + B

# C[k, :, :] == A + B[k, :, :]
# C.shape == (3, 3, 3)
```

### add-tensor-matrix

tensorとmatrixの順序を入れ替え，matrixをtensorの3枚のサブ行列へ反復して加算する図である．

```python
A = np.arange(27).reshape(3, 3, 3)
B = np.arange(9).reshape(3, 3)
C = A + B

# C[k, :, :] == A[k, :, :] + B
# C.shape == (3, 3, 3)
```

### add-tensor-tensor

shapeがともに`(3,3,3)`のtensorを要素ごとに加算する図であり，broadcastは発生しない．

```python
A = np.arange(27).reshape(3, 3, 3)
B = np.ones((3, 3, 3), dtype=int)
C = A + B

# C[k, i, j] == A[k, i, j] + 1
# C.shape == (3, 3, 3)
```

### incompatible-shapes

shape `(3,3)`のmatrixとshape `(2,)`のvectorを加算できない例を示す．

```python
A = np.arange(9).reshape(3, 3)
b = np.array([10, 20])

A + b
# ValueError: operands could not be broadcast together
# with shapes (3,3) (2,)
```

### row-vector

`a[None, :]`によって`a`をshape `(1,3)`の横一列に変換し，shape `(3,)`の`b`と加算する例を示す．

```python
a = np.arange(3)
b = np.array([10, 20, 30])
a_row = a[None, :]
c = a_row + b

# a_row.shape == (1, 3)
# c == [[10, 21, 32]]
# c.shape == (1, 3)
```

### column-vector

`a[:, None]`によって`a`をshape `(3,1)`の縦一列に変換し，二つの入力をshape `(3,3)`へbroadcastして加算する例を示す．

```python
a = np.arange(3)
b = np.array([10, 20, 30])
a_column = a[:, None]

a_broadcast = np.broadcast_to(a_column, (3, 3))
b_broadcast = np.broadcast_to(b, (3, 3))
C = a_column + b

# a_broadcast == [[0, 0, 0],
#                 [1, 1, 1],
#                 [2, 2, 2]]
# b_broadcast == [[10, 20, 30],
#                 [10, 20, 30],
#                 [10, 20, 30]]
# C == [[10, 20, 30],
#       [11, 21, 31],
#       [12, 22, 32]]
# C.shape == (3, 3)
```

### vector-inner-product

二つの1次元vectorに対する`np.dot(a,b)`と`a @ b`が同じscalarの内積を返す例を示す．

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c_dot = np.dot(a, b)
c_matmul = a @ b

# c_dot == c_matmul == 32
```

### matrix-product

二つの2次元matrixに対する`np.dot(A,B)`と`A @ B`が同じ行列積を返す例を示す．

```python
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])
C_dot = np.dot(A, B)
C_matmul = A @ B

# C_dot == C_matmul == [[19, 22],
#                       [43, 50]]
```

### batch-matmul

`@`が長さ1のバッチ軸をbroadcastし，各バッチで2行2列の行列積を計算する例を示す．

```python
A = np.array([[[1, 0],
               [0, 1]],
              [[2, 0],
               [0, 2]]])
B = np.array([[[1, 2],
               [3, 4]]])
C = A @ B

# C == [[[1, 2],
#        [3, 4]],
#       [[2, 4],
#        [6, 8]]]
# C.shape == (2, 2, 2)
```

### matmul-inner-product

`None`でshape `(1,3)`と`(3,1)`を作り，`@`でshape `(1,1)`の内積を得る例を示す．

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
a_row = a[None, :]
b_column = b[:, None]
C_matmul = a_row @ b_column
C_dot = np.dot(a_row, b_column)

# C_matmul == C_dot == [[32]]
# C_matmul.shape == (1, 1)
```

### matmul-outer-product

`None`でshape `(3,1)`と`(1,3)`を作り，`@`でshape `(3,3)`の外積を得る例を示す．

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
a_column = a[:, None]
b_row = b[None, :]
C_matmul = a_column @ b_row
C_dot = np.dot(a_column, b_row)

# C_matmul == C_dot == [[4, 5, 6],
#                       [8, 10, 12],
#                       [12, 15, 18]]
# C_matmul.shape == (3, 3)
```

### leading-axis

shape `(3,)`の`a`をshape `(2,3,3)`の`B`と加算するとき，不足する先頭軸が長さ1として暗黙的に補われる過程を示す．

```python
a = np.arange(3)
B = np.arange(18).reshape(2, 3, 3)
a_aligned = a.reshape(1, 1, 3)
a_expanded = np.broadcast_to(a_aligned, B.shape)
C = a + B

# a.shape == (3,)
# a_aligned.shape == (1, 1, 3)
# a_expanded.shape == C.shape == (2, 3, 3)
# np.array_equal(C, a_expanded + B)
```

## 11. 検証

修正後は次を確認する．

- `SPECIFICATION.md`に定義した各図に対応するSVGが存在する．
- 全てのPDFとSVGの描画エリア幅が180 mmである．
- SVGへの変換後もPDFのページ全体と余白が維持されている．
- 図中の全文字が12 ptである．
- ファイル名とオペランド順が一致する．
- 全セルの計算結果が正しい．
- 各データの上に変数名とshapeがタイプライタ体で表示される．
- 同じ演算段の変数名とshapeの垂直位置が揃っている．
- 図全体のタイトルが存在しない．
- broadcast後の配列で，元配列に対応する中央のセル群だけが実線と黒文字である．
- broadcastによって反復されたセルだけが破線と灰色の文字である．
- 破線セルの縦線と横線で，線分の長さと間隔が一致している．
- 破線セルの共有境界が重ね描きされていない．
- broadcast前後の配列の中心が揃い，矢印が真下を向いている．
- vector，matrix，tensorのtrailing-axis対応が正しい．
- tensorとバッチ配列が右上方向へ積層されている．
- tensorとバッチ配列に奥行き方向の添字ラベルがない．
- 図中の`@`がタイプライタ体で表示される．
- PDFとSVGにクリッピングがない．
- 14では末尾次元の不一致に対して，図中に`Error`が表示されている．
- 15の結果がshape `(1,3)` で，要素が `[10,21,32]` である．
- 16の結果がshape `(3,3)` で，各行が `[10,20,30]`，`[11,21,31]`，`[12,22,32]` である．
- 16ではshape `(3,1)` の入力が列方向へ，shape `(3,)` の入力が行方向へそれぞれ正しくbroadcastされている．
- 17の `np.dot(a,b)` と `a @ b` がともにscalar `32` である．
- 18の2種類の行列積がともに `[[19,22],[43,50]]` である．
- 19でshape `(1,2,2)` の `B` がバッチ軸方向へbroadcastされ，結果のshapeが `(2,2,2)` である．
- 20の結果がshape `(1,1)` で，値が `32` である．
- 21の結果がshape `(3,3)` で，各行が `[4,5,6]`，`[8,10,12]`，`[12,15,18]` である．
- 22ではshape `(3,)`が`(1,1,3)`として扱われ，`(2,3,3)`へbroadcastされている．

## 12. 参考資料

- [numpy.dot](https://numpy.org/doc/stable/reference/generated/numpy.dot.html)
- [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)
- [Indexing on ndarrays](https://numpy.org/doc/stable/user/basics.indexing.html)
