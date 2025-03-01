# Transformerを数式から理解する

## 1. はじめに

Transformerの構成要素のうち、特にmulti-head attentionを中心に理解する。Attention is all you need論文で紹介されている式について、そのイメージと合わせて解説する。

## 2. 時代背景と問題設定

### 2.1 Sequence-to-Sequence モデルの限界

RNNベースのモデルには以下の本質的な課題があった：

1. 逐次処理による計算の遅さ
2. 長期依存関係の捕捉が困難
3. 勾配消失/勾配爆発の問題

これらの問題は以下の数式で表現できる：

RNNの基本式：

$$
h_t = f(W_h h_{t-1} + W_x x_t + b)
$$

この再帰的な計算により、勾配は以下のように伝播する：

$$
\frac{\partial L}{\partial h_t} = ∏_{i=t}^T \frac{\partial h_{i+1}}{\partial h_i}
$$

### 2.2 解決すべき課題

Transformerは以下の要件を満たす必要があった：

1. 並列計算が可能な構造
2. 任意の距離の依存関係を直接モデル化
3. 系列における位置情報の保持

## 3. Transformerのアーキテクチャ概観

### 3.1 全体構造

Transformerは以下の主要コンポーネントで構成されている：

1. エンコーダー層（N=6）
   - Multi-head Self-attention
   - Position-wise Feed-Forward Network
   - Layer Normalization
   - Residual Connection

2. デコーダー層（N=6）
   - Masked Multi-head Self-attention
   - Multi-head Cross-attention
   - Position-wise Feed-Forward Network
   - Layer Normalization
   - Residual Connection

### 3.2 各レイヤーの数学的表現

Layer Normalization:

$$
\begin{align*}
LayerNorm(x) = \gamma\ \frac{x - \mu}{\sigma} + \beta \\
where\ \mu = E[x], \sigma = \sqrt{Var[x] + \epsilon}
\end{align*}
$$

Feed Forward Network:

$$
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
$$

## 4. Positional Encodingの数理

### 4.1 位置情報の表現方法

Transformerは位置情報を以下の式で表現します：

$$
\begin{align*}
\mathrm{PE}(i, 2k) = \sin \left( \frac{i}{10000^{2k/d_{model}}} \right) \\
\mathrm{PE}(i, 2k+1) = \cos \left( \frac{i}{10000^{2k/d_{model}}} \right)
\end{align*}
$$

次元が大きくなるごとに波長が$2\pi$から$2\pi \times 10000$まで徐々に大きくなる。

この式の特徴：
1. 位置の絶対値を保持
2. 相対位置の計算が容易
3. 任意の長さの系列に対応可能

### 4.2 実装と性質

位置エンコーディングは入力埋め込みに加算されます：

$$
Input = WordEmbedding + PositionalEncoding
$$

## 5. Attention Mechanismの数理

### 5.1 Scaled Dot-Product Attention

#### 基本式

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
$$

ここで：
- $Q = \begin{pmatrix} \overrightarrow{q_1}\\ \vdots\\ \overrightarrow{q_n} \end{pmatrix}$ : クエリ行列
- $K = \begin{pmatrix} \overrightarrow{k_1}\\ \vdots\\ \overrightarrow{k_m} \end{pmatrix}$ : キー行列
- $V = \begin{pmatrix} \overrightarrow{v_1}\\ \vdots\\ \overrightarrow{v_m} \end{pmatrix}$ : 値行列
- d_k: キーの次元数

softmaxは列方向のみに対して適用されるため、n個のクエリを一括で計算する式になっている。
※この論文中では、ベクトルが横ベクトルで表記されるため注意が必要。

そのため、まずは1行だけに着目して、計算式を理解する。

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax} \left( \frac{\overrightarrow{q_i} K^T}{\sqrt{d_k}} \right) V
$$

#### 1ステップ

$$
\begin{align*}
\overrightarrow{q_i} K^T &= \overrightarrow{q_i}
\left( \overrightarrow{k_1}^T, \cdots, \overrightarrow{k_m}^T \right) \\
&= \left( \overrightarrow{q_i}\ \overrightarrow{k_1}^T, \cdots, \overrightarrow{q_i}\ \overrightarrow{k_m}^T \right) \\
&= \left( \overrightarrow{q_i} \cdot \overrightarrow{k_1}, \cdots, \overrightarrow{q_i} \cdot \overrightarrow{k_m} \right) \\
\end{align*}
$$

内積は二つのベクトルの類似度を示すので、$\overrightarrow{q_i} \cdot \overrightarrow{k_1}$は$\overrightarrow{q_i}$と$\overrightarrow{k_1}$の類似度を示す。
そのため、$\overrightarrow{q_i} K^T$は、ある1つのクエリと全てのキーとの類似度を示した横ベクトルになる。

#### 2ステップ

$$
\frac{\overrightarrow{q_i} K^T}{\sqrt{d_k}}
$$

高次元のベクトルは、ノルムが長い傾向にある。
入力によって$\overrightarrow{q_i} K^T$が長かったり短かったりするのは、計算的にも不具合が起こりやすい。（詳しく言えば、後段のsoftmaxを行う際に、長いベクトルで勾配消失が起きやすい）
そこで、次元数に応じたスケーリングを行う必要がある。

（補足）
Q, Kの各成分が独立で標準正規分布（平均0、分散1）に従うとき、内積の期待値は0となるが、分散はdに比例する。そのため、標準偏差$\sqrt{d}$で割ることで、分散を1に調整できる。

#### 3ステップ

$$
\mathrm{softmax} \left( \frac{\overrightarrow{q_i} K^T}{\sqrt{d_k}} \right)
\coloneqq \left( p_1, \cdots, p_m \right)
$$

各類似度を総和1の確率に変換する。
すなわち、$\sum_i p_i = 1 (0 \leq p_i \leq 1)$を満たす確率$p_i$の横ベクトルに変換する。

#### 4ステップ
$$
\begin{align*}
\mathrm{softmax} \left( \frac{\overrightarrow{q_i} K^T}{\sqrt{d_k}} \right) V &= \left( p_1, \cdots, p_m \right)
\begin{pmatrix}
\overrightarrow{v_1}\\ \vdots\\ \overrightarrow{v_m}
\end{pmatrix} \\
&= \sum_i p_i \overrightarrow{v_i}
\end{align*}
$$

$\mathrm{softmax} \left( \frac{\overrightarrow{q_i} K^T}{\sqrt{d_k}} \right) V$は、重み$p_i$に基づく$\overrightarrow{v_i}$の重みづけ和を意味する。
クエリベクトル$\overrightarrow{q_i}$とキーベクトル$\overrightarrow{k_i}$が同じ方向を向いている場合に、値ベクトル$\overrightarrow{v_i}$の情報を重視して足し合わせるようなイメージ。

#### 5ステップ

$$
\mathrm{softmax} \left( \frac{Q K^T}{\sqrt{d_k}} \right) V = 
\begin{pmatrix}
p_{11} & \cdots & p_{1m} \\
\vdots & & \vdots \\
p_{n1} & \cdots & p_{nm}
\end{pmatrix}
\begin{pmatrix}
\overrightarrow{v_1}\\ \vdots\\ \overrightarrow{v_m}
\end{pmatrix} \\
$$

あとは、複数のクエリを行方向に並べて今の計算を行うと、基本式となる。

### 閑話休題
$K$, $V$をうまく学習することが重要そうだが、
アーキテクチャの図を見ると、$Q$にも$K$にも$V$にも同じ入力が入っている。
$Q$はクエリ行列だから入力がそのまま入るのは納得だが、$K$も$V$も入力そのままでは何の意味もなさそう。

### 5.2 Multi-Head Attention

#### 基本式

$$
\begin{align*}
\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head_1}, \cdots, \mathrm{head_h})W^O \\
where\ \mathrm{head_i} = \mathrm{Attention}(QW_i^Q, KW_i^K, VW_i^V)
\end{align*}
$$

#### 入力を回転させる
入ってきた入力$Q, K, V$に対して、それぞれ別の行列$W_i^Q, W_i^K, W_i^V$をかけて回転させる。
直感的な役割は以下の通り。（例：I am learning.）
- $Q$が「どの単語が重要か」を決定します。
    - 例: 「am」に焦点を当てる。
- $K$が「文脈情報」を提供します。
    - 例: 「am」の隣に「I」と「learning」があることを記憶。
- $V$が「具体的な単語の情報」を保持します。
    - 例: 「am」の具体的な意味や単語の特徴。
ここで、$h$個のheadを用意する場合は、$W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_{model} \times \frac{d_{model}}{h}}$。

#### 複数のattentionを行い、Concatで並べる
単一のheadの場合は、入力データ全体を1つの重み空間（$W_i^Q, W_i^K, W_i^V$による空間）に圧縮するため、異なる単語の特徴が同じ空間に集約され、重要な特徴が複数あったとしても平均化されてしまう可能性がある。
一方で、各ヘッドが異なる$W_i^Q, W_i^K, W_i^V$を持つ場合、入力データを様々な観点で解析できる。これにより、単語の文法的役割、意味的関連性、感情的ニュアンスなどをそれぞれ個別に捉えることが可能。

「猫がマットの上に座っている」という文脈の例を以下に示す。
文脈: “The cat sat on the mat.”
- 単一ヘッド:
    - すべての単語間の関係性を1つのスコアで表現。
    - 主語 (“cat”) と動詞 (“sat”) の関連性が他の単語の影響で埋もれる可能性。
- Multi-Head:
    - ヘッド1: 文法的な役割（主語-動詞、修飾語）に注目。
    - ヘッド2: 意味的な関連性（「cat」と「mat」の物理的関連性）。
    - ヘッド3: 時間的情報（「sat」動作の過去性）。

これらを統合することで、豊かな文脈情報を得る。

## 6. 実装上の考慮点

主要なポイント：

1. パラメータ初期化
```python
torch.nn.init.xavier_uniform_(
    tensor,
    gain=torch.nn.init.calculate_gain('relu')
)
```

2. 学習率スケジューリング
```
lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
```

## 7. 発展的話題

最新の改良点：

1. Sparse Attention
   - メモリ効率: O(n^2) → O(n log n)

2. Transformer-XL
   - セグメント単位の処理
   - 相対位置エンコーディング

## 8. まとめと今後の展望

Transformerの主な利点：
1. 並列計算による高速化
2. 長期依存関係の効率的な捕捉
3. スケーラブルなアーキテクチャ

今後の課題：
1. 計算量・メモリ使用量の削減
2. より効率的な位置表現の開発
3. ドメイン特化型アーキテクチャの探求