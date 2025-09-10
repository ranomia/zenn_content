---
title: Mambaを数式から理解する
emoji: "🔍"
type: "tech"
topics: ["Mamba", "SSM"]
published: false
---

# はじめに

近年、Transformer以外のアーキテクチャが注目を集める中、**Mamba**は特に有望な選択肢として登場した。MambaはState Space Model（SSM）をベースとした新しいアーキテクチャで、長いシーケンスを効率的に処理できる特徴を持つ。

本記事では、Mambaの核となる数学的原理を段階的に解説する。まず基本的なState Space Modelから始まり、Selective SSMの概念、そして最終的にMambaアーキテクチャの全体像まで、数式を用いて詳しく説明する。

Transformerの計算量がシーケンス長に対して二次的に増加するのに対し、Mambaは線形時間で処理可能という大きなアドバンテージがある。この効率性の秘密を数式レベルで理解することで、なぜMambaが長文処理に優れているのかが明確にしたい。

**注釈：** 本記事は以下の論文を参照している：  
Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv preprint arXiv:2312.00752.

Gu, A., Goel, K., & Re, C. (2022). *Efficiently Modeling Long Sequences with Structured State Spaces*. arXiv preprint arXiv:2111.00396.

# 0. 時系列モデリングの本質的課題: LRDs
時系列モデリングにおいて、**Long-range dependences(LRDs)を含むデータの扱い**がしばしば問題になる。

LRDsとは、CTM(Continuous-time model), RNN, CNN, Transformerなどの手法には、それぞれLRDsに対応するための特殊な手法が提案されている。

> 例
> 
> - Orthogonal RNN, Lipschitz RNN (勾配消失への対策)
> 
> - Dilated convolutions (コンテキストサイズの拡張)
>
> - Transformerの拡張手法 (2次依存性への対策)

しかし、これらの手法は、Long Range Arena(LRA)や生のオーディオ分類のような難しいベンチマークに対しては性能が低い。

そこで、制御工学をはじめとする多くの学問において時系列データのモデリングに使われてきた**状態空間表現**を踏襲した、線形状態空間層(Linear State Space Layer: LSSL)が注目を浴びた。

# 1. 線形状態空間層(Linear State Spave Layer: LSSL)

![](/images/understand_mamba/three_veiws_of_lssl.png)

## 1-1. LSSLの概要
状態空間表現 $A,B,C,D$ が与えられたとき、LSSLは線形状態空間モデル(1), (2)を離散化して定義されるシーケンス間のマッピングである。

$$
\begin{align}
\dot{x}(t) &= Ax(t) + Bu(t) \\
y(t) &= Cx(t) + Du(t)
\end{align}
$$

具体的には、LSSL層はパラメータ $A,B,C,D,\Delta$ を持つ。これは、各タイムステップが$D$次元の特徴ベクトルを持つ長さ$L$のシーケンスを表す入力 $u\in\mathbb{R}^{L\times H}$ に対して動作する。

各特長 $d\in [D]$ は、シーケンス$(u_t^{(h)})_{t\in[L]}$を定義し、これをタイムスケール$\Delta$と組み合わせて離散化状態空間モデル(3)+(4)により出力 $y^{(h)} \in \mathbb{R}^L$ を定める。

$$
\begin{align}
x_t &= \bar{A}x_{t-1} + \bar{B}u_t \\
y_t &= Cx_t + Du_t
\end{align}
$$

LSSLの有用性については、原著を参照。「LSSLはリカレントである」「LSSLは畳み込みである」「LSSLは連続時間である」という3つの観点から、説明が記載されている。

## 1-2. LSSLの課題点

**第一に、LSSLは計算量が膨大である。**

LSSLは状態次元 $N$ 、シーケンス長 $L$ に対して、潜在状態の計算に **$O(N^2 L)$** の演算と **$O(NL)$** のメモリ空間が必要であり、RNNやCNNと比較して桁違いに大きい。

> [解決策]
> 
> 次節の**S4モデル**では、 **$\tilde{O}(N+L)$** の演算と $O(N+L)$ のメモリ空間に改善

**第二に、純粋なLSSLは勾配がシーケンス長に比例して消失・発散する問題がある。**

離散状態空間モデルでは、状態行列が以下の表現となり、これは $A$ を反復的に乗算することに等しい。
そのため、$A$の形状次第で、勾配が消失・発散する可能性が高い。
$$
\begin{align*}
\bar{A} &= \mathrm{exp}(\Delta A) \\
&= I + \Delta A + \frac{(\Delta A)^2}{2!} + \cdots
\end{align*}
$$

> [解決策]
>
> Aの形状を以下に指定するHiPPOフレームワークを用いることで、上記問題を解消。
> 
> $$
A_{nk} = 
\begin{cases}
(2n+1)^{\frac{1}{2}} (2k+1)^{\frac{1}{2}} & \text{if } n>k, \\
n+1 & \text{if } n=k, \\
0 & \text{if } n<k
\end{cases}
$$

# 2. Structured State Space Model (S4 Model)

## 2-1. S4の概要

S4は、離散時間SSMのボトルネックであるAの反復的な乗算を解消するために提案されたモデル。

モチベーションは$A^N$を効率的に解くことで、**対角化**によってこれを実現した。

以下、ざっくりとした理論部分紹介。詳細は原著を参照。

連続時間SSMをゼロ次ホールドで離散化すると、以下のような式になる。

$$
\begin{align*}
\bar{A} &= \mathrm{exp} (\Delta A) \\
\bar{B} &= (\Delta A)^{-1} (\mathrm{exp}(\Delta A) - I) \cdot \Delta B
\end{align*}
$$

S4では、Aを「Normal Plus Low-rank(NPLR)」として表現する。

$$
\begin{align*}
A &= V \Lambda V^\ast - pq^\top \\
&= V(\Lambda - V^\ast p (V^\ast p)^\ast)V^\ast
\end{align*}
$$
ここで、 $V\in\mathbb{C}^{N\times N}$ はユニタリ行列、 $\Lambda$ は対角行列、 $p, q\in\mathbb{R}^{N\times r}$ は低ランク行列を示す。

理想的には $A=V \Lambda V^\ast$ が成り立てば容易に対角化可能だが、任意のHiPPO行列に対して成り立たない。

そこで、低ランク行列との和の形式にすることで、任意のHiPPO行列がNPLR表現を有することを示した。

## 2-2. S4の課題点

**S4はLinear Time Invariant(LTI)モデル**

S4（及びSSM）は、一定のダイナミクス(すなわち $\bar{A}, \bar{B}$ 等のパラメータが静的)であり、文脈から正しい情報を選択させたり、入力依存的に状態に影響を与えられない。

> [解決策]
>
> 次節のSelective SSM(Mamba)では、モデルを時不変から**時変**とすることで、入力や状態に対する集中度を制御できる。

# 3. Selective SSM(Mamba)

![](/images/understand_mamba/selective_state_space_model.png)

## 3-0. Selective SSMの動機

筆者はシーケンスモデルの本質的な問題は、「コンテキストを小さな状態に圧縮すること」にあると言及している。

例えば、Attentionは文脈を明示的に全く圧縮しないため、効果的であるが**非効率的**。

実際に、コンテキスト全体(=KVキャッシュ)を明示的に保存する必要があり、これがTransformerのネックになっている。
> Transformerでは、学習時間が2次、推論時間が線形

一方、リカレントモデルでは有限の状態を持つため、**効率的**
ただし、状態が文脈をどれだけ圧縮しているかで**効果**が異なる。
> リカレントモデルでは、学習時間が線形、推論時間が定時間

選択メカニズムをモデルに組み込む方法の一つは、シーケンスに沿った相互作用に影響を与えるパラメータ(例えば、RNNのリカレントダイナミクスやCNNの畳み込みカーネル)を入力依存にすることである。

そこで、Selective SSMでは


## 3-1. Selective SSMの概要

Selective SSM(Mamba)の大きな工夫は、パラメータ $B, C, \Delta$ が次元 $L$を有するようになり、モデルが時不変から時変になった点である。

理論部分の概要を以下に示す。

Selective SSMでは、 $B, C, \Delta$ を以下に定義する $s_B(x), s_C(x), \tau_\Delta(\mathrm{Parameter}+s_\Delta(x))$ で置き換えている。

$$
\begin{align*}
s_B(x) &= \mathrm{Linear}_N(x) \\
s_C(x) &= \mathrm{Linear}_N(x) \\
s_\Delta(x) &= \mathrm{Broadcast}_D(\mathrm{Linear}_1(x)) \\
\tau_\Delta &= \mathrm{softplus}
\end{align*}
$$

論文中で、S4とSelective SSMのアルゴリズムの違いが示されている。

![](/images/understand_mamba/mamba_algorithm.png)

> ここで、RNNの古典的なゲーティング機構は、SSMの選択機構の特殊ケースとなる。
> 
> $N=1, A=-1, B=1, s_\Delta=\mathrm{Linear}(x), \tau_\Delta=\mathrm{softplus}$の時、Selective SSMは次のような形になる。
> 
> $$
\begin{align*}
g_t &= \sigma (\mathrm{Linear}(x_t)) \\
h_t &= (1-g_t)h_{t-1} + g_t x_t
\end{align*} $$
> 
> "um"のようなフィラーの英単語は、 $g_t=0$ でフィルター可能となる。

$\Delta$は、現在の入力$x_t$をどれだけ注目／無視するかを制御するパラメータであり、RNNゲートを一般化した変数である。

連続時間SSM(1)-(2)は、タイムステップ $\Delta$ で離散化された連続系と解釈でき、 $\Delta$ が大きい場合は現在の入力に長く注目、 $\Delta$ が小さい場合は現在の入力を軽視するような挙動を示す。

## 3-2. Selective SSMの課題点

本稿では扱わないが、Mamba-2の論文では、Mambaの課題として以下の点が挙げられている。

- アルゴリズム／システム面での訓練効率
  - Transformerに比べ、理論的整理やハードウェア最適化の蓄積が乏しい
- Selective scanカーネルのハードウェア親和性の低さ
  - Mambaの最適化スキャンは、行列演算ユニットを活用できない
- 状態次元 $N$ 拡大時のスケーリング悪化
  - Mambaのスキャン実装は状態次元 $N$ を大きくすると線形に遅くなる
  - Mamba2のState Space Duality(SSD)では、大きな状態サイズを扱える
- 並列学習の難しさ
  - データ依存の射影がブロック内に散在し、並列化構成をとりづらい設計。

# 4. その他備忘

## FAQ

### Q. 状態更新の数式変形がわからない

**A.** 詳細な数式変形については以下の資料を参照：
https://hyaguchi947d.github.io/pdf/discretization.pdf

### Q. Mambaのデータの次元がわからない

**A.** Mambaでは以下の3つの軸を意識するのが重要：

- **系列軸**：$t=1,\dots, L$（時間／トークン位置）
- **特徴軸**：埋め込み／チャンネル次元 $D$
- **状態軸**：SSMの次数 $S$（各チャンネルが持つ内部状態の次元）

入力テンソルの形状は $U \in \mathbb{R}^{B \times L \times D}$ となる。

各層は時間$t$ごとに、チャンネルごとに小さな状態$x_t \in \mathbb{R}^S$を更新する。

> **補足：**
> - 一般的なNLPにおいて、$B$（Batch size）が同時に処理する系列の本数、$L$（Sequence length）が系列の長さ、$D$（Feature）が各トークンを表すベクトルの次元として、入力テンソルの形は `X: (B, L, D)` となる
> - Mamba固有の特徴として、Mamba層は各チャンネル（=$D$の各成分）ごとに小さな状態ベクトル（次元を$S$と表すことが多い）を持ち、時刻$t$を進めながら更新する
