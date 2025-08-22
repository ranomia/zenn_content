---
title: Mambaを数式から理解する
emoji: "🔍"
type: "tech"
topics: ["Mamba", "SSM"]
published: false
---

# 1. はじめに

近年、Transformer以外のアーキテクチャが注目を集める中、**Mamba**は特に有望な選択肢として登場した。MambaはState Space Model（SSM）をベースとした新しいアーキテクチャで、長いシーケンスを効率的に処理できる特徴を持つ。

本記事では、Mambaの核となる数学的原理を段階的に解説する。まず基本的なState Space Modelから始まり、Selective SSMの概念、そして最終的にMambaアーキテクチャの全体像まで、数式を用いて詳しく説明する。

Transformerの計算量がシーケンス長に対して二次的に増加するのに対し、Mambaは線形時間で処理可能という大きなアドバンテージがある。この効率性の秘密を数式レベルで理解することで、なぜMambaが長文処理に優れているのかが明確にしたい。

**注釈：** 本記事は以下の論文を参照している：  
Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv preprint arXiv:2312.00752.

# 2. State Space Model(SSM)とは

State Space Model（状態空間モデル）は、システムの内部状態を数学的に表現する手法である。SSMは以下の2つの方程式で定義される：

**状態方程式：**
$$h_{t+1} = Ah_t + Bx_t$$

**出力方程式：**
$$y_t = Ch_t + Dx_t$$

ここで、$h_t \in \mathbb{R}^N$は時刻$t$での隠れ状態、$x_t \in \mathbb{R}$は入力、$y_t \in \mathbb{R}$は出力である。$A \in \mathbb{R}^{N \times N}$は状態遷移行列、$B \in \mathbb{R}^{N \times 1}$は入力行列、$C \in \mathbb{R}^{1 \times N}$は出力行列、$D \in \mathbb{R}$は直接伝達項である。

この定式化により、RNNと比較して並列計算が可能となり、長いシーケンスでも効率的な処理が実現される。MambaはこのSSMを発展させ、パラメータを動的に調整することで高い表現力を獲得している。

![](/images/understand_mamba/selective_state_space_model.png)

# 3. Selective SSMの概念

従来のSSMではパラメータ$A$、$B$、$C$は固定値だったが、Selective SSMでは入力$x_t$に応じてこれらを動的に変化させる。この選択的機構により、重要な情報を記憶し、不要な情報を忘却する能力が向上する。

**Selective SSMの定式化：**
$$A_t = \text{Selective}(x_t, \Delta_t)$$
$$B_t = \text{Linear}_B(x_t)$$
$$C_t = \text{Linear}_C(x_t)$$

ここで各記号の定義は以下の通りである：

- $A_t \in \mathbb{R}^{N \times N}$：時刻$t$での状態遷移行列（入力依存）
- $B_t \in \mathbb{R}^{N \times 1}$：時刻$t$での入力行列（入力依存）
- $C_t \in \mathbb{R}^{1 \times N}$：時刻$t$での出力行列（入力依存）
- $\Delta_t \in \mathbb{R}$：時刻$t$での離散化パラメータ
- $\text{Selective}(\cdot)$：選択的関数（入力と離散化パラメータから状態遷移行列を生成）
- $\text{Linear}_B(\cdot)$、$\text{Linear}_C(\cdot)$：線形変換関数

状態更新は以下のように表される：

$$\bar{A}_t = \exp(\Delta_t A_t)$$
$$\bar{B}_t = (\Delta_t A_t)^{-1}(\bar{A}_t - I)\Delta_t B_t$$
$$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$$

ここで：

- $\bar{A}_t \in \mathbb{R}^{N \times N}$：離散化された状態遷移行列
- $\bar{B}_t \in \mathbb{R}^{N \times 1}$：離散化された入力行列
- $I \in \mathbb{R}^{N \times N}$：単位行列
- $\exp(\cdot)$：行列の指数関数

この機構により、入力に応じて適応的に状態を更新でき、長期依存関係の学習が大幅に改善される。

# 4. Mambaのアーキテクチャ

Mambaは前述のSelective SSMを核として構築されたアーキテクチャである。ここでは、完全なMambaブロックの数学的定式化と、その革新的な計算効率性の実現方法について解説する。

## 4.1 Mambaブロックの全体構造

Mambaブロックは以下のような構造を持つ：

**入力処理：**
$$\mathbf{x}' = \text{LayerNorm}(\mathbf{x})$$
$$\mathbf{u} = \mathbf{x}' W_U$$
$$\mathbf{v} = \mathbf{x}' W_V$$

**選択的パラメータ生成：**
$$\Delta = \text{Softplus}(\mathbf{u} W_{\Delta} + b_{\Delta})$$
$$\mathbf{B} = \mathbf{u} W_B$$
$$\mathbf{C} = \mathbf{u} W_C$$

ここで：
- $\mathbf{x} \in \mathbb{R}^{L \times D}$：入力シーケンス（$L$：シーケンス長、$D$：特徴次元）
- $\mathbf{u}, \mathbf{v} \in \mathbb{R}^{L \times d_{model}}$：中間表現
- $W_U, W_V \in \mathbb{R}^{D \times d_{model}}$：線形投影行列
- $\Delta \in \mathbb{R}^{L \times N}$：離散化パラメータ
- $\mathbf{B} \in \mathbb{R}^{L \times N}$、$\mathbf{C} \in \mathbb{R}^{L \times N}$：選択的パラメータ

![](/images/understand_mamba/mamba_block_architecture.png)

## 4.2 Selective SSMの効率的実装

Mambaの核心は、Selective SSMを効率的に計算する機構にある。状態遷移は以下のように計算される：

**離散化処理：**
$$\bar{\mathbf{A}} = \exp(\Delta \odot \mathbf{A})$$
$$\bar{\mathbf{B}} = \Delta \odot \mathbf{B}$$

**状態更新の再帰的計算：**
$$\mathbf{h}_t = \bar{\mathbf{A}}_t \odot \mathbf{h}_{t-1} + \bar{\mathbf{B}}_t \odot \mathbf{u}_t$$
$$\mathbf{y}_t = \mathbf{C}_t \odot \mathbf{h}_t$$

ここで：
- $\odot$：要素ごとの積（Hadamard積）
- $\mathbf{A} \in \mathbb{R}^N$：学習可能な状態遷移パラメータ（対角化により効率化）
- $\mathbf{h}_t \in \mathbb{R}^{N}$：時刻$t$での隠れ状態

## 4.3 ゲーティング機構とスキップ接続

Mambaではゲーティング機構を用いて情報の流れを制御する：

**SiLU活性化によるゲーティング：**
$$\mathbf{z} = \text{SiLU}(\mathbf{v}) \odot \mathbf{y}$$

**最終出力：**
$$\text{output} = \mathbf{x} + \mathbf{z} W_O$$

ここで：
- $\text{SiLU}(x) = x \cdot \sigma(x)$：SiLU（Swish）活性化関数
- $W_O \in \mathbb{R}^{d_{model} \times D}$：出力投影行列

## 4.4 計算効率性の実現

Mambaが線形時間複雑度を達成する秘密は、以下の工夫にある：

1. **対角化による効率化**：状態遷移行列$\mathbf{A}$を対角行列に制限することで、行列積計算を要素ごとの積に簡約

2. **並列スキャン**：再帰的な状態更新を並列スキャンアルゴリズムで効率化

3. **ハードウェア最適化**：GPU上での効率的な実装を可能にする数値計算の工夫

これらの工夫により、Mambaはシーケンス長$L$に対して$O(L)$の計算量を実現し、Transformerの$O(L^2)$を大幅に改善している。特に長いシーケンスにおいて、この差は顕著な性能向上をもたらす。

# 5. その他備忘
## FAQ

### Q. 状態更新の数式変形がわからない

**A.** 詳細な数式変形については以下の資料を参照してください：
https://hyaguchi947d.github.io/pdf/discretization.pdf

### Q. Mambaのデータの次元がわからない

**A.** Mambaでは以下の3つの軸を意識するのが重要です：

- **系列軸**：$t=1,\dots, L$（時間／トークン位置）
- **特徴軸**：埋め込み／チャンネル次元 $D$
- **状態軸**：SSMの次数 $S$（各チャンネルが持つ内部状態の次元）

入力テンソルの形状は $U \in \mathbb{R}^{B \times L \times D}$ となります。

各層は時間$t$ごとに、チャンネルごとに小さな状態$x_t \in \mathbb{R}^S$を更新します。

> **補足：**
> - 一般的なNLPにおいて、$B$（Batch size）が同時に処理する系列の本数、$L$（Sequence length）が系列の長さ、$D$（Feature）が各トークンを表すベクトルの次元として、入力テンソルの形は `X: (B, L, D)` となります
> - Mamba固有の特徴として、Mamba層は各チャンネル（=$D$の各成分）ごとに小さな状態ベクトル（次元を$S$と表すことが多い）を持ち、時刻$t$を進めながら更新します