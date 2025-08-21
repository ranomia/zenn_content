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

# 2. State Space Model(SSM)とは

State Space Model（状態空間モデル）は、システムの内部状態を数学的に表現する手法である。SSMは以下の2つの方程式で定義される：

**状態方程式：**
$$h_{t+1} = Ah_t + Bx_t$$

**出力方程式：**
$$y_t = Ch_t + Dx_t$$

ここで、$h_t \in \mathbb{R}^N$は時刻$t$での隠れ状態、$x_t \in \mathbb{R}$は入力、$y_t \in \mathbb{R}$は出力である。$A \in \mathbb{R}^{N \times N}$は状態遷移行列、$B \in \mathbb{R}^{N \times 1}$は入力行列、$C \in \mathbb{R}^{1 \times N}$は出力行列、$D \in \mathbb{R}$は直接伝達項である。

この定式化により、RNNと比較して並列計算が可能となり、長いシーケンスでも効率的な処理が実現される。MambaはこのSSMを発展させ、パラメータを動的に調整することで高い表現力を獲得している。

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
