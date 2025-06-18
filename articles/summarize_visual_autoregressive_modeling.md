---
title: 論文要約メモ "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"
emoji: "📝"
type: "tech"
topics: ["機械学習", "NLP", "LLM", "トークナイザー", "BPE"]
published: False
---

## 1. どんなもの？
画像生成における新しいオートレグレッシブ（AR）学習手法VAR（Visual AutoRegressive）を提案。従来の「次のトークン予測」ではなく、「次の解像度（scale）予測」によるcoarse-to-fine型の生成を採用。ImageNetにおいて高精度かつ高速な画像生成を実現し、初めてARモデルが拡散モデルを超えた。

## 2. 先行研究と比べてどこがすごいの？
- GPT型ARモデルで拡散モデル（DiTなど）を初めて超えた（FID, IS, スピードなど多軸で）。
- 既存ARのボトルネック（flattening, 時間計算量, 双方向依存性無視）を次スケール予測で解消。
- LLMで観測されるスケーリング則とゼロショット汎化能力を画像生成モデルで再現。

## 3. 技術や手法の"キモ"はどこにある？
- multi-scale VQVAEで画像を複数解像度にトークン化
- Transformerは1×1 → 2×2 → … のようにトークンマップを段階的に予測（各段階で全トークン並列生成）
- 各scaleに対して因果的アテンションマスクを適用し、前のscaleだけに依存
- flatten不要で局所性保持＆計算量$O(n^4)$に抑制（従来ARは$O(n^6)$）

## 4. どうやって有効だと検証した？
- ImageNet 256×256 / 512×512で評価（クラス条件付き生成）
- FID=1.73, IS=350.2でDiT-XL/2, L-DiT-3B/7Bより良好
- 推論速度20倍、学習エポックも少ない（例: DiTは1400ep、VARは350ep）
- スケーリング則（log-logの線形関係）を12モデルで検証（Pearson相関0.998）
- in-painting, out-painting, class-conditional editingをゼロショットで達成

## 5. 議論はあるか？
- tokenizer（VQVAE）は既存のまま → 改良余地あり
- text-to-imageやvideo生成は未対応だが構造上は拡張可能
- 訓練時のteacher forcing依存のため、純粋な条件生成への一般化課題は残る可能性
- モデルの計算資源や収束安定性についてはさらなる調査必要

## 6. 次に読むべき論文はあるか？
- [VQGAN (Esser et al. 2021)]：本手法のtokenizerのベース
- [DiT (Peebles et al. 2022)]：比較対象の拡散トランスフォーマー
- [Henighan et al. 2020, Kaplan et al. 2020]：LLMのスケーリング則関連論文
- [MaskGIT (Chang et al. 2022)]：BERT型の画像生成モデルの代表

## 論文情報・リンク
Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, Liwei Wang, "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction," arXiv preprint, arXiv:2404.02905v2, 2024.
https://arxiv.org/abs/2404.02905
