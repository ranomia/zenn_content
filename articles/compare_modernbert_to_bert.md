---
title: ModernBERTとBERTを比較してみる(JcommonsenseQAタスク)
emoji: "⚖️"
type: "tech"
topics: ["ModernBERT", "BERT"]
published: false
---

# はじめに
LLMの話題が飛び交う中、昨年末からパラメータ数100M程度のModernBERTが騒がれている。
ModernBERTの性能を肌で感じておきたいと思い、JcommonsenseQAタスクに適用した。
比較対象としてBERT（東北大）を用いた。

# 問題設定
JcommonsenseQAは、一般常識に対する推論能力を評価するためのデータセットであり、
問題文と5つの選択肢、その解答がペアとして与えられる。

> question: 主に子ども向けのもので、イラストのついた物語が描かれているものはどれ？
> choice0: 世界
> choice1: 写真集
> choice2: 絵本
> choice3: 論文
> choice4: 図鑑
> label: 2

# 実験概要
- google colab (GPU: Tesla T4 (16GB))で実行。
- JcommonsenseQAでは、testデータセットが公開されていないため、validationデータセットを評価用として利用。trainデータセットのうち8割を学習用、2割を検証用として利用。
- BERT（東北大v3、111M）、ModernBERT（SB Intuitions、132M）の比較。
- その他学習条件
    - バッチサイズ: 16
    - 学習率: 1e-5
    - 混合精度学習: on

# 実験結果
## 予測性能
### BERT（東北大v3）
|  | precision | recall | f1-score | support |
| ---- | ---: | ---: | ---: | ---: |
| label 0 | 0.84 | 0.85 | 0.85 | 216 |
| label 1 | 0.84 | 0.77 | 0.81 | 237 |
| label 2 | 0.86 | 0.86 | 0.86 | 240 |
| label 3 | 0.83 | 0.79 | 0.81 | 228 |
| label 4 | 0.79 | 0.90 | 0.84 | 198 |
| macro avg | 0.83 | 0.83 | 0.83 | 1119 |
| weight avg | 0.83 | 0.83 | 0.83 | 1119 |

### ModernBERT（SB Intuitions）
|  | precision | recall | f1-score | support |
| ---- | ---: | ---: | ---: | ---: |
| label 0 | 0.92 | 0.92 | 0.92 | 216 |
| label 1 | 0.91 | 0.90 | 0.91 | 237 |
| label 2 | 0.92 | 0.92 | 0.92 | 240 |
| label 3 | 0.94 | 0.86 | 0.90 | 228 |
| label 4 | 0.87 | 0.96 | 0.91 | 198 |
| macro avg | 0.91 | 0.91 | 0.91 | 1119 |
| weight avg | 0.91 | 0.91 | 0.91 | 1119 |

## ファインチューニング時の計算時間
ただし、学習データはtrainデータセットのうち8割（7,151サンプル）、
推論データはtrianデータセットのうち2割（1,788サンプル）
|  | 学習 | 推論 |
| ---- | ---- | ---- |
| BERT | 3分10秒 / epoch | 1分0秒 / epoch |
| ModernBERT | 3分46秒 / epoch | 1分1秒 / epoch |

# まとめ
ModernBERTの性能が顕著に良い傾向が確認できた。
学習率やdropoutの割合など、各種条件は調整の余地あり。
