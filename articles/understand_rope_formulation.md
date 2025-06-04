---
title: RoPE(Rotary Positional Encoding)の数式を理解する
emoji: "😵‍💫"
type: "tech"
topics: ["LLM", "RoPE", "Positional Encoding", "Encoding"]
published: False
---

# はじめに
RoPE(Rotary Positional Encoding)はLLMにおけるPositional Encodingの一種であり、
Su [2021]によって提案された。
従来のPositional Encodingは、位置情報をベクトルに加算することで、
位置情報をモデルに与える方法であった。
一方のRoPEは、文脈表現に直接位置を追加するのではなく、正弦波関数を乗算することで相対位置情報を取り込むことを提案している。

# RoPEの数式
$$
\langle f_q (\bm{x}_m) \rangle
$$