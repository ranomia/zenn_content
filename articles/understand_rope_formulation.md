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
\langle f_q (\bm{x}_m, m), f_k(\bm{x}_n, n) \rangle = g(\bm{x}_m, \bm{x}_n, m-n).
$$

$$
f_q(\bm{x}_m, m) = (\bm{W}_q \bm{x}_m) e^{i m \theta} \\
f_k(\bm{x}_n, n) = (\bm{W}_k \bm{x}_n) e^{i n \theta} \\
g(\bm{x}_m, \bm{x}_n, m-n) =\mathrm{Re} [(\bm{W}_q \bm{x}_m) (\bm{W}_k \bm{x}_n)^* e^{i (m-n) \theta}] 
$$

$$
f_{\{q,k\}} (\bm{x}_m, m) =
\begin{pmatrix}
\cos m \theta & - \sin m \theta \\
\sin m \theta & \cos m \theta
\end{pmatrix}

\begin{pmatrix}
W_{\{q,k\}}^{(11)} & W_{\{q,k\}}^{(12)} \\
W_{\{q,k\}}^{(21)} & W_{\{q,k\}}^{(22)}
\end{pmatrix}

\begin{pmatrix}
x_m^{(1)} \\
x_m^{(2)}
\end{pmatrix}
$$

$$
f_{\{q,k\}} (\bm{x}_m, m) = \bm{R}_{\Theta, m}^{d} \bm{W}_{\{q,k\}} \bm{x}_m
$$

$$
\bm{R}_{\Theta, m}^{d} = 
\begin{pmatrix}
\cos m \theta_1 & - \sin m \theta_1 & 0 & 0 & \cdots & 0 & 0 \\
\sin m \theta_1 & \cos m \theta_1 & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos m \theta_2 & - \sin m \theta_2 & \cdots & 0 & 0 \\
0 & 0 & \sin m \theta_2 & \cos m \theta_2 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos m \theta_{d/2} & - \sin m \theta_{d/2} \\
0 & 0 & 0 & 0 & \cdots & \sin m \theta_{d/2} & \cos m \theta_{d/2}
\end{pmatrix}
$$

$$
\bm{q}_m^\mathsf{T} \bm{k}_n = (\bm{R}_{\Theta, m}^{d} \bm{W}_q \bm{x}_m)^\mathsf{T} (\bm{R}_{\Theta, n}^{d} \bm{W}_k \bm{x}_n) = \bm{x}^\mathsf{T} \bm{W}_q \bm{R}_{\Theta, n-m}^{d} \bm{W}_k \bm{x}_n
$$

$$
\mathrm{Attention} (\bm{Q}, \bm{K}, \bm{V})_m = 
\frac{\Sigma_{n=1}^{N} \mathrm{sim} (\bm{q}_m, \bm{k}_n) \bm{v}_n}{\Sigma_{n=1}^{N} \mathrm{sim} (\bm{q}_m, \bm{k}_n)} \\
\mathrm{where} \mathrm{sim} (\bm{q}_m, \bm{k}_n) = \mathrm{exp} (\bm{q}_{m}^{\mathsf{T}} \bm{k}_n / \sqrt{d})
$$

$$
\mathrm{Attention} (\bm{Q}, \bm{K}, \bm{V})_m = 
\frac{\Sigma_{n=1}^{N} \phi (\bm{q}_m)^{\mathsf{T}} \varphi (\bm{k}_n) \bm{v}_n}{\Sigma_{n=1}^{N} \phi (\bm{q}_m)^{\mathsf{T}} \varphi (\bm{k}_n)}
$$

$$
\mathrm{Attention} (\bm{Q}, \bm{K}, \bm{V})_m = 
\frac{\Sigma_{n=1}^{N} (\bm{R}_{\Theta, m}^{d} \phi (\bm{q}_m))^\mathsf{T} (\bm{R}_{\Theta, n}^{d} \varphi (\bm{k}_n)) \bm{v}_n}{\Sigma_{n=1}^{N} \phi (\bm{q}_m)^{\mathsf{T}} \varphi (\bm{k}_n)}
$$