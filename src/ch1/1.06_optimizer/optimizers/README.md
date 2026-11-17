# 优化器：从 SGD 到 Muon

反向传播算出梯度 $g$，告诉我们"往哪个方向改参数，loss 下降得最快"。但梯度只是方向，**每一步走多远、各个参数怎么区别对待、历史上的梯度要不要用**，这些都由优化器决定。

这里的 6 个优化器，基本是按时间顺序排的。每一个都在解决前一个留下的问题。代码就是这个目录下的文件，每个都只有十几行，接口统一：`Optimizer(params, lr)` 创建，`step(decay)` 更新一步（`decay` 是学习率调度给出的比例）。

记号：$\theta$ 是参数，$g_t$ 是第 $t$ 步的梯度，$\eta$ 是学习率。

---

## 1. SGD：沿着梯度往下走

$$\theta \leftarrow \theta - \eta\, g_t$$

```python
p -= lr * p.grad
```

最直接的想法：梯度指向上坡最陡的方向，那就往反方向走一小步。"随机"（stochastic）指梯度不是在全部数据上算的，而是在一个 batch 上算的，所以带噪声。

**问题：所有参数共用一个学习率。** 想象 loss 是一条狭长的山谷：横向很陡，纵向很平。

- 学习率要照顾最陡的方向，否则横向会越跳越高，直接发散。
- 但这样一来，平缓的纵向就走得极慢，参数在山谷两壁之间来回震荡，沿谷底前进得很慢。

所以 SGD 的学习率被"最陡的那个方向"卡住了。我们的实验里正是这样：学习率越大越好，一直到 2；再加到 2.83 就在第 12 步发散了。最好的学习率就在悬崖边上。

**在大模型中：** 训练 Transformer 基本不用。Zhang et al. (2020) 发现，BERT 这类注意力模型的梯度噪声是重尾分布（偶尔出现特别大的梯度），而 ResNet 在 ImageNet 上的梯度噪声比较集中。SGD 在后者上表现很好，在前者上明显不如 Adam。Zhao et al. (2024) 在 150M 到 600M 的语言模型上对比了多种优化器：除了 SGD，其他几种调好以后表现都差不多。

---

## 2. Momentum：让方向攒起来

$$m \leftarrow \beta\, m + g_t, \qquad \theta \leftarrow \theta - \eta\, m$$

```python
buf.mul_(beta).add_(p.grad)
p -= lr * buf
```

不直接用当前的梯度，而是用梯度的滑动累积 $m$（动量）。像一个球滚下山：

- 在来回震荡的方向上，梯度的正负交替，累加起来互相抵消。
- 在一直朝同一个方向的谷底，梯度不断叠加，越滚越快。如果梯度一直不变，$m$ 最终会是 $g/(1-\beta)$，即 $\beta=0.9$ 时步长放大 10 倍。

我们的实验里，momentum 的最佳学习率是 0.707，等效步长约 $0.707/(1-0.9) \approx 7$，是 SGD 发散边界（2 和 2.83 之间）的好几倍。震荡被抵消掉了，所以能在平缓的方向上迈更大的步子而不发散。结果从 1.319 降到 1.217，提升很大。

**还剩下的问题：** 不同参数的梯度大小可以差几个数量级。比如一个很少出现的 token，它的 embedding 只有在 batch 里出现时才有梯度。全局只有一个学习率，对梯度大的参数刚好，对梯度小的参数就太小了。

**在大模型中：** SGD + momentum 是 CNN 时代的主力（例如 ResNet 在 ImageNet 上），但在 Transformer 预训练中基本不用，原因同上。

---

## 3. Adam：每个参数有自己的步长

$$
\begin{aligned}
m &\leftarrow \beta_1 m + (1-\beta_1)\, g_t &&\text{（梯度的平均：往哪走）}\\
v &\leftarrow \beta_2 v + (1-\beta_2)\, g_t^2 &&\text{（梯度平方的平均：梯度一般有多大）}\\
\hat m &= \frac{m}{1-\beta_1^t},\quad \hat v = \frac{v}{1-\beta_2^t} &&\text{（修正一开始偏向 0）}\\
\theta &\leftarrow \theta - \eta\, \frac{\hat m}{\sqrt{\hat v} + \epsilon}
\end{aligned}
$$

```python
m.mul_(beta1).add_((1 - beta1) * p.grad)
v.mul_(beta2).add_((1 - beta2) * p.grad ** 2)
m_hat = m / (1 - beta1 ** self.t)
v_hat = v / (1 - beta2 ** self.t)
p -= lr * m_hat / (v_hat ** 0.5 + eps)
```

关键在于除以 $\sqrt{\hat v}$：**每个参数都用自己梯度的典型大小去归一化**。梯度大的参数，步子不会过大；梯度小的参数，步子也不会过小。最后每个参数每步大约都走 $\eta$ 这么远。学习率的含义从"梯度乘多少"变成了"每一步大约改多少"，所以它对梯度的尺度不再敏感。

**bias correction** 是做什么的：$m$ 和 $v$ 从 0 开始，前几步的估计会严重偏小。除以 $1-\beta^t$ 正好把它修正回来（$t$ 大了以后，这一项趋近于 1，就不起作用了）。

我们的实验里，Adam 的学习率–loss 曲线有一个明显的谷底：从 0.00125 到 0.04（跨 32 倍）都不发散，0.01 最好（1.176）。这就是常说的"Adam 好调"。

**代价：** 每个参数要额外存 $m$ 和 $v$ 两份状态，优化器的状态是参数量的 2 倍。大模型里，这些状态通常用 fp32 存储，每个参数要 8 字节，比 bf16 的参数本身（2 字节）还大好几倍。这也是 ZeRO 这类"把优化器状态切分到多张卡上"的技术出现的原因之一。

**在大模型中：** Adam 和它的变体 AdamW 是大模型训练的标准选择。GPT-3 用的是 Adam（$\beta_1=0.9,\ \beta_2=0.95,\ \epsilon=10^{-8}$，配合 0.1 的 weight decay）。我们沿用 microgpt 的 $\beta_1=0.85,\ \beta_2=0.99$；大模型常用 $\beta_2=0.95$，对梯度大小的估计更新得更快，训练更稳。

---

## 4. AdamW：把 weight decay 和梯度分开

weight decay 的本意是：每一步把所有参数按比例往 0 拉一点，防止参数无限变大。在 SGD 里，它和"在 loss 里加一项 $\frac{\lambda}{2}\lVert\theta\rVert^2$"（L2 正则）完全等价，因为后者的梯度正好是 $\lambda\theta$。

但在 Adam 里不再等价：如果把 $\lambda\theta$ 加进梯度，它会和梯度一起被 $\sqrt{\hat v}$ 除掉。结果是，梯度大的参数反而被拉得少，这不是我们想要的。Loshchilov & Hutter (2019) 的修正很简单：**把 weight decay 单独拿出来，不经过梯度**。

$$\theta \leftarrow \theta - \eta\lambda\,\theta - \eta\, \frac{\hat m}{\sqrt{\hat v} + \epsilon}$$

```python
p.mul_(1 - lr * weight_decay) # 和 Adam 唯一的区别
```

我们的实验里，AdamW（$\lambda=0.1$）比 Adam 略差（1.183 对 1.176），而且这个差距大于种子之间的波动。原因：

- 模型小、训练短，模型还处在"学不够"的阶段，不需要正则。
- 我们的 weight decay 其实不弱：学习率线性衰减，4096 步的 $\eta$ 加起来约 $0.01\times 2048$，乘以 $\lambda=0.1$，相当于如果没有梯度，参数会被拉到原来的 $e^{-2} \approx 13\%$。

**在大模型中：** AdamW 是事实上的默认选择。LLaMA 用的是 AdamW（$\beta_1=0.9,\ \beta_2=0.95$，weight decay 0.1，梯度裁剪 1.0），DeepSeek-V3 也是同样的设置。Chinchilla 的附录专门比过：Gopher 用 Adam，Chinchilla 改用 AdamW，在 417M 和 1.4B 的模型上 AdamW 都更好。在大模型里，weight decay 不只是正则：Moonshot (2025) 发现，如果不加 weight decay，长时间训练后权重会越来越大，超出 bf16 能精确表示的范围。

---

## 5. Lion：只看方向，不看大小

$$
\begin{aligned}
\theta &\leftarrow \theta - \eta\cdot \mathrm{sign}\big(\beta_1 m + (1-\beta_1)\, g_t\big)\\
m &\leftarrow \beta_2 m + (1-\beta_2)\, g_t
\end{aligned}
$$

```python
p -= lr * (beta1 * m + (1 - beta1) * p.grad).sign()
m.mul_(beta2).add_((1 - beta2) * p.grad)
```

Lion 是 Google (Chen et al. 2023) 用程序搜索"找"出来的。它把 Adam 的思路推到极端：Adam 用 $\hat m/\sqrt{\hat v}$ 来归一化，Lion 干脆只取符号。**每个参数每一步都恰好走 $\pm\eta$**。

- **好处：** 只需要一份状态 $m$，显存是 Adam 的一半；更新规则也非常简单。
- **代价：** 每个参数每步都走满 $\eta$，更新的总幅度比 Adam 大，所以学习率要小 3 到 10 倍。原论文建议同时把 weight decay 调大 3 到 10 倍，以保持 $\eta\lambda$ 不变。我们的最佳学习率是 0.00125，正好是 Adam 的 1/8。
- **问题：** 完全丢掉了梯度大小的信息。哪怕某个参数的梯度只是噪声，它也要走满一步。我们的实验里，Lion 在学习率 0.0025 和 0.00177 时都是训练到中途（第 776 步、第 3403 步）才发散，不是一开始就发散。一个可能的原因：我们没有加 weight decay，参数每步都在走固定大小的一步，可能会慢慢变大，直到不稳定。

**在大模型中：** 独立评测普遍认为 Lion 和 AdamW 差不多。Wen et al. (2025) 发现它最多比 AdamW 慢 5%，而且模型越大，劣势越明显；Zhao et al. (2024) 也认为它和 Adam 表现相当。在我们查到的主流大模型技术报告里，没有看到用 Lion 做预训练的。

---

## 6. Muon：把权重当作矩阵来更新

前面的优化器都把参数看成一堆**独立的数**，每个数各更新各的。但 Transformer 里的大部分参数是矩阵 $W$，它的作用是把一个向量映射到另一个向量。Muon (Keller Jordan, 2024) 从这个角度出发。

**观察：** 用 SGD-momentum 或 Adam 训练 Transformer 时，矩阵参数的更新往往接近**低秩**：少数几个方向占了绝大部分幅度，其他方向几乎不动。

**想法：** 把动量矩阵 $M$ 做奇异值分解 $M = U\Sigma V^\top$，再把所有奇异值都换成 1，得到 $UV^\top$。方向保留下来，但每个方向的幅度都一样，原本被压制的方向也能学起来。这叫**正交化**。

$$\theta \leftarrow \theta - \eta \cdot \sqrt{\max\!\big(1, \tfrac{\text{行数}}{\text{列数}}\big)}\cdot \mathrm{orth}(M), \qquad \mathrm{orth}(U\Sigma V^\top) = UV^\top$$

**怎么算得快：** SVD 在 GPU 上很慢。Muon 用 Newton–Schulz 迭代来近似，只用矩阵乘法：

$$X \leftarrow aX + b\,(XX^\top)X + c\,(XX^\top)^2X, \qquad (a,b,c) = (3.4445,\ -4.7750,\ 2.0315)$$

```python
X = G / (G.norm() + 1e-7)  # 先归一化，让所有奇异值都不超过 1
for _ in range(steps):     # 5 次
    A = X @ X.T
    X = a * X + (b * A + c * A @ A) @ X
```

这个迭代对每个奇异值 $\sigma$ 做同样的事：$\sigma \leftarrow a\sigma + b\sigma^3 + c\sigma^5$，反复几次就把 $\sigma$ 推到 1 附近。这组系数是调出来的，目的是在 5 步内尽快推过去，代价是不精确等于 1。我们实测，迭代后的奇异值落在 0.68 到 1.13 之间。

**只用在 Transformer 内部的矩阵上**（q/k/v/o、MLP）。其余参数仍然用 Adam(W)：
- embedding 是查表，每一行只在对应的 token 出现时才更新，不是普通的矩阵乘法；
- 向量参数（比如 norm 的权重）不是矩阵，谈不上正交化；
- lm_head 虽然是矩阵乘法，但原作者的实验发现，它和 embedding 一样用 AdamW 效果更好。

**我们的实验里：**
- 按步数算，Muon 最好（1.163），比 Adam 好 0.013，大于种子之间的波动。
- 学习率的谷底也最宽：0.01 到 0.04（跨 4 倍）的结果都在最佳值的 0.02 以内，Adam 只有 0.01 到 0.02。
- 但每步慢了 21%（40 秒对 33 秒）。按时间算，两者基本打平。

慢的原因和 [1.04 的分析](../../1.04_cuda/analysis/notes.md)一样：我们的矩阵只有 48×48 这么大，5 次迭代的计算量几乎可以忽略，时间都花在启动几十个额外的 kernel 上。原作者估算，大模型里 Muon 额外的计算量不到 1%。另外，在我们这个模型里，Muon 只管 2.8 万个参数，占 20%；其余的是词表和位置编码，仍然用 Adam。

**在大模型中：** Muon 正在被前沿实验室采用。
- **NanoGPT speedrun**（2024 年 10 月）：换成 Muon 后，训练速度纪录提升约 35%，此后一直是 speedrun 的标准选择。
- **Moonshot, *Muon is Scalable* (2025)**：要把 Muon 用到大模型上，需要两个改动。一是加上 weight decay；二是按 $0.2\sqrt{\max(A,B)}$ 缩放更新，让更新的幅度和 AdamW 一致，这样可以直接沿用 AdamW 的学习率和 weight decay。他们用 Muon 训练了 16B 的 MoE 模型 Moonlight（5.7T token），scaling law 实验显示，计算效率约是 AdamW 的 2 倍。
- **Kimi K2**（1T 参数，15.5T token）：用的是 MuonClip，也就是 Muon 加上 QK-Clip。QK-Clip 在注意力 logit 过大时缩小 $W_q$ 和 $W_k$，报告称整个预训练没有出现一次 loss spike。
- **GLM-4.5**：除了 word embedding、bias 和 RMSNorm 的权重，其余参数都用 Muon，理由是"收敛更快，能用更大的 batch"。
- **DeepSeek-V4**（2026）：同样换成了 Muon，理由是"收敛更快，训练更稳定"。
- **但独立评测更保守。** Wen et al. (2025) 把 AdamW 认真调好以后再比，发现 Muon 这类矩阵优化器的加速比从 0.1B 模型的 1.4 倍降到 1.2B 模型的 1.1 倍；数据量远超 Chinchilla 比例时，SOAP 等方法还会反超 Muon。很多论文宣称的 2 倍提升，有一部分来自没调好的 AdamW 基线：光调 GPT-3 配方里的学习率，就可能快 2 倍。

---

## 7. 其他方向（简述）

- **Adafactor**（Shazeer & Stern 2018）：把 Adam 的 $v$ 分解成"每行一个数 + 每列一个数"，显存大大减少。T5 和 PaLM 都用过。
- **Shampoo / SOAP**：给每个矩阵左右各配一个预条件矩阵，近似二阶方法。Distributed Shampoo 赢得了 AlgoPerf 第一届比赛的外部调参赛道（Kasimbeg et al. 2025）。SOAP 相当于在 Shampoo 的特征基下跑 Adam。
- **Schedule-Free AdamW**：不需要学习率调度。赢得了 AlgoPerf 的自调参赛道。和 1.07 的学习率调度有关。

---

## 汇总

| 优化器 | 提出 | 一句话 | 额外状态 | 我们的最佳 lr | 我们的 val bpb（3 个种子） | 在大模型中 |
|---|---|---|---|---|---|---|
| SGD | 1951 | 沿梯度反方向走 | 无 | 2（紧挨发散点） | 1.319 | Transformer 预训练基本不用 |
| Momentum | 1964 | 梯度累积成动量，抵消震荡 | 1× | 0.707（紧挨发散点） | 1.217 | CNN 时代的主力，LLM 不用 |
| Adam | 2014 | 每个参数按自己的梯度大小归一化 | 2× | 0.01 | 1.176 | 标准选择（如 GPT-3） |
| AdamW | 2017 | Adam + 解耦的 weight decay | 2× | 0.01 | 1.183 | **事实上的默认**：LLaMA、Chinchilla、DeepSeek-V3 |
| Lion | 2023 | 只取动量的符号 | 1× | 0.00125 | 1.226 | 独立评测与 AdamW 相当或略差，少见于大模型 |
| Muon | 2024 | 把矩阵的更新正交化 | 1×（矩阵）+ Adam（其余） | 0.02 | **1.163** | **正在被采用**：Kimi K2、GLM-4.5、DeepSeek-V4；独立评测加速 1.1～1.4 倍 |

我们的实验和文献大体一致：SGD 明显最差，Adam 系列好调，Muon 按步数算最好。区别在于规模：我们的模型小、训练短，所以 weight decay 没有帮助；Muon 额外的开销在这里显得很大，到大模型里可以忽略。

## 参考

- Zhang et al. 2020, [Why are Adaptive Methods Good for Attention Models?](https://proceedings.neurips.cc/paper/2020/hash/b05b57f6add810d3b7490866d74c0053-Abstract.html)
- Kingma & Ba 2014, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- Loshchilov & Hutter 2019, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- Brown et al. 2020, [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)，附录 B
- Touvron et al. 2023, [LLaMA](https://arxiv.org/abs/2302.13971)，2.3 节
- Hoffmann et al. 2022, [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)，附录图 A6、A7
- DeepSeek-AI 2024, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)，4.2 节
- Chen et al. 2023, [Symbolic Discovery of Optimization Algorithms (Lion)](https://arxiv.org/abs/2302.06675)
- Keller Jordan 2024, [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/)
- Liu et al. (Moonshot) 2025, [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)
- Kimi Team 2025, [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534)
- GLM-4.5 Team 2025, [GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models](https://arxiv.org/abs/2508.06471)，2.4 节
- DeepSeek-AI 2026, [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348)
- Wen et al. 2025, [Fantastic Pretraining Optimizers and Where to Find Them](https://arxiv.org/abs/2509.02046)
- Zhao et al. 2024, [Deconstructing What Makes a Good Optimizer for Language Models](https://arxiv.org/abs/2407.07972)
- Kasimbeg et al. 2025, [Accelerating Neural Network Training: An Analysis of the AlgoPerf Competition](https://arxiv.org/abs/2502.15015)
