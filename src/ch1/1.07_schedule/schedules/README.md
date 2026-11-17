# 学习率调度：从 constant 到 WSD

优化器决定每一步的**方向**，以及各个参数之间步长的**相对大小**。学习率调度决定的是另一件事：**整体的步长怎样随训练进程变化**。

在 `train.py` 里，第 $t$ 步的学习率是峰值 $\eta_{\max}$ 乘以一个 0 到 1 之间的比例。这个比例由两部分相乘：

```python
def lr_mult(step):
    return min(1, (step + 1) / warmup if warmup else 1) * schedule(step, num_steps)
```

- 前一部分是 **warmup**：开头若干步从接近 0 线性升到 1。
- 后一部分是**衰减的形状**，定义在这个目录下，每种一个文件，每个只有一两行。

下面先讲为什么要衰减，再讲几种衰减形状，然后讲 WSD 为什么"随时可停"，最后讲 warmup。

记号：$t$ 是当前步数，$T$ 是总步数，$\eta_{\max}$ 是峰值学习率。

---

## 1. 为什么要衰减

### 1.1 constant：一直用同一个学习率

```python
def lr(step, num_steps):
    return 1.0
```

学习率太大和太小各有各的问题：

- **大学习率走得快，但停不下来。** 梯度是在一个 batch 上算的，带有噪声。即使已经到了谷底附近，每一步还是会被噪声踢一下。步子越大，被踢得越远，参数就在谷底附近一片区域里来回跳，loss 降不到底。这片区域的大小，大致和学习率成正比。
- **小学习率能停得很准，但走得很慢。** 离谷底还远的时候，就浪费了很多步。

所以只用一个固定的学习率，只能在"走得快"和"停得准"之间折中。我们的实验里，constant 的最佳学习率是 0.005，只有其他调度的一半，因为它只能靠小学习率来压低噪声。结果是 1.242，比衰减到 0 的调度差了 0.07。

**衰减**的想法很直接：先用大学习率快速前进，再逐渐减小学习率，让参数停稳。

### 1.2 河谷：衰减到底做了什么

Wen et al. (2024) 用"河谷"来描述语言模型的 loss 地形：

- 有一个方向像**河道**：坡度很缓，沿着它一直往下游走，loss 会持续降低。这对应模型真正在"学东西"。
- 其他方向像**两岸的山坡**：非常陡。

大学习率下，参数在两岸的山坡之间来回跳，同时被梯度推着往下游走。这时看到的 loss 主要是"跳在山坡上有多高"，它掩盖了沿河道走了多远。减小学习率以后，跳动的幅度变小，参数落到河底，之前沿河道取得的进展才在 loss 上显现出来。

我们的实验里能直接看到这一点。WSD 在前 80% 的步数里就是 constant：

- 学习率 0.01 的 constant 跑完 3072 步时是 1.271，再跑 1024 步只降到 1.257。
- 同样从这个点出发，WSD 在最后 20% 线性衰减到 0，降到了 1.171。

同样是最后的 1024 步，衰减多降了 0.085。这多出来的部分并不是这 1024 步里新学到的东西，而是把前面积累的进展"兑现"出来。

---

## 2. 衰减的形状

### 2.1 历史上的几种

- **step decay**：在固定的几个时间点把学习率乘以 0.1。CNN 时代的标准做法，比如 ResNet（He et al. 2016）在训练的 1/2 和 3/4 处各降一次。
- **inverse square root**：原始的 Transformer（Vaswani et al. 2017）先 warmup 4000 步，然后按 $1/\sqrt{t}$ 衰减。
- **cosine**：Loshchilov & Hutter (2017) 在 SGDR 里提出，后来成了大模型预训练的默认选择。
- **linear**：BERT 用的就是 warmup 加线性衰减。
- **WSD**：MiniCPM（Hu et al. 2024）推广开来，Hägele et al. (2024) 做了系统的研究。

这一节对比后三种，外加 constant 作为对照。

### 2.2 linear：线性降到 0

$$\eta_t = \eta_{\max}\left(1 - \frac{t}{T}\right)$$

```python
def lr(step, num_steps):
    return 1 - step / num_steps
```

1.01 到 1.06 一直用的就是它。

### 2.3 cosine：沿余弦曲线降到 0

$$\eta_t = \eta_{\max}\cdot \frac{1}{2}\left(1 + \cos\frac{\pi t}{T}\right)$$

```python
def lr(step, num_steps):
    return 0.5 * (1 + math.cos(math.pi * step / num_steps))
```

开头和结尾降得慢，中间降得快。和 linear 相比，它在前半段把学习率保持得更高，在最后又降得更平缓。

GPT-3 和 LLaMA 都是降到峰值的 10% 就停（$\eta_{\min} = 0.1\,\eta_{\max}$），没有降到 0。我们降到 0。Bergsma et al. (2025) 发现，降到 0 比降到 10% 更好，而且 linear 降到 0 的效果最好。

### 2.4 WSD：大部分时间不变，最后再降

WSD 是 warmup–stable–decay 的缩写：先 warmup，然后**保持峰值**，到最后一小段才衰减。

$$
\eta_t = \begin{cases}
\eta_{\max} & t < (1-f)\,T \\[4pt]
\eta_{\max}\cdot \dfrac{T - t}{f\,T} & t \ge (1-f)\,T
\end{cases}
$$

```python
decay_frac = 0.2

def lr(step, num_steps):
    decay_start = (1 - decay_frac) * num_steps
    return 1.0 if step < decay_start else (num_steps - step) / (num_steps - decay_start)
```

我们取 $f = 0.2$，即最后 20% 线性降到 0。MiniCPM 的衰减段约占 10%，用的是指数衰减。Hägele et al. (2024) 发现，衰减段占 20% 左右时 WSD 就能追平 cosine，而且衰减段用 $1-\sqrt{x}$ 的形状比线性更好。

用河谷的话说：stable 段一直用大学习率沿河道前进，decay 段再一次性落到河底。

### 2.5 我们的实验

linear、cosine、WSD 的最佳学习率都是 0.01，3 个种子的均值分别是 1.176、1.178、1.174，种子之间的范围互相重叠，**三者打平**。constant 明显更差（1.242）。

也就是说，**只要最后降到 0，衰减的具体形状影响不大**。这和 Hägele et al. (2024) 的结论一致。

---

## 3. WSD 为什么"随时可停"

cosine 和 linear 有一个实际上的麻烦：**每一步的学习率都取决于总步数 $T$**。

- 训练开始前就要决定训练多久。
- 中途想多训练一段，前面的学习率曲线就"不对"了。
- 想拿到不同训练长度的模型（比如做 scaling law 实验），每个长度都要从头训练一次。

Chinchilla（Hoffmann et al. 2022）专门检查过这一点：cosine 的周期如果比实际训练长度长 25% 以上，结果会明显变差。所以他们给每个训练长度单独训练了模型。

WSD 没有这个问题。stable 段的学习率不依赖 $T$，**所有训练长度共用同一条 constant 主干**。想要哪个长度的模型，就从主干对应的位置分出一个分支，单独跑一小段衰减即可。

```text
constant 主干 ──────────────────────────────────────────▶
          ├─ 衰减 ─▶ 1024 步的模型
                    ├─ 衰减 ─▶ 2048 步的模型
                                         ├─ 衰减 ─▶ 4096 步的模型
```

我们的实验（第三组）就是这样做的。三个长度的 WSD 在衰减之前和 constant 主干完全是同一条曲线；每个长度上，它都比从头训练的 cosine 好（1024 步：1.285 对 1.314；2048 步：1.216 对 1.228；4096 步：1.171 对 1.175）。总计算量也更少：WSD 一共约 4700 步，cosine 要 1024 + 2048 + 4096 = 7168 步。

这对大模型的意义：

- **scaling law 实验便宜很多。** Hägele et al. (2024) 指出，一次 constant 训练加上若干次短的衰减，就能代替许多次完整的 cosine 训练。
- **可以接着训练。** 数据多了、算力多了，就从主干上没有衰减的 checkpoint 继续往后跑。DeepSeek LLM（2024）用分段衰减的理由也是这个：第一阶段的训练可以复用，方便继续训练。
- **衰减段可以换数据。** 既然衰减是单独的一段，就可以在这一段换上更高质量的数据。MiniCPM 最早这样做，后来 Llama 3、OLMo 2 也都在最后的退火阶段使用高质量数据。

---

## 4. Warmup：开头慢慢升上去

$$\eta_t = \eta_{\max}\cdot \min\!\left(1,\ \frac{t+1}{W}\right)\cdot \text{schedule}(t)$$

```python
min(1, (step + 1) / warmup)
```

前 $W$ 步，学习率从接近 0 线性升到峰值，然后再交给衰减的形状。

### 4.1 为什么要 warmup

warmup 最早用于两种场景：Goyal et al. (2017) 用大 batch 训练 ResNet，以及原始的 Transformer。之后几乎成了标配。为什么训练开始时不能直接用峰值学习率，主要有三种解释：

1. **Adam 一开始对梯度大小的估计不可靠。** Adam 用 $\sqrt{\hat v}$ 归一化步长，但训练刚开始时 $v$ 只见过几个梯度。bias correction 修正了它的均值，但修正不了它的方差：只见过几个样本的估计，波动很大，步长也就忽大忽小。RAdam（Liu et al. 2019）认为 warmup 的作用正是绕开这一段，等 $v$ 估计准了再用大学习率。
2. **Post-LN 结构在初始化时梯度很大。** 原始的 Transformer 把 LayerNorm 放在残差相加之后（Post-LN）。Xiong et al. (2020) 证明，这种结构在初始化时，靠近输出的参数梯度很大，必须用 warmup。把 LayerNorm 移到每个子层之前（Pre-LN），梯度一开始就比较规整，可以不用 warmup。
3. **warmup 让参数先走到更平坦的地方。** Kalra & Barkeshli (2024) 发现，warmup 期间 loss 地形的陡峭程度（sharpness）会降下来，于是之后能承受更大的学习率。按这个解释，warmup 的主要好处是**让更大的峰值学习率变得可用**。

### 4.2 我们的实验：几乎没有作用

我们的模型是 Pre-LN（每个子层前先做 rmsnorm），优化器是带 bias correction 的 Adam。按解释 1 和解释 2，本来就不太需要 warmup。实验结果（第二组）也是这样：

- **最好的结果只好了 0.002**（warmup 512 步：1.169，不加：1.171），比种子之间的波动还小。
- **没有让更大的学习率变得可用。** 峰值 0.08 和 0.16 加了 warmup 以后不再在开头就发散，但学习率一升到 0.063 到 0.073 附近，就会发散，和 warmup 是 128 步还是 512 步无关。不加 warmup 时，0.04 在开头不发散，0.08 在第 11 步发散。所以开头能承受的学习率上限一直在 0.04 到 0.08 之间，warmup 没有把它推高。
- **0.04 在训练中途发散**（第 791 到 1457 步），这时 warmup 早就结束了。

第三点很有意思。0.04 在开头是稳定的，训练了上千步以后反而不稳定了。一个可能的解释是 **edge of stability**（Cohen et al. 2021, 2022）：训练过程中 loss 地形会逐渐变陡（progressive sharpening），一旦陡到当前学习率承受不了，训练就会不稳定。

这个解释和第一组的结果对得上：

- 学习率 0.04 时，constant 和 WSD 在同一步（第 1315 步）发散，因为那时 WSD 还处在 stable 段，和 constant 完全一样。
- linear 和 cosine 却都跑完了（1.219 和 1.222）。到第 1315 步时，它们的学习率已经降到了约 0.027 和 0.030。

也就是说，在我们这里，限制峰值学习率的是**训练中途的不稳定**，warmup 管不到这一段。我们没有实际测量 sharpness，所以这只是推测。Wortsman et al. (2023) 在小模型上复现了大模型训练中的几种不稳定，发现大学习率下注意力 logit 会越来越大，最终导致发散；他们发现 qk-layernorm 可以缓解这个问题。我们会在第二章讲架构时再看。

### 4.3 大模型为什么还是都加 warmup

我们的结论是"可以不加"，但几乎所有大模型都加了：

- GPT-3 warmup 3.75 亿 token；
- LLaMA 2000 步；
- DeepSeek-V3 2000 步；
- Llama 3 405B 8000 步；
- SmolLM3 2000 步。

原因有几个：大模型的 batch 和学习率都更大，又用 bf16 训练，训练开头更容易不稳定；一次训练的成本极高，一次发散的损失远大于几千步 warmup 的代价；而几千步只占一百多万步的不到 1%，几乎不影响最终结果。所以 warmup 更像是一份便宜的保险。

---

## 5. 大模型实际怎么用

| 模型 | warmup | 衰减 | 最低学习率 |
|---|---|---|---|
| GPT-3 (2020) | 3.75 亿 token | cosine，持续 2600 亿 token | 峰值的 10% |
| Chinchilla (2022) | 有 | cosine，周期等于训练长度 | 峰值的 10% |
| LLaMA (2023) | 2000 步 | cosine | 峰值的 10% |
| DeepSeek LLM (2024) | 2000 步 | 分段：80% 处降到 31.6%，90% 处降到 10% | 峰值的 10% |
| MiniCPM (2024) | 有 | WSD，衰减段约 10%，指数衰减，衰减段混入高质量数据 | 很小 |
| Llama 3 405B (2024) | 8000 步 | cosine，120 万步；最后 4000 万 token 线性退火到 0，同时提高高质量数据的比例 | 0 |
| DeepSeek-V3 (2024) | 2000 步 | 保持 2.2e-4 到 10T token；cosine 4.3T token 降到 2.2e-5；最后两段常数 2.2e-5 和 7.3e-6 | 7.3e-6 |
| OLMo 2 (2024) | 有 | 预训练之后单独一个 mid-training 阶段，用 Dolmino 高质量数据，学习率线性降到 0 | 0 |
| SmolLM3 (2025) | 2000 步 | WSD，最后 10% 衰减 | — |

可以看到一个趋势：

- **早期（GPT-3、LLaMA）：** cosine 降到 10%，训练前就定好总长度。
- **之后：** 越来越多的模型把训练拆成"长时间保持高学习率"和"最后一段衰减"两部分，也就是 WSD 的思路。DeepSeek-V3 的调度虽然名义上包含 cosine，但实质上就是这个思路：前 10T token 学习率不变，之后才开始降。
- **衰减段和数据配合使用。** Llama 3、OLMo 2、MiniCPM 都在学习率降下来的最后阶段使用最高质量的数据。

也有不同的声音。Yano et al. (2026) 发现，如果预训练之后还要做 SFT，预训练阶段不衰减（只做 warmup 和 stable，即 WSO），SFT 之后的效果反而更好（1B 和 8B 模型）：衰减让模型落进更陡的极小值，不衰减则停留在更平坦的地方，更容易适应下游任务。

---

## 6. 其他方向（简述）

- **Schedule-Free**（Defazio et al. 2024）：不使用学习率调度，而是对参数的历史做平均来达到衰减的效果，不需要事先知道训练长度。Schedule-Free AdamW 赢得了 AlgoPerf 的自调参赛道。
- **WSD 的变体**：Wen et al. (2024) 提出 WSD-S，在主干上多次衰减、再升回去继续训练，进一步减少多个长度所需的计算量。

---

## 汇总

| 调度 | 提出 | 一句话 | 我们的最佳 lr | 我们的 val bpb（3 个种子） | 在大模型中 |
|---|---|---|---|---|---|
| constant | — | 学习率不变 | 0.005 | 1.242 | 不单独使用，是 WSD 的 stable 段 |
| linear | BERT (2018) | 线性降到 0 | 0.01 | 1.176 | 降到 0 时效果很好（Bergsma et al. 2025） |
| cosine | SGDR (2017) | 余弦降到 0 | 0.01 | 1.178 | **早期默认**：GPT-3、Chinchilla、LLaMA、Llama 3 |
| WSD | MiniCPM (2024) | 保持峰值，最后一段才降 | 0.01 | **1.174** | **越来越普遍**：MiniCPM、SmolLM3，DeepSeek-V3 的思路也相同 |
| + warmup | Goyal et al. (2017) | 开头线性升到峰值 | — | 几乎无影响（−0.002） | **几乎都加**，几千步，作为保险 |

在我们的实验里，最终都降到 0 的三种形状打平，constant 明显更差；WSD 不需要事先定下训练长度，三个长度都比 cosine 好；warmup 在 Pre-LN + Adam 的小模型上几乎没有作用，限制峰值学习率的是训练中途的不稳定。

## 参考

- He et al. 2016, [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- Vaswani et al. 2017, [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Loshchilov & Hutter 2017, [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)
- Devlin et al. 2018, [BERT](https://arxiv.org/abs/1810.04805)
- Goyal et al. 2017, [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)
- Liu et al. 2019, [On the Variance of the Adaptive Learning Rate and Beyond (RAdam)](https://arxiv.org/abs/1908.03265)
- Xiong et al. 2020, [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
- Kalra & Barkeshli 2024, [Why Warmup the Learning Rate? Underlying Mechanisms and Improvements](https://arxiv.org/abs/2406.09405)
- Cohen et al. 2021, [Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability](https://arxiv.org/abs/2103.00065)
- Cohen et al. 2022, [Adaptive Gradient Methods at the Edge of Stability](https://arxiv.org/abs/2207.14484)
- Wortsman et al. 2023, [Small-scale proxies for large-scale Transformer training instabilities](https://arxiv.org/abs/2309.14322)
- Brown et al. 2020, [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)，附录 B
- Hoffmann et al. 2022, [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)
- Touvron et al. 2023, [LLaMA](https://arxiv.org/abs/2302.13971)，2.3 节
- DeepSeek-AI 2024, [DeepSeek LLM: Scaling Open-Source Language Models with Longtermism](https://arxiv.org/abs/2401.02954)
- Hu et al. 2024, [MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies](https://arxiv.org/abs/2404.06395)
- Hägele et al. 2024, [Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations](https://arxiv.org/abs/2405.18392)
- Wen et al. 2024, [Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective](https://arxiv.org/abs/2410.05192)
- Llama Team 2024, [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)，3.4 节
- DeepSeek-AI 2024, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)，4.2 节
- OLMo Team 2024, [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656)
- Hugging Face 2025, [SmolLM3: smol, multilingual, long-context reasoner](https://huggingface.co/blog/smollm3)
- Bergsma et al. 2025, [Straight to Zero: Why Linearly Decaying the Learning Rate to Zero Works Best for LLMs](https://arxiv.org/abs/2502.15938)
- Defazio et al. 2024, [The Road Less Scheduled (Schedule-Free)](https://arxiv.org/abs/2405.15682)
- Yano et al. 2026, [Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning](https://arxiv.org/abs/2603.16127)
