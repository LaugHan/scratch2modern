# 训练细节：初始化、weight decay、梯度裁剪、dropout

打开任何一个大模型的训练配置，除了优化器、学习率、精度这些大的选择，总会有一组"小参数"：初始化的标准差、weight decay、梯度裁剪的上限、dropout 的概率。这一节一项一项地看：**每一项要解决什么问题？在我们的模型上有没有用？**

每一项都从最简单的做法讲起，一步一步说清楚"上一步为什么不够"，每一步接上我们的实验结果。

`train.py` 用开关来选，每次只改一项，其余保持默认：

```bash
python train.py init=xavier lr=0.005   # 初始化：zero、xavier、kaiming、0.02、0.08（默认）、zero_proj
python train.py wd=adamw               # weight decay：none（默认）、l2、adamw、adamw_noemb
python train.py clip=1.0 lr=0.04       # 梯度裁剪：none（默认），或者上限
python train.py dropout=0.1late        # dropout：0（默认）、0.1、0.1late
```

---

## 1. 初始化：训练开始时，参数取什么值

### 1.0 全部设成 0？

最直接的想法是全部从 0 开始。但这样训练不起来，原因是**对称性**：

- 同一层的所有神经元，初始值一样，算出来的输出一样，反向拿到的梯度也一样，更新以后还是一样。不管训练多久，它们都像同一个神经元。
- 在我们的模型里更糟：embedding 是 0，经过 rmsnorm 还是 0，后面每一层的输入都是 0，**所有梯度都是 0，参数一步都动不了**。

**实验：** 全零初始化训练 4096 步，val bpb 是 3.16，从头到尾没有变过。3.16 正好是"对 1024 个 token 均匀地瞎猜"：每个 token 要 $\log_2 1024 = 10$ 比特，平均每个 token 3.16 个字节，10 / 3.16 ≈ 3.16。

所以至少要有随机性来打破对称。最自然的选择是正态分布 $N(0, \sigma^2)$。问题变成：**σ 取多大？**

### 1.1 按维度推导 σ：Xavier 和 Kaiming

先看一个线性层 $y = Wx$。输出的每一个元素，是 $\text{fan\_in}$ 个乘积加起来的（**fan_in** 就是输入的维度，**fan_out** 是输出的维度）。如果 $x$ 的每个元素方差是 1，$W$ 的每个元素方差是 $\sigma^2$，那么：

$$\text{Var}(y_i) = \text{fan\_in} \cdot \sigma^2$$

- σ 太大，每过一层，方差就放大一次，层数一多就爆炸；
- σ 太小，每过一层就缩小一次，信号和梯度都消失。

所以要让 $\text{fan\_in} \cdot \sigma^2 \approx 1$。不同的工作关心的东西不一样：

| 做法 | 关心什么 | σ |
|---|---|---|
| **Xavier**（Glorot & Bengio 2010） | 前向时激活的方差不变（要求 $\sigma^2 = 1/\text{fan\_in}$），反向时梯度的方差也不变（要求 $\sigma^2 = 1/\text{fan\_out}$），两者取折中 | $\sqrt{2/(\text{fan\_in}+\text{fan\_out})}$ |
| **Kaiming**（He et al. 2015） | 只看前向。ReLU 会把一半的值变成 0，方差减半，所以要乘 2 补回来 | $\sqrt{2/\text{fan\_in}}$ |

```python
if init == 'xavier':
    return math.sqrt(2 / (nin + nout))
if init == 'kaiming':
    return math.sqrt(2 / nin)
```

在我们的模型里，48×48 的矩阵，Xavier 是 0.14，Kaiming 是 0.20。

### 1.2 固定一个常数：0.02

GPT-1 没有用 Xavier 或 Kaiming，而是直接用了一个固定的常数。原文是：

> Since layernorm is used extensively throughout the model, a simple weight initialization of N(0, 0.02) was sufficient.

**模型里到处都有 layernorm，所以简单的 N(0, 0.02) 就够了。** 1.1 节的推导，是为了让信号逐层不变大也不变小；而 layernorm 每一层都会把信号重新拉回固定的大小，这件事就不再那么要紧。

这个 0.02 从此被一路沿用：GPT-2、nanoGPT、NVIDIA 的大模型训练框架 Megatron-LM 的默认值、OLMo 2，都是 0.02，**不随模型宽度变化**。它和 Xavier 只是数值上碰巧接近：对于宽度约 2500 的矩阵，Xavier 正好算出 0.02。也有按宽度来定的，比如 DeepSeek-V3（宽度 7168）用的是 0.006。Qwen、Llama 的技术报告没有写初始化。

我们沿用 microgpt 的 0.08，也属于"固定常数"这一类。

**实验：** 每种初始化各扫 3 个学习率（0.005、0.01、0.02），在最好的那个上跑 3 个种子。

| 做法 | σ（48×48 的矩阵） | 最佳学习率 | val bpb | 种子范围 |
|---|---|---|---|---|
| Xavier | 0.14 | 0.01 | 1.1766 | 1.1723 ~ 1.1802 |
| Kaiming | 0.20 | **0.02** | **1.1864** | 1.1828 ~ 1.1883 |
| 0.02 | 0.02 | 0.01 | 1.1755 | 1.1714 ~ 1.1787 |
| 0.08（当前） | 0.08 | 0.01 | 1.1751 | 1.1740 ~ 1.1763 |

- **Xavier、0.02、0.08 三者打平，最佳学习率也都是 0.01。** σ 相差 7 倍，结果却一样，这正是 GPT-1 那句话的意思：有归一化以后，σ 的精确取值就不那么要紧了。
- **Kaiming 差了 0.011，而且需要更大的学习率。** 它的 σ 最大。初始时的 val bpb 是 4.85，比均匀瞎猜（3.16）还差：lm_head 太大，一开始的预测就"很自信地猜错"，要先花一段时间把这些错误的自信纠正回来。另外，Adam 每步大约走固定的距离，参数越大，要走更多步才能改变到同样的程度，所以它偏好更大的学习率。

### 1.3 不同部分不同处理：残差分支的输出层

1.1 节的推导默认"每一层是一个接一个串起来的"。但 Transformer 有**残差连接**：每一层的输出是**加回**到主干上的（`x = x + attn(x)`），层数越多，加的次数越多，主干上的值会越积越大。

所以有一类做法，单独处理残差分支的**输出层**（我们的 `attn_wo` 和 `mlp_fc2`）：

- **缩小**：σ 再乘 $1/\sqrt{2L}$，$L$ 是层数，2 是因为每层有注意力和 MLP 两个分支。GPT-2 首先这样做，nanoGPT 和 Megatron-LM 的默认设置都是如此。
- **置零**：输出层直接初始化为 0。一开始每一层什么都不加，主干原样通过，训练中再逐步学出每一层的贡献。modded-nanogpt（NanoGPT speedrun）用了这个做法；同样的思路更早见于 Fixup（Zhang et al. 2019）和 ReZero（Bachlechner et al. 2020）。

注意，这里只置零输出层，其他矩阵仍是随机的，所以不会遇到 1.0 节的对称性问题。

```python
if init == 'zero_proj': # 残差分支的输出层置零，其余 0.02
    return 0.0 if name.endswith(('attn_wo', 'mlp_fc2')) else 0.02
```

**实验：** 0.02 + 输出层置零，val bpb 1.1798（1.1764 ~ 1.1838），比单纯的 0.02 略差 0.004，和种子的波动差不多大。我们只有 1 层，残差只加两次，"越积越大"的问题在这里还不存在。按深度缩小的倍数只有 $1/\sqrt{2}$，和 0.02 几乎没有区别，所以没有单独测。

---

## 2. weight decay：每步把参数往 0 拉一点

### 2.0 不加

不加任何约束，参数可以变得任意大。这有两个问题：

- **过拟合**：参数越大，模型越容易记住训练数据里的细节和噪声，在新数据上反而变差。
- **数值问题**：在大模型的长时间训练里，参数越变越大，可能超出 bf16 能精确表示的范围（1.06 提到过 Moonshot 的发现）。

### 2.1 L2 正则：在 loss 上加一项

最经典的做法，是在 loss 上加一项参数的平方和，惩罚大的参数：

$$L' = L + \frac{\lambda}{2}\lVert\theta\rVert^2 \quad\Rightarrow\quad \nabla L' = \nabla L + \lambda\theta$$

梯度里多出一项 $\lambda\theta$，每一步都把参数往 0 拉一点。对 SGD 来说，这等价于每步把参数乘以 $(1-\eta\lambda)$，所以也叫 **weight decay**（权重衰减）。

```python
torch.optim.Adam(params, ..., weight_decay=0.1) # PyTorch 的 Adam 里，weight_decay 就是 L2：加进梯度
```

### 2.2 AdamW：和梯度分开

在 Adam 里，L2 出了问题。$\lambda\theta$ 加进梯度以后，会和梯度一起被 $\sqrt{v}$ 除掉：

- 梯度大的参数，$\sqrt{v}$ 大，$\lambda\theta$ 被除得多，拉得反而少；
- 梯度小的参数，$\lambda\theta$ 被除得少，拉得多。

Loshchilov & Hutter (2019) 的修正是：**把 weight decay 从梯度里拿出来，单独做**。每一步先按 Adam 更新，再单独把参数乘以 $(1-\eta\lambda)$：

$$\theta \leftarrow \theta - \eta\lambda\theta - \eta\frac{\hat m}{\sqrt{\hat v}+\epsilon}$$

```python
torch.optim.AdamW(params, ..., weight_decay=0.1) # 和梯度分开
```

$\lambda = 0.1$ 是大模型的主流取值：GPT-3、LLaMA、DeepSeek-V3 都是 AdamW 0.1。

### 2.3 只加在部分参数上

不是所有参数都适合往 0 拉。常见的做法是不加在偏置和 norm 的参数上（nanoGPT）。OLMo 2 还进一步把 **embedding** 也排除在外：他们发现 embedding 会被拉得太小，而 layernorm 对输入的梯度和输入的大小成反比，embedding 太小，前几层的梯度就会变大，训练不稳定。

我们的模型没有偏置，rmsnorm 也没有参数，所以只测"embedding 不加"这一种：

```python
groups = [{'params': [wte, wpe], 'weight_decay': 0.0},
          {'params': [其余参数], 'weight_decay': 0.1}]
```

### 2.4 实验

| 做法 | val bpb（3 个种子） | 种子范围 | 相对不加 |
|---|---|---|---|
| 不加 | 1.1751 | 1.1740 ~ 1.1763 | — |
| L2 0.1 | **2.4195** | 2.4190 ~ 2.4199 | 训练失败 |
| AdamW 0.1 | 1.1813 | 1.1773 ~ 1.1862 | +0.006 |
| AdamW 0.1，embedding 不加 | 1.1827 | 1.1800 ~ 1.1861 | +0.008 |

- **L2 0.1 用在 Adam 上，完全训练失败。** 算一下就明白：
  - 参数大约 0.08，L2 往梯度里加的是 0.1 × 0.08 = 0.008；
  - 而真正的梯度，总长度只有 0.5 左右，平摊到 13.8 万个参数上，每个只有约 0.001；
  - L2 那一项比真正的梯度大好几倍，Adam 再把它归一化，变成每一步都以满学习率把参数往 0 拉。模型基本学不到东西。

  这正是 2.2 节说的问题被放大到了极致。在 Adam 里用 L2，λ 必须小得多才行；而 AdamW 把两者分开以后，同一个 0.1 就没有这个问题。
- **AdamW 两种做法都略差。** 第 4 节会看到，我们的模型完全没有过拟合，不需要正则；训练又短，参数也不会大到出数值问题。weight decay 在这里只是多了一个约束。

---

## 3. 梯度裁剪：梯度太大时缩小

### 3.0 不裁剪

偶尔会有某一步的梯度特别大，比如遇到一个不寻常的 batch，或者训练进入了不稳定的区域。按这样的梯度走一步，参数会被推得很远，loss 突然飙升，甚至直接发散。

### 3.1 逐个值裁剪

最直接的做法，是把每个梯度值都截断到 $[-c, c]$：

```python
torch.nn.utils.clip_grad_value_(params, c)
```

问题是它会**改变梯度的方向**：大的分量被截断，小的分量不变，各分量之间的比例就变了。

### 3.2 按总长度裁剪

Pascanu et al. (2013) 提出：把所有参数的梯度拼成一个长向量，算它的总长度（范数）$\lVert g\rVert$。超过上限 $c$ 时，**整体按比例缩小**：

$$g \leftarrow g \cdot \min\left(1, \frac{c}{\lVert g\rVert}\right)$$

方向不变，只是步子变小。$c = 1.0$ 几乎是所有大模型的统一选择：GPT-3、LLaMA、PaLM、DeepSeek-V3、OLMo 2。

```python
norm = torch.nn.utils.clip_grad_norm_(params, 1.0) # 返回裁剪前的总长度
```

我们没有测逐个值裁剪，只比较"不裁剪"和"按总长度裁到 1.0"。

### 3.3 实验

| 学习率 | 裁剪 | val bpb | 发散 |
|---|---|---|---|
| 0.01 | 不裁 | 1.1751 | 0 / 3 |
| 0.01 | 1.0 | 1.1772 | 0 / 3 |
| 0.04 | 不裁 | — | **3 / 3**（2 个在前 128 步，1 个在第 3800 步左右） |
| 0.04 | 1.0 | 1.2222 | **1 / 3**（在前 128 步） |

- **学习率 0.01 时，梯度的总长度平时只有 0.3 到 0.6，从来碰不到 1.0**，只在训练开头触发过一次。所以裁不裁，结果一样。
- **学习率 0.04 时，开头几十步的梯度能冲到 3 到 6**，正是在这里发散的。裁剪挡住了一部分开头的尖峰，救回了 2 个种子。但救回来的结果（1.222）远不如学习率 0.01，而且还有 1 个照样发散。

所以梯度裁剪是**防止偶尔出现的尖峰**的保险，不能让一个本来就太大的学习率变得好用。它几乎没有代价，所以大模型都加上。在大模型里，loss 突然飙升（loss spike）是常见的问题，裁剪只是手段之一。PaLM 还用了 z-loss（惩罚 softmax 的归一化项变大），以及在飙升时退回到之前的 checkpoint、跳过那一段数据。

---

## 4. dropout：训练时随机丢掉一部分

### 4.0 不加

如果模型容量很大、数据又要重复看很多遍，模型可能会**死记**训练数据：在训练集上的 loss 很低，在新数据上却不行，这叫过拟合。

### 4.1 加 dropout

dropout（Srivastava et al. 2014）在训练时，以概率 $p$ 随机把一部分激活置成 0，剩下的乘以 $1/(1-p)$，保持期望不变。评估时不丢。

直觉上，每一步都像在训练一个随机缩小的子网络，模型不能依赖任何一个特定的神经元，被迫学到更稳健的特征。

GPT-1 和 GPT-2 在三个地方用 0.1 的 dropout：embedding 之后、注意力权重上、每个子层的输出上。我们照这个位置加：

```python
x = F.dropout(tok_emb + pos_emb, p_drop)                                  # embedding 之后
x_attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p_drop) # 注意力权重上
x = F.dropout(linear(x_attn, state_dict['attn_wo']), p_drop)              # 子层的输出上
```

**现在的大模型预训练不用 dropout**（LLaMA、PaLM 都是 0，nanoGPT 的注释写着"预训练用 0 就好，微调试试 0.1+"）。原因是数据量巨大，每条数据只看一遍，不存在死记的问题。

### 4.2 只在后期加

Xue et al. (2023) 研究了数据不够、要重复看很多遍的情况。他们发现：
- 在几种常用的正则手段里，dropout 是唯一明显有效的；
- 可以只在训练后期打开，不拖慢前期的学习。

我们的数据过了约 2.7 遍（一共训练 1680 万个 token，训练集只有 630 万个），正好属于这种情况，所以也测这种做法。

### 4.3 实验

| 做法 | val bpb（3 个种子） | 种子范围 |
|---|---|---|
| 0 | 1.1751 | 1.1740 ~ 1.1763 |
| 0.1 | **1.2989** | 1.2945 ~ 1.3052 |
| 后半段才开 0.1 | **1.2927** | 1.2791 ~ 1.2995 |

- **dropout 差了 0.12，非常明显。** 后半段才打开的那一组，打开的那一刻 val bpb 立刻跳高 0.08，之后再也回不来。
- **原因是我们的模型根本没有过拟合。** 我们在训练集上也取了同样多的固定窗口来评估：训练集和验证集的 bpb 从头到尾几乎重合（最终 1.1760 对 1.1751，3 个种子的均值）。13.8 万个参数，记不住 630 万个 token。模型连学都学不够，dropout 只会进一步削弱它的容量。

---

## 汇总

| 项目 | 主流做法 | 我们的最好选择 | 为什么在我们这里没用 |
|---|---|---|---|
| 初始化 | 0.02（GPT 系列、OLMo 2），残差输出层缩小或置零 | 0.08（和 0.02、Xavier 打平） | 有 rmsnorm，σ 在 4～7 倍范围内都一样；只有 1 层，残差累积不是问题 |
| weight decay | AdamW 0.1（GPT-3、LLaMA、DeepSeek-V3），embedding 不加（OLMo 2） | 不加 | 没有过拟合，训练又短；L2 用在 Adam 上会彻底失败 |
| 梯度裁剪 | 总长度裁到 1.0（几乎所有大模型） | 加不加都一样 | 梯度平时只有 0.3～0.6，碰不到 1.0；只在大学习率的开头有用 |
| dropout | 预训练不用（LLaMA、PaLM） | 不加 | 没有过拟合，加了只会削弱容量 |

四项可以分成两类：
- **正则类**（weight decay、dropout）：防止过拟合。我们的模型还处在学不够的阶段，用不上。
- **稳定类**（初始化的细节、梯度裁剪）：防止训练不稳定。我们的模型只有 1 层、有 rmsnorm、学习率不大，本来就很稳定。

它们在大模型里有用的那些情况，我们的小模型都没有遇到。这本身就是这一节的结论。

## 参考

- Glorot & Bengio 2010, [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html)（Xavier）
- He et al. 2015, [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)（Kaiming）
- Radford et al. 2018, [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)（GPT-1）
- Radford et al. 2019, [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)（GPT-2）
- Zhang et al. 2019, [Fixup Initialization: Residual Learning Without Normalization](https://arxiv.org/abs/1901.09321)
- Bachlechner et al. 2020, [ReZero is All You Need: Fast Convergence at Large Depth](https://arxiv.org/abs/2003.04887)
- OLMo Team 2024, [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656)，3.2 节、3.4 节
- DeepSeek-AI 2024, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)，4.2 节
- NVIDIA, [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)（`init_method_std` 默认 0.02）
- Keller Jordan et al., [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
- Loshchilov & Hutter 2019, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- Brown et al. 2020, [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)，附录 B
- Touvron et al. 2023, [LLaMA](https://arxiv.org/abs/2302.13971)，2.3 节
- Pascanu et al. 2013, [On the difficulty of training Recurrent Neural Networks](https://arxiv.org/abs/1211.5063)
- Chowdhery et al. 2022, [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)，5 节
- Srivastava et al. 2014, [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html)
- Xue et al. 2023, [To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis](https://arxiv.org/abs/2305.13230)
- Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT)
