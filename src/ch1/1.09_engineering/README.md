# 工程加速：数学不变，只追求快

前面几节改的都是"算什么"：优化器、学习率、精度。这一节不改任何数学，只改**怎么让 GPU 把同样的东西算得更快**。

要知道怎么加速，先要知道慢在哪。所以先讲 CPU 和 GPU 是怎么配合的，再一项一项看每种加速解决的是哪一种慢。

`train.py` 里有 6 级，每一级都在前一级的基础上再打开一项：

```python
levels = ['baseline', 'no_sync', 'fused_adam', 'sdpa', 'compile', 'cudagraph']
no_sync     = level >= 1 # 减少 CPU 和 GPU 之间的同步
fused_adam  = level >= 2 # Adam 所有参数的更新合并成一个 kernel
sdpa        = level >= 3 # 注意力用 PyTorch 内置的 scaled_dot_product_attention
use_compile = level >= 4 # torch.compile：把一串逐元素运算融合成少数几个 kernel
cudagraph   = level >= 5 # CUDA Graph：把一整步的 kernel 录下来，之后一次性重放
```

---

## 1. CPU 和 GPU 是怎么配合的

### 1.1 kernel：GPU 上的一个小程序

我们写的每一行 PyTorch 代码，比如 `x + y`、`x.exp()`、`x @ w`，在 GPU 上都是一个 **kernel**：一个专门做这件事的小程序。GPU 有成千上万个计算单元，一个 kernel 启动后，让它们同时处理张量里的不同元素。

但 GPU 不会自己决定下一步做什么。**每一个 kernel 都要由 CPU 来启动**：CPU 上的 Python 执行到这一行，经过 PyTorch 一层层的调度，最后调用 CUDA 的接口，告诉 GPU"用这些参数运行这个 kernel"。这一套流程叫**启动**（launch），每次要花几微秒到十几微秒的 CPU 时间。

### 1.2 异步：CPU 发指令，GPU 排队执行

CPU 启动 kernel 时，并不等 GPU 算完，而是把指令放进一个队列（CUDA stream）就立刻返回，接着执行下一行 Python。GPU 从队列里按顺序取指令来执行。这叫**异步执行**。

理想情况是：CPU 在前面不停地发指令，GPU 在后面不停地算，两边都不闲着。

### 1.3 同步：CPU 停下来等 GPU

有些操作需要 GPU 的计算结果，CPU 必须停下来，等 GPU 把队列里的活全部干完。这叫**同步**。常见的同步点：

- `loss.item()`、`tensor.tolist()`、`print(tensor)`：要把 GPU 上的数取回 CPU；
- `if tensor > 0:`：Python 的判断需要知道具体的值；
- 从普通内存往 GPU 拷数据（见第 2 节）；
- `torch.cuda.synchronize()`：显式地等。

同步一次，CPU 发指令的流水线就断一次：CPU 等 GPU 的时候，没有人往队列里放新指令；等 GPU 算完、CPU 拿到结果再继续发指令的时候，GPU 又在等。

### 1.4 瓶颈在哪一边

一步训练的时间，由更慢的那一边决定：

- **GPU 是瓶颈**：每个 kernel 的计算量大，GPU 算得比 CPU 发得慢，队列里总有活排着。这时减少 CPU 的开销没有用。大模型通常是这种情况。
- **CPU 是瓶颈**：每个 kernel 的计算量很小，GPU 一下就算完了，大部分时间在等 CPU 发下一条指令。这时 GPU 有多快都没用，要减少的是**启动的次数**和 CPU 上的开销。

**我们的模型属于后者。** 用 profiler（PyTorch 自带的性能分析工具，能列出一步里启动了哪些 kernel、各花多少时间）数一下：baseline 每一步启动 255 次 kernel，每步 3.5 毫秒，平均每次启动约 14 微秒。这正好是 CPU 启动一个 kernel 的典型开销。也就是说，**baseline 的时间几乎全花在 CPU 发指令上**。1.04 和 1.08 都看到了同样的现象：宽度 48 时，换更快的格式、更快的 GPU 都没用。

所以下面几项，按它们减少的是什么来看：

| 加速 | 减少的是 |
|---|---|
| 减少同步 | CPU 等 GPU 的时间 |
| fused 优化器、SDPA | kernel 的个数 |
| torch.compile | kernel 的个数，以及中间结果的读写 |
| CUDA Graph | CPU 启动的次数 |

---

## 2. 减少同步

baseline 有两处同步：

```python
# 1. 每一步都把 loss 取回 CPU
train_losses.append(loss.item())

# 2. 每一步在 CPU 上拼好 batch，再拷到 GPU
x = torch.stack([data[i:i + block_size] for i in ix])
return x.to(device), y.to(device)
```

改成：

```python
# 1. loss 留在 GPU 上累加，每 128 步才取回一次
loss_sum = loss_sum + loss.detach()

# 2. 训练数据整份放进显存（只有 48 MB），每步只把 16 个起点传过去，在 GPU 上切出窗口
ix = ix.pin_memory().to(device, non_blocking=True)
offsets = ix[:, None] + torch.arange(block_size, device=device)
return data[offsets], data[offsets + 1]
```

这里有两个新名词：

- **锁页内存**（pinned memory）：普通的内存，操作系统随时可能把它挪到别处（换页），GPU 不能直接从里面读。从普通内存拷数据到 GPU，要先拷到一块固定不动的中转区，这个过程 CPU 得等着。锁页内存就是一开始就固定不动的内存，GPU 可以直接读，CPU 不用等。
- **`non_blocking=True`**：拷贝也放进队列，CPU 立刻返回，不等拷完。只有源数据在锁页内存里时才真正有效。

**我们的实验里只快了 13%。** 原因就在 1.4 节：瓶颈在 CPU 发指令，GPU 早就算完在等，`loss.item()` 让 CPU 等 GPU 时，几乎不用等。去掉同步，GPU 也只是从"等 CPU 发指令"变成"等 CPU 发指令"。**只有 GPU 是瓶颈时，同步才真正浪费时间**：那时 CPU 本可以提前把后面几步的指令都发出去。

代价：训练数据放进显存，多占了 48 MB。

---

## 3. fused 优化器：Adam 只用一个 kernel

Adam 更新一个参数要做好几次运算：更新 m、更新 v、算 bias correction、除法、减法……每次运算一个 kernel。有三种实现：

- **for 循环**：每个参数、每种运算各一个 kernel。9 个参数 × 十来种运算，就是上百个 kernel。
- **foreach**（PyTorch 的默认）：同一种运算，所有参数打包成一个 kernel 一起做。kernel 数降到十几到几十个。
- **fused**：整个 Adam 更新，所有参数、所有运算，写在**一个** kernel 里。

```python
optimizer = torch.optim.Adam(params, lr=0.01, betas=(0.85, 0.99), fused=True)
```

我们的实验里，每步的 kernel 数从 269 降到 212，少了 57 个，快了 12%。数学上完全一样，只是运算的组织方式不同。

---

## 4. SDPA 和 FlashAttention

### 4.1 手写注意力的问题

我们手写的注意力是：

```python
attn_logits = q @ k.transpose(-2, -1) / head_dim**0.5        # (B, H, T, T)
attn_logits = attn_logits.masked_fill(~causal_mask, float('-inf'))
attn_weights = softmax(attn_logits)
x_attn = attn_weights @ v
```

每一行都是一到几个 kernel，而且每一步都要把一个 T×T 的矩阵完整地写进显存，下一步再读出来。上下文 256 时，这个矩阵有 $16 \times 4 \times 256 \times 256 = 420$ 万个数。反向传播时，还要把它存着备用。上下文越长，这个矩阵越大（和 $T^2$ 成正比）。

### 4.2 FlashAttention

FlashAttention（Dao et al. 2022）的想法是：**不把 T×T 的矩阵写进显存**。

- 把 q、k、v 切成小块，一块一块地读进 GPU 芯片上很小但极快的缓存（片上内存）；
- 在片上算完这一块的分数、softmax、乘 v，再处理下一块；
- softmax 需要知道整行的最大值和总和才能归一化，FlashAttention 用一个小技巧（online softmax），边算边更新最大值和总和，最后再统一修正。

整个注意力一个 kernel 完成，T×T 的矩阵从来不出现在显存里。反向传播时，再按需重新算一遍分数，用计算换显存。显存从 $O(T^2)$ 降到 $O(T)$，长上下文也能训练。

### 4.3 SDPA：让 PyTorch 来挑

PyTorch 把各种注意力实现统一成一个函数：

```python
x_attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

SDPA（scaled dot-product attention）会根据数据类型、head_dim、硬件，自动挑一个能用的最快实现：FlashAttention、memory-efficient attention、cuDNN，都不行就退回普通的写法。我们实测：head_dim = 12 时，FlashAttention 可以用，memory-efficient 和 cuDNN 不支持。

**我们的实验里**，每步少了 22 个 kernel，快了 11%，显存从 196 MB 降到 172 MB（注意力矩阵不用存了）。

---

## 5. torch.compile：把运算融合起来

### 5.1 融合是什么

我们的 rmsnorm 是：

```python
x * (x.pow(2).mean(dim=-1, keepdim=True) + 1e-5) ** -0.5
```

不编译时，这是 5 个 kernel：平方、求平均、加、开方、乘。每个 kernel 都要从显存读一遍输入，算完再写回显存，下一个 kernel 再读进来。真正的计算量极小，时间都花在启动和读写上。

**融合**（fusion）就是把这 5 步写成 1 个 kernel：读一次输入，在芯片上算完所有步骤，只写一次结果。

### 5.2 torch.compile 怎么做

```python
batch_loss = torch.compile(batch_loss)
```

第一次调用时，PyTorch 2 的编译器（Ansel et al. 2024）分三步：

1. **抓图**（TorchDynamo）：一边运行 Python 代码，一边把用到的张量运算记录成一张计算图。
2. **生成反向**（AOTAutograd）：提前推导出反向传播的计算图，前向和反向一起优化。
3. **生成 kernel**（Inductor）：找出可以融合的运算，为它们自动生成新的 GPU kernel（用 Triton 写的）。矩阵乘法仍然交给 cuBLAS 这类专门的库。

**我们的实验里：**
- 每步的 kernel 数从 190 降到 85，快了 1.6 倍，是单项最大的加速之一。
- 显存从 172 MB 降到 99 MB。融合以后，很多中间结果根本不写进显存；而且编译器发现，有些中间结果在反向时重新算一遍（几个逐元素运算），比存着更划算。
- **代价是编译时间**：第一次运行时，前 128 步多花了约 2.5 秒。编译结果会缓存在硬盘上，之后再跑就快了。

---

## 6. CUDA Graph：录下来，一次重放

### 6.1 想法

编译以后还剩 85 个 kernel，每一个仍然要 CPU 启动一次。但每一步训练做的事情一模一样：同样的 kernel、同样的顺序、同样的形状。

**CUDA Graph** 的想法是：把一整步的所有 kernel **录制**下来，存成一张图；之后每一步，CPU 只发一条"重放这张图"的指令，GPU 自己按顺序执行里面的所有 kernel。

```python
batch_loss = torch.compile(batch_loss, mode='reduce-overhead') # reduce-overhead 模式会自动用 CUDA Graph
...
torch.compiler.cudagraph_mark_step_begin() # 每一步开始时告诉它：新的一步开始了
```

### 6.2 代价和限制

录下来的是"在这些显存地址上，做这些运算"，所以：

- **形状必须固定**。换了 batch 大小或序列长度，就要重新录一张图。所以我们只在训练时用它，评估和生成文字时不用。
- **输入要放在固定的地址上**。每一步的新数据要先拷进录制时用的那块显存。
- **中间不能有同步**。录制时不能有 `.item()` 这类操作。

**我们的实验里：** GPU 执行的 kernel 数没有减少（97 个，比 compile 还多一点），但 **CPU 的启动调用从 83 次降到 12 次**，又快了 1.9 倍。这直接说明了省下的时间在 CPU 那一侧。

---

## 7. 词表 padding：维度凑成 64 的倍数

GPU 的 Tensor Core 按固定大小的小块做矩阵乘法。矩阵的维度如果不是这个大小的整数倍，边角上的那一块就要用效率更低的方式处理。

GPT-2 的词表有 50257 个 token，nanoGPT 把它补到 50304（64 的倍数），多出来的 47 个永远用不到，但矩阵乘法更快了。

**我们的词表是 1024，本来就是 64 的倍数**，所以这一项不用做。其他维度（48、192）也都是 16 的倍数。

---

## 8. 模型越大，这些加速越没用

`analysis/speed.py` 在不同宽度上测了全部 6 级：

| 宽度 | baseline | 全部打开 | 加速 |
|---|---|---|---|
| 48 | 3.5 ms | 0.78 ms | 4.5× |
| 192 | 3.3 ms | 0.82 ms | 4.0× |
| 768 | 4.0 ms | 2.5 ms | 1.6× |
| 1536 | 10.6 ms | 9.2 ms | 1.2× |

模型变大以后，矩阵乘法本身的时间成为大头，GPU 成了瓶颈，减少启动开销就没有用了：

- 宽度 768 时，CUDA Graph 已经不再有帮助；
- 宽度 1536 时，只剩 compile 的融合还能省下一些显存读写，全部加起来只快 1.2 倍。

大模型训练中，这几项仍然是标配（几乎没有代价），但真正决定速度的是别的东西：矩阵乘法 kernel 的效率、多卡之间的通信、各种并行方式。这些留到 infra 章。

---

## 9. 大模型里怎么用

- **nanoGPT**（Karpathy）：bf16 autocast、`torch.compile`、FlashAttention（SDPA）、fused AdamW、`pin_memory` + `non_blocking` 的数据加载、词表补到 50304，基本就是这一节的全部内容。
- **modded-nanogpt**（NanoGPT speedrun）：在 nanoGPT 的基础上继续压榨，除了 1.06 提到的 Muon，还有大量这一节之外的工程技巧。
- **Megatron-LM** 等大规模训练框架：大量手写的融合 kernel（比如融合 layernorm、融合 softmax、融合 Adam），以及 FlashAttention。很多做法后来被 `torch.compile` 自动化了。

---

## 汇总

| 加速 | 一句话 | 减少的是 | 我们的加速（宽度 48） | 宽度 1536 时 | 代价 |
|---|---|---|---|---|---|
| 减少同步 | loss 留在 GPU 上，数据提前放进显存 | CPU 等 GPU 的时间 | 1.13× | 无 | 数据占显存 |
| fused Adam | 整个优化器一个 kernel | kernel 个数 | 1.12× | 微小 | 无 |
| SDPA | 注意力用内置函数，自动选 FlashAttention | kernel 个数、显存 | 1.11× | 微小 | 无 |
| torch.compile | 自动融合运算 | kernel 个数、中间结果读写 | 1.61× | 1.14× | 编译时间 |
| CUDA Graph | 录下一整步，一次重放 | CPU 启动的次数 | 1.86× | 无 | 形状必须固定 |
| **合计** | | | **4.2×**，loss 不变 | 1.2× | |

## 参考

- PyTorch 文档, [CUDA semantics: Asynchronous execution](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution)
- PyTorch 文档, [torch.optim.Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)（foreach 和 fused 两种实现）
- Dao et al. 2022, [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- Dao 2023, [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- PyTorch 文档, [torch.nn.functional.scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- Ansel et al. 2024, [PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation](https://dl.acm.org/doi/10.1145/3620665.3640366)，ASPLOS 2024
- NVIDIA 2019, [Getting Started with CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/)
- Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT)
- Keller Jordan et al., [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
