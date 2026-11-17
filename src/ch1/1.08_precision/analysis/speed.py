"""
模型宽度从 48 加到 3072，每种精度测一遍训练速度。模型、优化器、精度的处理都和 train.py 相同（随机权重，速度和权重无关）。
每个组合只跑几十步测速度，不做完整训练。结果写到 out/1.08_precision/analysis/。
"""

import os, sys, time, importlib, contextlib
import torch
import matplotlib.pyplot as plt
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..')) # 用上一级的 precisions/
device = 'cuda:0'

out_dir = os.path.join(here, '..', '..', '..', 'out', '1.08_precision', 'analysis')
os.makedirs(out_dir, exist_ok=True)

V, T, B, H = 1024, 256, 16, 4 # 词表、上下文、batch、注意力头数，和 train.py 相同
precisions = ['fp64', 'fp32', 'tf32', 'bf16_autocast', 'bf16_pure', 'fp16_scaler', 'fp8']
widths = [48, 96, 192, 384, 768, 1536, 3072]

def softmax(logits):
    exps = (logits - logits.max(dim=-1, keepdim=True).values).exp()
    return exps / exps.sum(dim=-1, keepdim=True)

def rmsnorm(x):
    return x * (x.pow(2).mean(dim=-1, keepdim=True) + 1e-5) ** -0.5

def linear(x, w):
    return x @ w.T

def measure(name, C):
    P = importlib.import_module(f'precisions.{name}')
    torch.backends.cuda.matmul.allow_tf32 = P.tf32
    autocast = lambda: torch.autocast('cuda', dtype=P.autocast) if P.autocast else contextlib.nullcontext()
    block_linear = P.linear or linear
    D = C // H
    shapes = {'wte': (V, C), 'wpe': (T, C), 'lm_head': (V, C), 'wq': (C, C), 'wk': (C, C), 'wv': (C, C), 'wo': (C, C), 'fc1': (4 * C, C), 'fc2': (C, 4 * C)}
    p = {k: (torch.randn(*s, device=device, dtype=P.dtype) * 0.02).requires_grad_() for k, s in shapes.items()}
    optimizer = torch.optim.Adam(p.values(), lr=1e-4, betas=(0.85, 0.99), eps=1e-8)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    x = torch.randint(V, (B, T), device=device)
    y = torch.randint(V, (B, T), device=device)
    scaler = torch.amp.GradScaler('cuda', init_scale=P.loss_scale or 1.0, enabled=P.loss_scale is not None)

    def step(): # 和 train.py 的一步相同：前向、反向、（loss scaling）、Adam 更新
        with autocast():
            h = rmsnorm(p['wte'][x] + p['wpe'][:T])
            r = h
            h = rmsnorm(h)
            q = block_linear(h, p['wq']).view(B, T, H, D).transpose(1, 2)
            k = block_linear(h, p['wk']).view(B, T, H, D).transpose(1, 2)
            v = block_linear(h, p['wv']).view(B, T, H, D).transpose(1, 2)
            att = softmax((q @ k.transpose(-2, -1) / D**0.5).masked_fill(~mask, float('-inf')))
            h = block_linear((att @ v).transpose(1, 2).reshape(B, T, C), p['wo']) + r
            r = h
            h = block_linear(block_linear(rmsnorm(h), p['fc1']).relu(), p['fc2']) + r
            probs = softmax(linear(h, p['lm_head']))
            loss = -probs.gather(-1, y.unsqueeze(-1)).log().mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        return loss.item() # .item() 会等 GPU 算完这一步

    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(10):
        step() # 预热，也让 GPU 频率升上来
    n, t0 = 0, time.perf_counter()
    while n < 10 or time.perf_counter() - t0 < 1: # 每个组合至少测 1 秒
        step()
        n += 1
    ms = (time.perf_counter() - t0) / n * 1e3
    mem = torch.cuda.max_memory_allocated(device) / 2**20
    # 每步的计算量：矩阵乘法 2×（行×列）×token 数，前向 1 份、反向 2 份，一共 6 份；再加上注意力的两次矩阵乘法
    flops = 6 * B * T * (12 * C * C + V * C) + 12 * B * T * T * C
    return ms, flops / ms / 1e9, mem

results = []
for C in widths:
    for name in precisions:
        ms, tflops, mem = measure(name, C)
        results.append((C, name, ms, B * T / ms * 1e3, tflops, mem))
        print(f"width {C:5d} {name:14s}: {ms:8.2f} ms/步, {B * T / ms:8.1f}k tokens/s, {tflops:6.2f} TFLOPS, 显存峰值 {mem:7.0f} MB")
        torch.cuda.empty_cache()

with open(os.path.join(out_dir, 'speed.csv'), 'w') as f:
    f.write('width,precision,ms_per_step,tokens_per_s,tflops,peak_mem_mb\n')
    for r in results:
        f.write(','.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in r) + '\n')

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for name in precisions:
    rows = [r for r in results if r[1] == name]
    axes[0].plot([r[0] for r in rows], [r[2] for r in rows], 'o-', label=name)
    axes[1].plot([r[0] for r in rows], [r[4] for r in rows], 'o-', label=name)
    axes[2].plot([r[0] for r in rows], [r[5] for r in rows], 'o-', label=name)
axes[0].set(xlabel='width', ylabel='ms per step', xscale='log', yscale='log')
axes[1].set(xlabel='width', ylabel='achieved TFLOPS', xscale='log', yscale='log')
axes[2].set(xlabel='width', ylabel='peak GPU memory (MB)', xscale='log', yscale='log')
for ax in axes:
    ax.set_xticks(widths, [str(w) for w in widths], minor=False)
    ax.set_xticks([], minor=True)
axes[0].legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'speed.png'), dpi=150)
