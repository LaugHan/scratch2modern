"""
GPU 上 batch 从 1 一直加到显存放不下，看吞吐怎么变。模型和 1.04 相同（随机权重，速度和权重无关）。
每个 batch 只跑几十步测速度，不做完整训练。结果写到 out/1.04_cuda/analysis/。
"""

import os, time
import torch
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)
device = 'cuda:0'

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'out', '1.04_cuda', 'analysis')
os.makedirs(out_dir, exist_ok=True)

V, C, T, H = 89, 16, 256, 4
D = C // H

def linear(x, w):
    return x @ w.T

def softmax(logits):
    exps = (logits - logits.max(dim=-1, keepdim=True).values).exp()
    return exps / exps.sum(dim=-1, keepdim=True)

def rmsnorm(x):
    return x * (x.pow(2).mean(dim=-1, keepdim=True) + 1e-5) ** -0.5

def step(p, m_buf, v_buf, mask, x, y):
    B = x.shape[0]
    h = rmsnorm(p['wte'][x] + p['wpe'][:T])
    r = h
    h = rmsnorm(h)
    q = linear(h, p['wq']).view(B, T, H, D).transpose(1, 2)
    k = linear(h, p['wk']).view(B, T, H, D).transpose(1, 2)
    v = linear(h, p['wv']).view(B, T, H, D).transpose(1, 2)
    att = softmax((q @ k.transpose(-2, -1) / D**0.5).masked_fill(~mask, float('-inf')))
    h = linear((att @ v).transpose(1, 2).reshape(B, T, C), p['wo']) + r
    r = h
    h = linear(linear(rmsnorm(h), p['fc1']).relu(), p['fc2']) + r
    probs = softmax(linear(h, p['lm_head']))
    loss = -probs.gather(-1, y.unsqueeze(-1)).log().mean()
    loss.backward()
    with torch.no_grad(): # 和 train.py 相同的 Adam 更新
        for k, w in p.items():
            m_buf[k] = 0.85 * m_buf[k] + 0.15 * w.grad
            v_buf[k] = 0.99 * v_buf[k] + 0.01 * w.grad ** 2
            w -= 0.01 * (m_buf[k] / 0.5) / ((v_buf[k] / 0.5) ** 0.5 + 1e-8)
            w.grad = None
    return loss.item() # .item() 会等 GPU 算完这一步

shapes = {'wte': (V, C), 'wpe': (T, C), 'lm_head': (V, C), 'wq': (C, C), 'wk': (C, C), 'wv': (C, C), 'wo': (C, C), 'fc1': (4 * C, C), 'fc2': (C, 4 * C)}
p = {k: (torch.randn(*s, device=device) * 0.08).requires_grad_() for k, s in shapes.items()}
m_buf = {k: torch.zeros_like(w) for k, w in p.items()}
v_buf = {k: torch.zeros_like(w) for k, w in p.items()}
mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))

# batch 按 1, 2, 3, 4, 6, 8, 12, 16, ... 增长，一直到显存放不下
batches = sorted({b * s for b in [2**i for i in range(16)] for s in (2, 3)} | {1}) # 1, 2, 3, 4, 6, 8, ...
results = []
for B in batches:
    x = torch.randint(V, (B, T), device=device)
    y = torch.randint(V, (B, T), device=device)
    try:
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(10):
            step(p, m_buf, v_buf, mask, x, y) # 预热，也让 GPU 频率升上来
        n, t0 = 0, time.perf_counter()
        while n < 10 or time.perf_counter() - t0 < 1: # 每个 batch 至少测 1 秒
            step(p, m_buf, v_buf, mask, x, y)
            n += 1
        ms = (time.perf_counter() - t0) / n * 1e3
    except torch.cuda.OutOfMemoryError:
        print(f"batch {B}: 显存放不下")
        break
    mem = torch.cuda.max_memory_allocated(device) / 2**30
    att_mb = B * H * T * T * 8 / 2**20 # 一份注意力矩阵的大小
    results.append((B, ms, B * T / ms * 1e3, mem, att_mb))
    print(f"batch {B:5d}: {ms:8.2f} ms/步, {B * T / ms * 1e3 / 1e3:7.1f}k tokens/s, 显存峰值 {mem:6.2f} GB, 一份注意力矩阵 {att_mb:7.0f} MB")
    del x, y
    torch.cuda.empty_cache()

with open(os.path.join(out_dir, 'batch_sweep.csv'), 'w') as f:
    f.write('batch,ms_per_step,tokens_per_s,peak_mem_gb,attn_mb\n')
    for r in results:
        f.write(','.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in r) + '\n')

B, ms, tps, mem, _ = zip(*results)
l2 = torch.cuda.get_device_properties(device).L2_cache_size / (H * T * T * 8) # 注意力矩阵正好装满 L2 的 batch
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(B, ms, 'o-')
axes[0].set(xlabel='batch', ylabel='ms per step', xscale='log', yscale='log')
axes[1].plot(B, [t / 1e3 for t in tps], 'o-')
axes[1].set(xlabel='batch', ylabel='tokens/s (k)', xscale='log')
axes[2].plot(B, mem, 'o-')
axes[2].set(xlabel='batch', ylabel='peak GPU memory (GB)', xscale='log', yscale='log')
for ax in axes:
    ax.axvline(l2, color='gray', ls='--', label=f'attention matrix = L2 ({l2:.0f})')
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'batch_sweep.png'), dpi=150)
