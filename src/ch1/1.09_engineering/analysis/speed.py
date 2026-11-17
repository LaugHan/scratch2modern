"""
每一级加速在不同宽度下各快多少，以及每一步启动了多少个 kernel。
模型、优化器、每一级加速的做法都和 train.py 相同（随机权重和随机数据，速度和它们无关）。
每个组合只跑几百步测速度，不做完整训练。结果写到 out/1.09_engineering/analysis/。
"""

import os, time
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
import matplotlib.pyplot as plt
device = 'cuda:0'

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'out', '1.09_engineering', 'analysis')
os.makedirs(out_dir, exist_ok=True)

V, T, B, H = 1024, 256, 16, 4 # 词表、上下文、batch、注意力头数，和 train.py 相同
levels = ['baseline', 'no_sync', 'fused_adam', 'sdpa', 'compile', 'cudagraph']
widths = [48, 192, 768, 1536]
cpu_data = torch.randint(V, (6_000_000,))  # 和训练数据差不多大
gpu_data = cpu_data.to(device)
mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))

def softmax(logits):
    exps = (logits - logits.max(dim=-1, keepdim=True).values).exp()
    return exps / exps.sum(dim=-1, keepdim=True)

def rmsnorm(x):
    return x * (x.pow(2).mean(dim=-1, keepdim=True) + 1e-5) ** -0.5

def linear(x, w):
    return x @ w.T

def make_step(level, C):
    no_sync, fused_adam, sdpa, use_compile, cudagraph = (level >= i for i in range(1, 6))
    D = C // H
    shapes = {'wte': (V, C), 'wpe': (T, C), 'lm_head': (V, C), 'wq': (C, C), 'wk': (C, C), 'wv': (C, C), 'wo': (C, C), 'fc1': (4 * C, C), 'fc2': (C, 4 * C)}
    p = {k: (torch.randn(*s, device=device) * 0.02).requires_grad_() for k, s in shapes.items()}
    optimizer = torch.optim.Adam(p.values(), lr=1e-4, betas=(0.85, 0.99), eps=1e-8, fused=fused_adam)

    def get_batch():
        ix = torch.randint(len(cpu_data) - T - 1, (B,))
        if no_sync:
            ix = ix.pin_memory().to(device, non_blocking=True)
            offsets = ix[:, None] + torch.arange(T, device=device)
            return gpu_data[offsets], gpu_data[offsets + 1]
        x = torch.stack([cpu_data[i:i + T] for i in ix])
        y = torch.stack([cpu_data[i + 1:i + T + 1] for i in ix])
        return x.to(device), y.to(device)

    def train_loss(x, y):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            h = rmsnorm(p['wte'][x] + p['wpe'][:T])
            r = h
            h = rmsnorm(h)
            q = linear(h, p['wq']).view(B, T, H, D).transpose(1, 2)
            k = linear(h, p['wk']).view(B, T, H, D).transpose(1, 2)
            v = linear(h, p['wv']).view(B, T, H, D).transpose(1, 2)
            if sdpa:
                a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                a = softmax((q @ k.transpose(-2, -1) / D**0.5).masked_fill(~mask, float('-inf'))) @ v
            h = linear(a.transpose(1, 2).reshape(B, T, C), p['wo']) + r
            r = h
            h = linear(linear(rmsnorm(h), p['fc1']).relu(), p['fc2']) + r
            probs = softmax(linear(h, p['lm_head']))
            return -probs.gather(-1, y.unsqueeze(-1)).log().mean()
    if use_compile:
        torch._dynamo.reset() # 换了模型就重新编译
        train_loss = torch.compile(train_loss, mode='reduce-overhead' if cudagraph else 'default')

    loss_sum = [torch.zeros((), device=device)]
    def step(): # 和 train.py 的一步相同
        x, y = get_batch()
        if cudagraph:
            torch.compiler.cudagraph_mark_step_begin()
        loss = train_loss(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if no_sync:
            loss_sum[0] += loss.detach()
        else:
            loss.item()
    return step

def count_launches(step, n=5):
    # 用 profiler 记下 n 步里 GPU 上实际执行了多少个 kernel、CPU 发出了多少次启动调用。
    # 注意：profiler 运行过一次以后，同一个进程里之后的每一步都会慢约 20%，所以放在所有计时都做完以后再数
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            step()
        torch.cuda.synchronize()
    events = prof.events()
    kernels = sum(1 for e in events if e.device_type == torch.autograd.DeviceType.CUDA)
    launches = sum(1 for e in events if e.name in ('cudaLaunchKernel', 'cudaLaunchKernelExC', 'cuLaunchKernel', 'cuLaunchKernelEx', 'cudaGraphLaunch'))
    return kernels / n, launches / n

results = []
for C in widths:
    for level, name in enumerate(levels):
        step = make_step(level, C)
        for _ in range(20):
            step() # 预热：编译、录制 CUDA Graph、让 GPU 频率升上来
        torch.cuda.synchronize()
        n, t0 = 0, time.perf_counter()
        while n < 50 or time.perf_counter() - t0 < 2: # 每个组合至少测 2 秒
            step()
            n += 1
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n * 1e3
        results.append([C, name, ms, B * T / ms * 1e3, float('nan'), float('nan')])
        print(f"width {C:5d} {name:10s}: {ms:7.3f} ms/步, {B * T / ms:8.1f}k tokens/s")

# 计时全部做完以后，再在最小的宽度上数 kernel
for level, name in enumerate(levels):
    step = make_step(level, widths[0])
    for _ in range(20):
        step()
    kernels, launches = count_launches(step)
    results[level][4:] = [kernels, launches]
    print(f"width {widths[0]:5d} {name:10s}: 每步 {kernels:5.0f} 个 kernel, {launches:5.0f} 次启动调用")

with open(os.path.join(out_dir, 'speed.csv'), 'w') as f:
    f.write('width,level,ms_per_step,tokens_per_s,kernels_per_step,launches_per_step\n')
    for r in results:
        f.write(','.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in r) + '\n')

# 阶梯图：每个宽度一张，横轴是依次打开的加速，纵轴是相对 baseline 的加速比
fig, axes = plt.subplots(1, len(widths), figsize=(4 * len(widths), 4), sharey=True)
for ax, C in zip(axes, widths):
    rows = [r for r in results if r[0] == C]
    ax.bar(range(len(levels)), [rows[0][2] / r[2] for r in rows], color=[f'C{i}' for i in range(len(levels))])
    ax.set_xticks(range(len(levels)), levels, rotation=45, ha='right')
    ax.set_title(f'width {C} (baseline {rows[0][2]:.1f} ms/step)')
axes[0].set_ylabel('speedup over baseline')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'speed.png'), dpi=150)
