"""
Where does the time of one training step go? Same model as 1.04 (random weights, speed does not depend on them).
CPU timings depend on glibc malloc, see notes.md; for a fair CPU number run with
MALLOC_MMAP_THRESHOLD_=1073741824 MALLOC_TRIM_THRESHOLD_=4294967296
"""

import time
import torch
from torch.profiler import profile, ProfilerActivity
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

V, C, T, L, H = 89, 16, 256, 1, 4
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
    with torch.no_grad(): # the same Adam update as train.py
        for k, w in p.items():
            m_buf[k] = 0.85 * m_buf[k] + 0.15 * w.grad
            v_buf[k] = 0.99 * v_buf[k] + 0.01 * w.grad ** 2
            w -= 0.01 * (m_buf[k] / 0.5) / ((v_buf[k] / 0.5) ** 0.5 + 1e-8)
            w.grad = None
    return loss.item()

shapes = {'wte': (V, C), 'wpe': (T, C), 'lm_head': (V, C), 'wq': (C, C), 'wk': (C, C), 'wv': (C, C), 'wo': (C, C), 'fc1': (4 * C, C), 'fc2': (C, 4 * C)}
for device in ['cpu', 'cuda:0']:
    for B in [1, 64]:
        p = {k: (torch.randn(*s, device=device) * 0.08).requires_grad_() for k, s in shapes.items()}
        m_buf = {k: torch.zeros_like(w) for k, w in p.items()}
        v_buf = {k: torch.zeros_like(w) for k, w in p.items()}
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
        x = torch.randint(V, (B, T), device=device)
        y = torch.randint(V, (B, T), device=device)
        for _ in range(20):
            step(p, m_buf, v_buf, mask, x, y) # warmup
        n = 50
        t0 = time.perf_counter() # time without the profiler, it adds its own overhead
        for _ in range(n):
            step(p, m_buf, v_buf, mask, x, y)
        wall = (time.perf_counter() - t0) / n
        activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device != 'cpu' else [])
        with profile(activities=activities) as prof:
            for _ in range(n):
                step(p, m_buf, v_buf, mask, x, y)
        print(f"=== {device}, batch {B}: {wall * 1e3:.2f} ms per step ===")
        if device != 'cpu':
            kernels = [e for e in prof.events() if e.device_type.name == 'CUDA']
            busy = sum(e.device_time for e in kernels) / n # us
            print(f"kernels per step: {len(kernels) / n:.0f}, GPU busy {busy / 1e3:.2f} ms ({busy / 1e3 / (wall * 1e3):.0%} of the step)")
            print(prof.key_averages().table(sort_by='self_device_time_total', row_limit=6, max_name_column_width=40))
        else:
            print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=6, max_name_column_width=40))
