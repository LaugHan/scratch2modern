"""
Measure what the hardware can actually do, in float64:
matmul FLOPS, memory bandwidth, and the fixed cost of launching one tiny op.
"""

import time
import torch
torch.set_num_threads(1)

def bench(f, sync, n):
    f(); sync() # warmup
    t0 = time.perf_counter()
    for _ in range(n):
        f()
    sync()
    return (time.perf_counter() - t0) / n

for device in ['cpu', 'cuda:0']:
    sync = torch.cuda.synchronize if device != 'cpu' else (lambda: None)

    # matmul: 2 * N^3 FLOPs
    N = 2048 if device == 'cpu' else 8192
    a = torch.randn(N, N, dtype=torch.float64, device=device)
    b = torch.randn(N, N, dtype=torch.float64, device=device)
    t = bench(lambda: a @ b, sync, 3)
    print(f"{device:6s} | float64 matmul {N}x{N}: {2 * N**3 / t / 1e9:8.1f} GFLOPS")

    # bandwidth: a copy reads and writes every byte once
    x = torch.empty(2**26 if device == 'cpu' else 2**28, dtype=torch.float64, device=device)
    y = torch.empty_like(x)
    t = bench(lambda: y.copy_(x), sync, 5)
    print(f"{device:6s} | memory copy {x.numel() * 8 / 2**30:.1f} GB:  {2 * x.numel() * 8 / t / 1e9:8.1f} GB/s")

    # overhead: an op so small that its cost is all overhead
    s = torch.zeros(1, dtype=torch.float64, device=device)
    t = bench(lambda: s.add_(1), sync, 10000)
    print(f"{device:6s} | one tiny op:              {t * 1e6:8.1f} us")
