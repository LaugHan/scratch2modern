"""
测几个尺寸的训练速度，用来挑这一节的模型大小。不训练，只量每步耗时和显存。
用法：CUDA_VISIBLE_DEVICES=0 python speed.py     写出 out/ch2/2.01_scale/speed.csv
"""

import os
import time

import torch

from train import GPT, Config  # import train 时，它把 ch2/ 加进了 sys.path
import common

sizes = [(6, 384), (8, 512), (12, 512), (12, 768)]  # (层数, 宽度)，每个头都是 64 维
n_steps, n_warmup = 30, 10

path = os.path.join(common.ROOT, 'out', 'ch2', '2.01_scale')
os.makedirs(path, exist_ok=True)
rows = []
for n_layer, n_embd in sizes:
    torch.cuda.reset_peak_memory_stats()
    cfg = Config(n_layer=n_layer, n_embd=n_embd, n_head=n_embd // 64)
    model = GPT(cfg).to(common.device)
    optimizers = common.make_optimizers(model, cfg)
    compiled = torch.compile(model)
    data = torch.randint(0, common.EOT, (10_000_000,), device=common.device, dtype=torch.int32)  # 随机 token 就够测速
    for step in range(n_warmup + n_steps):
        if step == n_warmup:  # 前几步包括编译，不计时
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        x, y = common.get_batch(data, cfg.batch_size, cfg.block_size)
        common.loss_fn(compiled, x, y).backward()
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    step_time = (time.perf_counter() - t0) / n_steps

    n_params = sum(p.numel() for p in model.parameters())
    n_blocks = sum(p.numel() for p in model.blocks.parameters())
    tokens_per_s = cfg.batch_size * cfg.block_size / step_time
    rows.append(f"{n_layer},{n_embd},{n_params},{n_blocks},{step_time:.4f},{tokens_per_s:.0f},"
                f"{torch.cuda.max_memory_allocated() / 2**30:.1f}")
    print(f"{n_layer} layers, width {n_embd}: {n_params / 1e6:.0f}M params ({n_blocks / 1e6:.1f}M in blocks), "
          f"{step_time * 1e3:.0f} ms/step, {tokens_per_s / 1e3:.0f}K tokens/s")
    del model, compiled, optimizers
    torch._dynamo.reset()

with open(os.path.join(path, 'speed.csv'), 'w') as f:
    f.write('n_layer,n_embd,params,block_params,step_time,tokens_per_s,peak_memory_gb\n')
    f.write('\n'.join(rows) + '\n')
