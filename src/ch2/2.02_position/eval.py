"""
用训练好的模型算每个位置的平均 loss，并把窗口拉长到训练时没见过的长度，看外推。
用法：CUDA_VISIBLE_DEVICES=0 python eval.py     先跑完 sweep.sh
写出 out/ch2/2.02_position/position_loss_<长度>.csv：每行一个位置，每列一种位置编码
"""

import os

import torch

from train import GPT, Config  # import train 时，它把 ch2/ 加进了 sys.path
import common

out = os.path.join(common.ROOT, 'out', 'ch2')
# 学出来的绝对位置就是 2.01 的基线
runs = {'learned': os.path.join(out, '2.01_scale', 'default')}
for pos in ['rope', 'alibi', 'none', 'sincos', 'sin', 'binary', 'frac', 'int']:
    runs[pos] = os.path.join(out, '2.02_position', f'pos{pos}')

lengths = [1024, 2048, 4096]   # 训练时是 1024
eval_tokens = 512 * 1024       # 每种长度都评估同样多的 token，和训练时的验证集一样
batch_tokens = 16 * 1024


@torch.no_grad()
def position_loss(model, x, y):
    # 每个位置上的平均 loss：先对每个窗口算逐 token 的 loss，再沿窗口方向平均
    batch_size = batch_tokens // x.size(1)
    nats = sum(common.loss_fn(model, x[i:i + batch_size], y[i:i + batch_size], 'none')
               .view(-1, x.size(1)).double().sum(0)
               for i in range(0, len(x), batch_size))
    return (nats / len(x)).tolist()


val_data = common.load_tokens('fineweb_edu', 'val')
results = {T: {} for T in lengths}
for pos, path in runs.items():
    model = GPT(Config(pos=pos)).to(common.device)
    model.load_state_dict(torch.load(os.path.join(path, 'model.pt')))
    model.eval()
    for T in lengths:
        if pos == 'learned' and T > model.block_size:
            continue  # 只有 1024 个位置向量，更长的根本没法跑
        x, y = common.fixed_windows(val_data, eval_tokens // T, T)
        loss = position_loss(model, x, y)
        results[T][pos] = loss
        inside = sum(loss[:1024]) / 1024
        beyond = f", beyond 1024 {sum(loss[1024:]) / (T - 1024):.4f}" if T > 1024 else ''
        print(f"{pos:8s} length {T}: mean {sum(loss) / T:.4f}, first 1024 {inside:.4f}{beyond}", flush=True)
    del model

for T in lengths:
    names = list(results[T])
    with open(os.path.join(out, '2.02_position', f'position_loss_{T}.csv'), 'w') as f:
        f.write('position,' + ','.join(names) + '\n')
        for t in range(T):
            f.write(f"{t}," + ','.join(f"{results[T][n][t]:.5f}" for n in names) + '\n')
