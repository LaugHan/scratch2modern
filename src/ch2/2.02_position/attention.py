"""
外推时注意力落在哪：窗口 4096，看第 3072 到 4095 个位置的 query，
分给距离超过 1024 的 key 的注意力占多少（训练时距离最多是 1023）。
用法：CUDA_VISIBLE_DEVICES=0 python attention.py     写出 out/ch2/2.02_position/attention_far.csv
"""

import os

import torch

from train import GPT, Config, norm, rope, alibi_bias
import common

T, n_windows, limit = 4096, 4, 1024
out = os.path.join(common.ROOT, 'out', 'ch2', '2.02_position')
x_all, _ = common.fixed_windows(common.load_tokens('fineweb_edu', 'val'), n_windows, T)
t = torch.arange(T, device=common.device)
distance = t[:, None] - t[None, :]
far = (distance >= limit)[3 * T // 4:]  # 只看最后 1024 个 query：它们前面都有 3000 多个 token

rows = []
for pos in ['alibi', 'rope']:
    model = GPT(Config(pos=pos)).to(common.device)
    model.load_state_dict(torch.load(os.path.join(out, f'pos{pos}', 'model.pt')))
    mass = torch.zeros(len(model.blocks), 8, device=common.device)
    with torch.no_grad():
        for idx in x_all:
            x = norm(model.wte(idx[None]))
            for i, block in enumerate(model.blocks):
                # 照着 Attention.forward 把注意力权重算出来
                attn, h = block.attn, norm(x)
                q, k = (w(h).view(1, T, attn.n_head, -1).transpose(1, 2) for w in (attn.wq, attn.wk))
                if pos == 'rope':
                    q, k = rope(q), rope(k)
                    bias = torch.zeros(T, T, device=common.device).masked_fill(distance < 0, float('-inf'))
                else:
                    bias = alibi_bias(attn.n_head, T, idx.device)
                weights = torch.softmax(q @ k.transpose(-1, -2) / q.size(-1) ** 0.5 + bias, dim=-1)
                mass[i] += (weights[0, :, 3 * T // 4:] * far).sum(-1).mean(-1) / n_windows
                x = block(x)
    for i in range(len(model.blocks)):
        rows.append(f"{pos},{i}," + ','.join(f"{m:.5f}" for m in mass[i].tolist()))
        print(f"{pos:6s} layer {i}: " + ' '.join(f"{m:.3f}" for m in mass[i].tolist()))
    del model

with open(os.path.join(out, 'attention_far.csv'), 'w') as f:
    f.write('pos,layer,' + ','.join(f'head{h}' for h in range(8)) + '\n')
    f.write('\n'.join(rows) + '\n')
