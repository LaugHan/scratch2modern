"""
2.01 更大的模型，更多的数据。
模型结构和 1.12 完全相同，只是放大，数据从 TinyStories 换成 FineWeb-Edu，上下文从 256 加到 1024。
先运行 prepare.py。
用法：python train.py [n_layer=8] [n_embd=512] [lr=0.02] [steps=...] ...    字段见 Config 和 common.TrainConfig
"""

import os
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import common


@dataclass
class Config(common.TrainConfig):
    n_layer: int = 8
    n_embd: int = 512
    n_head: int = 8      # 每个头 64 维
    init: float = 0.02   # 所有参数都从 N(0, 0.02²) 开始；1.10 的 0.08 放大以后太大，2.01 扫过


def norm(x):
    # 没有可学权重的 RMSNorm，和 microgpt 一样
    return F.rms_norm(x, (x.size(-1),))


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.wq = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.wk = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.wv = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.wo = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        # (B, T, C) -> (B, H, T, D)，所有头一起算
        q, k, v = (w(x).view(B, T, self.n_head, -1).transpose(1, 2) for w in (self.wq, self.wk, self.wv))
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(x.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.fc2 = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(norm(x))
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.block_size = cfg.block_size
        self.wte = nn.Embedding(common.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)  # 学出来的绝对位置
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.lm_head = nn.Linear(cfg.n_embd, common.vocab_size, bias=False)
        for p in self.parameters():
            nn.init.normal_(p, std=cfg.init)

    def forward(self, idx):
        x = self.wte(idx) + self.wpe(torch.arange(idx.size(1), device=idx.device))
        x = norm(x)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)


if __name__ == '__main__':
    cfg, run_name = common.parse_config(Config)
    torch.manual_seed(cfg.seed)
    model = GPT(cfg).to(common.device)
    n_params = sum(p.numel() for p in model.parameters())
    n_embed = model.wte.weight.numel() + model.wpe.weight.numel() + model.lm_head.weight.numel()
    print(f"run {run_name}: {n_params / 1e6:.1f}M params, {(n_params - n_embed) / 1e6:.1f}M in blocks")

    path = common.out_dir(__file__, run_name)
    if common.train(model, cfg, path) is not None:
        common.write_samples(model, path, ['The water cycle', 'In 1492,', 'Photosynthesis is'])
