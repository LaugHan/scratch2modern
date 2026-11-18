"""
2.02 位置编码。
模型和 2.01 相同，只换告诉模型"第几个位置"的方式，用 pos=... 选择：
    加到 embedding 上：int（整数）、frac（[0, 1]）、binary（二进制）、sin、sincos、learned（2.01 的基线）
    改注意力：rope、alibi
    不加：none
用法：python train.py pos=rope [steps=...] ...    字段见 Config 和 common.TrainConfig
"""

import os
import sys
import math
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
    init: float = 0.02   # 2.01
    pos: str = 'learned'  # 默认就是 2.01，它的结果直接用 2.01 的，不重跑


def norm(x):
    # 没有可学权重的 RMSNorm，和 microgpt 一样
    return F.rms_norm(x, (x.size(-1),))


# ---------------- 加到 embedding 上的位置编码：每个函数给出 (T, d) 的表 ----------------
# 都只依赖序列长度 T，所以评估时可以直接换成更长的 T

def positions(T, device):
    return torch.arange(T, device=device, dtype=torch.float32)[:, None]  # (T, 1)


def int_pos(T, d, device):
    # 第 t 个位置，每一维都加 t
    return positions(T, device).expand(T, d)


def frac_pos(T, d, device):
    # 把位置缩到 [0, 1]：t / (T - 1)，T 是这条序列的长度
    return (positions(T, device) / max(T - 1, 1)).expand(T, d)


def binary_pos(T, d, device, n_bits=16):
    # t 的二进制，第 i 维是第 i 位；1024 个位置只用到前 10 位，其余维是 0
    bits = (torch.arange(T, device=device)[:, None] >> torch.arange(n_bits, device=device)) & 1
    return F.pad(bits.float(), (0, d - n_bits))


def frequencies(d, device):
    # 第 i 维的频率是 1 / 10000^(i/d)：从每个位置转 1 弧度，慢到几万个位置才转一圈
    return 10000 ** (-torch.arange(d, device=device) / d)


def sin_pos(T, d, device):
    # 每一维一个频率的 sin
    # 除以 √d：后面紧跟着 RMSNorm，这和原版 Transformer 把 token embedding 乘以 √d 完全等价，
    # 否则 sin 每一维约 0.7，token embedding 只有 0.02，一开始归一化以后几乎只剩下位置
    return torch.sin(positions(T, device) * frequencies(d, device)) / math.sqrt(d)


def sincos_pos(T, d, device):
    # 原版 Transformer：每两维共用一个频率，一维 sin、一维 cos。
    # 这样位置 t+k 的那一对 (sin, cos) 正好是位置 t 的旋转 k·频率，旋转角只和 k 有关
    angle = positions(T, device) * frequencies(d // 2, device)  # (T, d/2)
    return torch.stack([angle.sin(), angle.cos()], dim=-1).view(T, d) / math.sqrt(d)  # 除以 √d，同上


ADDITIVE = {'int': int_pos, 'frac': frac_pos, 'binary': binary_pos, 'sin': sin_pos, 'sincos': sincos_pos}


# ---------------- 改注意力的位置编码 ----------------

def rope(x, base=10000):
    # RoPE：把 q、k 每个头的 64 维分成 32 对，第 j 对按位置 t 旋转 t·频率_j 的角度。
    # q 在 t、k 在 s 时，两次旋转在点积里抵消成 (t - s)·频率，于是注意力分数只和相对距离有关
    T, D = x.shape[-2:]
    angle = positions(T, x.device) * base ** (-torch.arange(0, D, 2, device=x.device) / D)  # (T, D/2)
    cos, sin = angle.cos(), angle.sin()
    x1, x2 = x.float().chunk(2, dim=-1)  # 第 j 维和第 j + D/2 维配成一对（Llama 的写法）
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).type_as(x)


def alibi_bias(n_head, T, device):
    # ALiBi：注意力分数减去 斜率 × 距离。每个头的斜率不同，1/2, 1/4, ..., 1/256，
    # 斜率大的头只看附近，斜率小的头能看很远。同时把因果掩码也放进去
    slopes = 2 ** (-8 * torch.arange(1, n_head + 1, device=device) / n_head)
    t = torch.arange(T, device=device)
    distance = t[:, None] - t[None, :]  # 第 i 个 query 到第 j 个 key 的距离
    bias = -slopes[:, None, None] * distance
    return bias.masked_fill(distance < 0, float('-inf'))  # (H, T, T)


# ---------------- 模型 ----------------

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.pos = cfg.pos
        self.wq = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.wk = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.wv = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.wo = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        # (B, T, C) -> (B, H, T, D)，所有头一起算
        q, k, v = (w(x).view(B, T, self.n_head, -1).transpose(1, 2) for w in (self.wq, self.wk, self.wv))
        if self.pos == 'rope':
            q, k = rope(q), rope(k)
        if self.pos == 'alibi':
            bias = alibi_bias(self.n_head, T, x.device).to(q.dtype)
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        else:
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
        assert cfg.pos in ['learned', 'rope', 'alibi', 'none', *ADDITIVE], cfg.pos
        self.pos = cfg.pos
        self.block_size = cfg.block_size
        self.wte = nn.Embedding(common.vocab_size, cfg.n_embd)
        if cfg.pos == 'learned':
            self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)  # 只有 block_size 个，更长的序列没法用
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.lm_head = nn.Linear(cfg.n_embd, common.vocab_size, bias=False)
        for p in self.parameters():
            nn.init.normal_(p, std=cfg.init)

    def forward(self, idx):
        T = idx.size(1)
        x = self.wte(idx)
        if self.pos == 'learned':
            x = x + self.wpe(torch.arange(T, device=idx.device))
        elif self.pos in ADDITIVE:
            x = x + ADDITIVE[self.pos](T, x.size(-1), idx.device)
        x = norm(x)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)


if __name__ == '__main__':
    cfg, run_name = common.parse_config(Config)
    torch.manual_seed(cfg.seed)
    model = GPT(cfg).to(common.device)
    n_params = sum(p.numel() for p in model.parameters())
    n_blocks = sum(p.numel() for p in model.blocks.parameters())
    print(f"run {run_name}: {n_params / 1e6:.1f}M params, {n_blocks / 1e6:.1f}M in blocks")

    path = common.out_dir(__file__, run_name)
    if common.train(model, cfg, path) is not None:
        common.write_samples(model, path, ['The water cycle', 'In 1492,', 'Photosynthesis is'])
