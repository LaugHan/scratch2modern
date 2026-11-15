"""
Muon：对 Transformer 内部的矩阵（q, k, v, o, MLP），把动量矩阵"正交化"后再更新，
让更新在各个方向上的幅度一样大。正交化用 Newton–Schulz 迭代近似。
wte、wpe、lm_head 不是普通的矩阵乘法，仍然用 Adam，学习率固定为 1.05 的 0.01。
"""

import torch
from . import adam

beta = 0.95
adam_lr = 0.01

def orthogonalize(G, steps=5):
    # Newton–Schulz 迭代：把 G 的奇异值都推向 1，相当于近似计算 U Vᵀ（G = U S Vᵀ）
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    transposed = G.shape[0] > G.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if transposed else X

class Optimizer:
    def __init__(self, params, lr):
        self.lr = lr
        self.matrices = [p for name, p in params.items() if name.startswith('layer')]
        self.adam = adam.Optimizer({name: p for name, p in params.items() if not name.startswith('layer')}, adam_lr)
        self.buf = [torch.zeros_like(p) for p in self.matrices]
        self.state = self.buf + self.adam.state

    @torch.no_grad()
    def step(self, decay):
        lr = self.lr * decay # decay 是学习率调度给出的比例
        for p, buf in zip(self.matrices, self.buf):
            buf.mul_(beta).add_(p.grad)
            g = p.grad + beta * buf # Nesterov 动量
            scale = max(1, p.shape[0] / p.shape[1]) ** 0.5 # 让高矩阵和宽矩阵的更新幅度一致
            p -= lr * scale * orthogonalize(g)
            p.grad = None
        self.adam.step(decay) # 学习率衰减同样作用在 Adam 部分
