"""
SGD：沿着梯度的反方向走一步。
"""

import torch

class Optimizer:
    def __init__(self, params, lr):
        self.lr = lr
        self.params = list(params.values())
        self.state = [] # 不需要额外的缓冲

    @torch.no_grad()
    def step(self, decay):
        lr = self.lr * decay # decay 是学习率调度给出的比例
        for p in self.params:
            p -= lr * p.grad
            p.grad = None
