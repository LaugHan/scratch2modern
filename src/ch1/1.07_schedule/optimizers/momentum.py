"""
SGD + momentum：用梯度的滑动平均（动量）代替当前梯度，方向更稳，也能冲过平坦的区域。
"""

import torch

beta = 0.9

class Optimizer:
    def __init__(self, params, lr):
        self.lr = lr
        self.params = list(params.values())
        self.buf = [torch.zeros_like(p) for p in self.params] # 动量缓冲
        self.state = self.buf

    @torch.no_grad()
    def step(self, decay):
        lr = self.lr * decay # decay 是学习率调度给出的比例
        for p, buf in zip(self.params, self.buf):
            buf.mul_(beta).add_(p.grad)
            p -= lr * buf
            p.grad = None
