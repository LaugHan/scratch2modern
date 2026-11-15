"""
Adam：动量（梯度的一阶矩）除以梯度大小的估计（二阶矩的平方根），每个参数有自己的步长。
β 沿用 microgpt 的设置。
"""

import torch

beta1, beta2, eps = 0.85, 0.99, 1e-8

class Optimizer:
    def __init__(self, params, lr):
        self.lr = lr
        self.params = list(params.values())
        self.m = [torch.zeros_like(p) for p in self.params] # 一阶矩
        self.v = [torch.zeros_like(p) for p in self.params] # 二阶矩
        self.state = self.m + self.v
        self.t = 0

    @torch.no_grad()
    def step(self, decay):
        lr = self.lr * decay # decay 是学习率调度给出的比例
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            m.mul_(beta1).add_((1 - beta1) * p.grad)
            v.mul_(beta2).add_((1 - beta2) * p.grad ** 2)
            m_hat = m / (1 - beta1 ** self.t) # 修正开始时缓冲偏向 0 的问题
            v_hat = v / (1 - beta2 ** self.t)
            p -= lr * m_hat / (v_hat ** 0.5 + eps)
            p.grad = None
