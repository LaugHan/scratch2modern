"""
AdamW：Adam 加上解耦的 weight decay，每一步把参数按比例往 0 拉一点，不经过梯度。
"""

import torch

beta1, beta2, eps = 0.85, 0.99, 1e-8
weight_decay = 0.1

class Optimizer:
    def __init__(self, params, lr):
        self.lr = lr
        self.params = list(params.values())
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.state = self.m + self.v
        self.t = 0

    @torch.no_grad()
    def step(self, decay):
        lr = self.lr * decay # decay 是学习率调度给出的比例
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            p.mul_(1 - lr * weight_decay) # 和 Adam 唯一的区别
            m.mul_(beta1).add_((1 - beta1) * p.grad)
            v.mul_(beta2).add_((1 - beta2) * p.grad ** 2)
            m_hat = m / (1 - beta1 ** self.t)
            v_hat = v / (1 - beta2 ** self.t)
            p -= lr * m_hat / (v_hat ** 0.5 + eps)
            p.grad = None
