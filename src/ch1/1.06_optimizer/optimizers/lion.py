"""
Lion：只取动量的符号，每个参数每一步都走同样大小的一步（±lr）。只需要一个缓冲。
步长固定，所以学习率通常比 Adam 小 3～10 倍。
"""

import torch

beta1, beta2 = 0.9, 0.99

class Optimizer:
    def __init__(self, params, lr):
        self.lr = lr
        self.params = list(params.values())
        self.m = [torch.zeros_like(p) for p in self.params]
        self.state = self.m

    @torch.no_grad()
    def step(self, decay):
        lr = self.lr * decay # decay 是学习率调度给出的比例
        for p, m in zip(self.params, self.m):
            p -= lr * (beta1 * m + (1 - beta1) * p.grad).sign() # 更新用的是 β1 混合
            m.mul_(beta2).add_((1 - beta2) * p.grad)          # 动量本身用 β2 更新
            p.grad = None
