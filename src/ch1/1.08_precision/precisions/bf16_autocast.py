"""
bf16 混合精度：参数存 fp32，前向时矩阵乘法用 bf16 算，exp、求和这类对精度敏感的运算由 autocast 自动提升到 fp32。
"""

import torch

dtype = torch.float32     # 参数和优化器状态用什么格式存
autocast = torch.bfloat16 # 前向时矩阵乘法换成什么格式（torch.autocast），None 表示不换
tf32 = False              # fp32 的矩阵乘法是否用 tf32 算
loss_scale = None         # loss 放大的初始倍数（动态调整），None 表示不放大
linear = None             # 层内线性层的特殊实现，None 表示普通的矩阵乘法
