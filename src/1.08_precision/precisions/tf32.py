"""
tf32：参数和计算都用 fp32，但矩阵乘法在 Tensor Core 上用 tf32 算（尾数从 23 位截到 10 位）。
"""

import torch

dtype = torch.float32     # 参数和优化器状态用什么格式存
autocast = None           # 前向时矩阵乘法换成什么格式（torch.autocast），None 表示不换
tf32 = True               # fp32 的矩阵乘法是否用 tf32 算
loss_scale = None         # loss 放大的初始倍数（动态调整），None 表示不放大
linear = None             # 层内线性层的特殊实现，None 表示普通的矩阵乘法
