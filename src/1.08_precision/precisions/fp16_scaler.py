"""
fp16 混合精度：和 bf16 混合精度一样，只是换成 fp16。fp16 的范围小，小梯度会下溢成 0，所以先把 loss 放大，反向传播完再把梯度缩回去。
"""

import torch

dtype = torch.float32     # 参数和优化器状态用什么格式存
autocast = torch.float16  # 前向时矩阵乘法换成什么格式（torch.autocast），None 表示不换
tf32 = False              # fp32 的矩阵乘法是否用 tf32 算
loss_scale = 2.0 ** 16    # loss 放大的初始倍数（动态调整），None 表示不放大
linear = None             # 层内线性层的特殊实现，None 表示普通的矩阵乘法
