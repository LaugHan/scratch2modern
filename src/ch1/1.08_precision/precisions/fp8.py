"""
fp8：在 bf16 混合精度的基础上，把 Transformer 层内的 6 个线性层（q、k、v、o、MLP 的两个）换成 fp8 矩阵乘法。
前向用 E4M3（尾数多一位，精度高一些），反向的梯度用 E5M2（指数多一位，范围大一些），每个张量一个缩放因子。
lm_head 和注意力里的两次矩阵乘法保持 bf16。
"""

import torch

dtype = torch.float32     # 参数和优化器状态用什么格式存
autocast = torch.bfloat16 # 前向时矩阵乘法换成什么格式（torch.autocast），None 表示不换
tf32 = False              # fp32 的矩阵乘法是否用 tf32 算
loss_scale = None         # loss 放大的初始倍数（动态调整），None 表示不放大

E4M3, E5M2 = torch.float8_e4m3fn, torch.float8_e5m2

def quantize(x, fmt):
    # fp8 能表示的范围很小（E4M3 最大 448，E5M2 最大 57344），先把整个张量缩放到刚好填满这个范围
    scale = x.abs().max().float().clamp(min=1e-12) / torch.finfo(fmt).max
    return (x / scale).to(fmt), scale

def mm(a, b, out_dtype):
    # 算 a @ b。a、b 都是 (fp8 张量, 缩放因子)，结果再乘回两个缩放因子。_scaled_mm 要求 a 按行存、b 按列存
    (qa, sa), (qb, sb) = a, b
    return torch._scaled_mm(qa.contiguous(), qb.t().contiguous().t(), scale_a=sa, scale_b=sb, out_dtype=out_dtype)

class Linear(torch.autograd.Function):
    # y = x @ w.T；前向和反向的三次矩阵乘法都用 fp8
    @staticmethod
    def forward(ctx, x, w):
        x2 = x.reshape(-1, x.shape[-1])
        xq, wq = quantize(x2, E4M3), quantize(w, E4M3)
        ctx.save_for_backward(*xq, *wq) # 反向直接用量化后的 x 和 w
        ctx.x_shape, ctx.x_dtype = x.shape, x.dtype
        y = mm(xq, (wq[0].t(), wq[1]), torch.bfloat16)
        return y.reshape(*x.shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, grad_y):
        xq, sx, wq, sw = ctx.saved_tensors
        gq, sg = quantize(grad_y.reshape(-1, grad_y.shape[-1]), E5M2)
        grad_x = mm((gq, sg), (wq, sw), torch.bfloat16)               # grad_y @ w
        grad_w = mm((gq.t(), sg), (xq, sx), torch.float32)            # grad_y.T @ x
        return grad_x.reshape(ctx.x_shape).to(ctx.x_dtype), grad_w

def linear(x, w):
    return Linear.apply(x, w)
