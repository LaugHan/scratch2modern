"""
WSD（warmup-stable-decay）：大部分时间保持峰值，只在最后 20% 的步数线性降到 0。
（warmup 在 train.py 里统一处理。）
"""

decay_frac = 0.2

def lr(step, num_steps):
    decay_start = (1 - decay_frac) * num_steps
    return 1.0 if step < decay_start else (num_steps - step) / (num_steps - decay_start)
