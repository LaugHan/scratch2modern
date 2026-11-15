"""
cosine：沿余弦曲线从峰值降到 0。开头和结尾降得慢，中间降得快。
"""

import math

def lr(step, num_steps):
    return 0.5 * (1 + math.cos(math.pi * step / num_steps))
