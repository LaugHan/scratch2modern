"""
linear：从峰值线性降到 0。1.01 到 1.06 一直用的就是它。
"""

def lr(step, num_steps):
    return 1 - step / num_steps
