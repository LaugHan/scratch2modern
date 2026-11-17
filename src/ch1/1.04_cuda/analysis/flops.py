"""
FLOPs per token of the 1.04 model, and the FLOPS / MFU we actually reached in 1.03 (CPU) and 1.04 (GPU).
"""

import os, csv

# Model (TinyStories setting)
V, C, T, L = 89, 16, 256, 1 # vocab size, width, context length, layers

# Forward FLOPs per token. A matmul with an (out, in) weight costs 2 * out * in per token.
matmul = L * (4 * 2 * C * C + 2 * 2 * C * 4 * C) + 2 * C * V # q, k, v, o + mlp + lm_head
attention = L * (2 * T * C + 2 * T * C) # q @ k^T and weights @ v, over all T positions (the mask does not skip work)
forward = matmul + attention
train = 3 * forward # backward is about 2x forward
print(f"forward FLOPs per token: matmul {matmul}, attention {attention}, total {forward}")
print(f"training FLOPs per token (3x forward): {train}\n")

# Peak float64 FLOPS
peaks = {
    # Zen 4: 2 FMA units x 256-bit (4 doubles) x 2 FLOPs per FMA = 16 FLOPs/cycle; 3.75 GHz boost; 1 thread
    'CPU': 16 * 3.75e9,
    # Ada: 110 SMs x 128 FP32 lanes x 2 FLOPs x 2.46 GHz boost = 69.3 TFLOPS float32; float64 runs at 1/64 of that
    'GPU': 110 * 128 * 2 * 2.46e9 / 64,
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'out')
runs = {'CPU': '1.03_batch/tinystories_b{}', 'GPU': '1.04_cuda/b{}'}
print(f"{'':4s} {'batch':>5s} {'tokens/s':>10s} {'GFLOPS':>8s} {'MFU':>7s}")
for device, path in runs.items():
    for b in [1, 4, 16, 64]:
        last = list(csv.DictReader(open(os.path.join(out, path.format(b), 'log.csv'))))[-1]
        tokens_per_s = int(last['tokens']) / float(last['time'])
        flops = tokens_per_s * train
        print(f"{device:4s} {b:5d} {tokens_per_s:10.0f} {flops / 1e9:8.2f} {flops / peaks[device]:7.2%}")
    print(f"{device} float64 peak: {peaks[device] / 1e9:.0f} GFLOPS\n")
