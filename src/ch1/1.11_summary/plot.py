import os, csv, math
import matplotlib.pyplot as plt

out_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "out")
out = os.path.join(out_root, "1.11_summary")

def last(path):
    # 最终的 (训练时间, val bpb)。1.03、1.04 记的是每个字符的 loss（nats），char 每个 token 约 1 字节，换算成 bpb
    r = list(csv.DictReader(open(os.path.join(out_root, path, "log.csv"))))[-1]
    bpb = float(r["val_bpb"]) if "val_bpb" in r else float(r["val_loss"]) / math.log(2)
    return float(r["time"]), bpb

# 阶梯：每个点是一种标志性的改进，在上一个点的基础上多改一样东西；都是种子 42。每个点改了什么见 notes.md 的图注
points = [
    "1.03_batch/tinystories_b1",           # ① CPU，char，宽度 16，batch 1
    "1.04_cuda/b1",                        # ② 搬到 GPU
    "1.05_tokenizer/char",                 # ③ batch 16，训练 4096 步
    "1.08_precision/fp64/s42",             # ④ bpe1024 + 宽度 48
    "1.08_precision/bf16_autocast/s42",    # ⑤ bf16 混合精度
    "1.11_summary/adam_linear_s42",        # ⑥ 编译 + CUDA Graph 等（现在）
]
marks = "①②③④⑤⑥"
xy = [last(p) for p in points]
fig, ax = plt.subplots(figsize=(8, 5.5))
for i, (t, b) in enumerate(xy):
    print(f"{marks[i]} {points[i]:36s} time {t:6.1f}s, val bpb {b:.4f}")
    ax.scatter([t], [b], s=70, color="C0", zorder=3)
    ax.annotate(marks[i], (t, b), xytext=(8, 6), textcoords="offset points", fontsize=14)
    if i > 0:
        ax.annotate("", xy=(t, b), xytext=xy[i - 1], arrowprops=dict(arrowstyle="->", color="gray", lw=1, shrinkA=6, shrinkB=6))
ax.set(xscale="log", xlabel="training time (s), log scale", ylabel="final val bits per byte", xlim=(2.5, 150), ylim=(1.0, 2.8))
ticks = [3, 5, 10, 20, 50, 100]
ax.set_xticks(ticks, [str(t) for t in ticks])
ax.set_xticks([], minor=True)
plt.tight_layout()
plt.savefig(os.path.join(out, "ladder.png"), dpi=150)

# 现在的配置换上 Muon、WSD：3 个种子
for opt in ["adam", "muon"]:
    for sched in ["linear", "wsd"]:
        finals = [list(csv.DictReader(open(os.path.join(out, f"{opt}_{sched}_s{s}", "final.csv"))))[0] for s in (42, 43, 44)]
        v = [float(f["val_bpb"]) for f in finals]; t = [float(f["time"]) for f in finals]
        print(f"{opt:5s} {sched:7s}: val bpb {sum(v) / 3:.4f} ({min(v):.4f}-{max(v):.4f}), time {sum(t) / 3:.1f}s, peak memory {finals[0]['peak_memory']} MB")
