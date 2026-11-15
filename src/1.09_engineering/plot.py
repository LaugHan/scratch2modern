import os, csv, glob
import matplotlib.pyplot as plt

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "out", "1.09_engineering")
levels = ["baseline", "no_sync", "fused_adam", "sdpa", "compile", "cudagraph"]

def read(path):
    rows = list(csv.DictReader(open(path)))
    return {k: [float(r[k]) if r[k] else None for r in rows] for k in rows[0]}

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
xs = range(len(levels))
base = None
for i, name in enumerate(levels):
    final = [read(p) for p in sorted(glob.glob(os.path.join(out, name, "s*", "final.csv")))]
    ms = [f["ms_per_step"][0] for f in final]
    bpb = [f["val_bpb"][0] for f in final]
    m, b = sum(ms) / len(ms), sum(bpb) / len(bpb)
    base = base or m
    # 左图：每步耗时（3 个种子的均值，竖线是范围），柱子上标出相对 baseline 的加速比
    axes[0].bar(i, m, yerr=[[m - min(ms)], [max(ms) - m]], color=f"C{i}", capsize=4)
    axes[0].text(i, max(ms), f"{base / m:.1f}x", ha="center", va="bottom")
    # 中图：最终 val bpb，应该都一样
    axes[1].errorbar([i], [b], yerr=[[b - min(bpb)], [max(bpb) - b]], fmt="o", color=f"C{i}", capsize=4)
    # 右图：loss 曲线（种子 42），横轴是训练时间
    log = read(os.path.join(out, name, "s42", "log.csv"))
    axes[2].plot(log["time"], log["val_bpb"], color=f"C{i}", label=name)
    print(f"{name:10s}: {m:.2f} ms/step ({min(ms):.2f}-{max(ms):.2f}), {base / m:.2f}x, val bpb {b:.4f} ({min(bpb):.4f}-{max(bpb):.4f}), "
          f"first 128 steps {final[0]['first_interval'][0]:.1f}s, memory {final[0]['peak_memory_mb'][0]:.0f} MB")

for ax in axes[:2]:
    ax.set_xticks(xs, levels, rotation=45, ha="right")
axes[0].set(ylabel="ms per step (after the first 128 steps)", title="each level adds one speedup on top of the previous")
axes[1].set(ylabel="final val bits per byte", title="should not change (bar = seed range)")
axes[2].set(xlabel="training time (s)", ylabel="val bits per byte", ylim=(1.15, 1.5))
axes[2].legend()
plt.tight_layout()
plt.savefig(os.path.join(out, "compare.png"), dpi=150)
