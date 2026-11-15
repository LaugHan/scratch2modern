import os, csv, glob
import matplotlib.pyplot as plt

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "out", "1.08_precision")
precisions = ["fp64", "fp32", "tf32", "fp16_scaler", "bf16_autocast", "fp8", "bf16_pure", "fp16_noscaler"] # 基线 | 论文路线 | 消融

def read(path):
    rows = list(csv.DictReader(open(path)))
    return {k: [float(r[k]) if r[k] else None for r in rows] for k in rows[0]}

fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
xs = range(len(precisions))
for i, name in enumerate(precisions):
    final = [read(p) for p in sorted(glob.glob(os.path.join(out, name, "s*", "final.csv")))]
    fp64 = [f["val_bpb_fp64"][0] for f in final]
    own = [f["val_bpb"][0] for f in final]
    mean = sum(fp64) / len(fp64)
    # 第 1 张：最终 loss。实心是用 fp64 评估（3 个种子的均值，竖线是范围），空心是用训练时的精度评估
    axes[0].errorbar([i], [mean], yerr=[[mean - min(fp64)], [max(fp64) - mean]], fmt="o", color=f"C{i}", capsize=4)
    axes[0].plot([i], [sum(own) / len(own)], "o", mfc="none", color=f"C{i}")
    # 第 2、3 张：训练时间、显存峰值
    axes[1].bar(i, sum(f["time"][0] for f in final) / len(final), color=f"C{i}")
    axes[2].bar(i, final[0]["peak_memory_mb"][0], color=f"C{i}")
    # 第 4 张：loss 曲线（种子 42，用训练时的精度评估）
    log = read(os.path.join(out, name, "s42", "log.csv"))
    axes[3].plot(log["step"], log["val_bpb"], color=f"C{i}", label=name)
    print(f"{name:14s}: val bpb {mean:.4f} (seeds {min(fp64):.4f}-{max(fp64):.4f}), in own precision {sum(own) / len(own):.4f}, "
          f"time {sum(f['time'][0] for f in final) / len(final):.1f}s, memory {final[0]['peak_memory_mb'][0]:.0f} MB")

for ax in axes[:3]:
    ax.set_xticks(xs, precisions, rotation=45, ha="right")
    for x in [2.5, 5.5]: # 三组之间的分隔线
        ax.axvline(x, color="gray", ls=":")
axes[0].set(ylabel="final val bits per byte", title="filled: eval in fp64 (bar = seeds)\nhollow: eval in training precision")
axes[1].set(ylabel="training time (s)")
axes[2].set(ylabel="peak GPU memory (MB)")
axes[3].set(xlabel="step", ylabel="val bits per byte", ylim=(1.15, 1.5))
axes[3].legend()
plt.tight_layout()
plt.savefig(os.path.join(out, "compare.png"), dpi=150)
