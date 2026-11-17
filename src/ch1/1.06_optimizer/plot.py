import os, csv, glob
import matplotlib.pyplot as plt

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "out", "1.06_optimizer")
optimizers = ["sgd", "momentum", "adam", "adamw", "lion", "muon"]
top = 2.0 # 纵轴上限，发散的点画在这里

def read(path):
    rows = list(csv.DictReader(open(path)))
    return [int(r["step"]) for r in rows], [float(r["time"]) for r in rows], [float(r["val_bpb"]) for r in rows]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, name in enumerate(optimizers):
    runs = {} # lr -> 种子 42 的日志
    for path in glob.glob(os.path.join(out, name, "lr*_s42", "log.csv")):
        runs[float(path.split(os.sep)[-2][2:].split("_")[0])] = read(path)
    done = sorted(lr for lr in runs if runs[lr][0][-1] == 4096)
    diverged = sorted(lr for lr in runs if runs[lr][0][-1] < 4096)
    best = min(done, key=lambda lr: runs[lr][2][-1])

    # 左图：学习率 -> 最终 loss；星号是最佳学习率，误差棒是 3 个种子的范围，× 是发散
    axes[0].plot(done, [runs[lr][2][-1] for lr in done], "o-", color=f"C{i}", label=name)
    axes[0].plot(diverged, [top] * len(diverged), "x", color=f"C{i}", clip_on=False)
    seeds = [read(p)[2][-1] for p in glob.glob(os.path.join(out, name, f"lr{best:g}_s*", "log.csv"))]
    mean = sum(seeds) / len(seeds)
    axes[0].errorbar([best], [mean], yerr=[[mean - min(seeds)], [max(seeds) - mean]], fmt="*", ms=14, color=f"C{i}", capsize=4)

    # 中图和右图：最佳学习率下的 loss 曲线（种子 42）
    step, time, bpb = runs[best]
    axes[1].plot(step, bpb, color=f"C{i}", label=f"{name} (lr {best:g})")
    axes[2].plot(time, bpb, color=f"C{i}")
    print(f"{name:8s}: best lr {best:g}, val bpb {mean:.4f} (seeds {min(seeds):.4f}-{max(seeds):.4f}), train time {time[-1]:.1f}s")

axes[0].set(xlabel="learning rate", ylabel="final val bits per byte", xscale="log", ylim=(1.1, top),
            title="★ best lr (mean of 3 seeds, bar = range), × diverged")
axes[1].set(xlabel="step", ylabel="val bits per byte (best lr)", ylim=(1.1, top))
axes[2].set(xlabel="train time (s)", ylabel="val bits per byte (best lr)", ylim=(1.1, top))
axes[0].legend()
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(out, "compare.png"), dpi=150)
