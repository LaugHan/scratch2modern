import os, csv, glob
import matplotlib.pyplot as plt

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "out", "1.07_schedule")
schedules = ["constant", "linear", "cosine", "wsd"]
top = 1.6 # 纵轴上限，发散的点画在这里

def read(path):
    rows = list(csv.DictReader(open(path)))
    return {k: [float(r[k]) if r[k] else None for r in rows] for k in ["step", "lr", "val_bpb"]}

def runs(schedule, warmup=0, steps=4096, seed=42):
    # 峰值学习率 -> 日志
    found = {}
    for path in glob.glob(os.path.join(out, schedule, f"lr*_w{warmup}_n{steps}_s{seed}", "log.csv")):
        found[float(path.split(os.sep)[-2][2:].split("_")[0])] = read(path)
    return found

# 第一组：衰减的形状（不加 warmup）
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, name in enumerate(schedules):
    r = runs(name)
    done = sorted(lr for lr in r if r[lr]["step"][-1] == 4096)
    diverged = sorted(lr for lr in r if r[lr]["step"][-1] < 4096)
    best = min(done, key=lambda lr: r[lr]["val_bpb"][-1])
    seeds = [read(p)["val_bpb"][-1] for p in glob.glob(os.path.join(out, name, f"lr{best:g}_w0_n4096_s*", "log.csv"))]
    mean = sum(seeds) / len(seeds)

    # 左图：峰值学习率 -> 最终 loss；★ 是最佳（3 个种子的均值，竖线是范围），× 是发散
    axes[0].plot(done, [r[lr]["val_bpb"][-1] for lr in done], "o-", color=f"C{i}", label=name)
    axes[0].plot(diverged, [top] * len(diverged), "x", color=f"C{i}", clip_on=False)
    axes[0].errorbar([best], [mean], yerr=[[mean - min(seeds)], [max(seeds) - mean]], fmt="*", ms=14, color=f"C{i}", capsize=4)
    # 中图：学习率随步数的变化；右图：最佳峰值学习率下的 loss 曲线
    axes[1].plot(r[best]["step"], r[best]["lr"], color=f"C{i}", label=f"{name} (peak {best:g})")
    axes[2].plot(r[best]["step"], r[best]["val_bpb"], color=f"C{i}")
    print(f"{name:8s}: best peak lr {best:g}, val bpb {mean:.4f} (seeds {min(seeds):.4f}-{max(seeds):.4f})")

axes[0].set(xlabel="peak learning rate", ylabel="final val bits per byte", xscale="log", ylim=(1.15, top),
            title="★ best (mean of seeds, bar = range), × diverged")
lrs = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08]
axes[0].set_xticks(lrs, [f"{lr:g}" for lr in lrs], minor=False)
axes[0].set_xticks([], minor=True)
axes[1].set(xlabel="step", ylabel="learning rate")
axes[2].set(xlabel="step", ylabel="val bits per byte (best peak lr)", ylim=(1.15, top))
axes[0].legend()
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(out, "shape.png"), dpi=150)

# 第二组：warmup（衰减形状固定为 WSD）
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
lrs = [0.01, 0.02, 0.04, 0.08, 0.16]
for i, warmup in enumerate([0, 128, 512]):
    r = runs("wsd", warmup=warmup)
    done = sorted(lr for lr in r if lr in lrs and r[lr]["step"][-1] == 4096)
    diverged = sorted(lr for lr in r if lr in lrs and r[lr]["step"][-1] < 4096)
    axes[0].plot(done, [r[lr]["val_bpb"][-1] for lr in done], "o-", color=f"C{i}", label=f"warmup {warmup}")
    axes[0].plot(diverged, [top] * len(diverged), "x", color=f"C{i}", ms=10, clip_on=False)
    best = min(done, key=lambda lr: r[lr]["val_bpb"][-1])
    axes[1].plot(r[best]["step"], r[best]["val_bpb"], color=f"C{i}", label=f"warmup {warmup} (peak {best:g})")
    print(f"warmup {warmup:3d}: " + ", ".join(f"{lr:g}: {r[lr]['val_bpb'][-1]:.4f}" if lr in done else f"{lr:g}: diverged at {r[lr]['step'][-1]:.0f}" for lr in sorted(r) if lr in lrs))
axes[0].set(xlabel="peak learning rate", ylabel="final val bits per byte", xscale="log", ylim=(1.15, top), title="WSD, × diverged")
axes[0].set_xticks(lrs, [f"{lr:g}" for lr in lrs], minor=False)
axes[0].set_xticks([], minor=True)
axes[1].set(xlabel="step", ylabel="val bits per byte (best peak lr)", ylim=(1.15, top))
axes[0].legend()
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(out, "warmup.png"), dpi=150)

# 第三组：WSD 随时可停。constant 是主干，不同长度的 WSD 从主干上分出去衰减；cosine 每个长度从头训练
fig, ax = plt.subplots(figsize=(7, 4.5))
trunk = runs("constant")[0.01]
ax.plot(trunk["step"], trunk["val_bpb"], color="gray", lw=3, alpha=0.4, label="constant (trunk)")
for i, steps in enumerate([1024, 2048, 4096]):
    w = runs("wsd", steps=steps)[0.01]
    c = runs("cosine", steps=steps)[0.01]
    ax.plot(w["step"], w["val_bpb"], color=f"C{i}", label=f"wsd {steps} steps")
    ax.plot(c["step"], c["val_bpb"], color=f"C{i}", ls="--", label=f"cosine {steps} steps")
    print(f"{steps} steps: wsd {w['val_bpb'][-1]:.4f}, cosine {c['val_bpb'][-1]:.4f}")
ax.set(xlabel="step", ylabel="val bits per byte", ylim=(1.15, 1.45), title="one constant run, branch off and decay anytime")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(out, "anytime.png"), dpi=150)
