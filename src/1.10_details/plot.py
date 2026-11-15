import os, re, csv, glob
import matplotlib.pyplot as plt

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "out", "1.10_details")
top = 1.32 # 纵轴上限，发散或差得太远的点画在这里

def read(path):
    rows = list(csv.DictReader(open(path)))
    return {k: [float(r[k]) if r[k] else None for r in rows] for k in rows[0]}

def runs(init="0.08", wd="none", clip="none", dropout="0", lr="0.01", suffix="", key="val_bpb"):
    # 所有种子的最终 val bpb（或 train bpb）；发散的记为 None。suffix 为 "_f20" 时是只用 1/20 训练集的那组
    result = []
    name = f"init{init}_wd{wd}_clip{clip}_drop{dropout}_lr{lr}_s"
    for d in sorted(glob.glob(os.path.join(out, name + "*"))):
        if not re.fullmatch(re.escape(name) + r"\d+" + re.escape(suffix), os.path.basename(d)): continue
        f = os.path.join(d, "final.csv")
        result.append(read(f)[key][0] if os.path.exists(f) else None)
    return result

def point(ax, x, vals, color, label=None):
    # 均值和种子范围；发散的画成 ×
    done = [v for v in vals if v is not None]
    if done:
        m = sum(done) / len(done)
        ax.errorbar([x], [min(m, top)], yerr=[[m - min(done)], [max(done) - m]], fmt="o", color=color, capsize=4, label=label)
    for v in vals:
        if v is None or v > top:
            ax.plot([x], [top], "x", color=color, ms=10, clip_on=False)
    return done

fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), gridspec_kw={"width_ratios": [1.6, 1, 1, 1]})

# 一、初始化：每种初始化在不同学习率下的结果（种子 42），★ 是最佳学习率上 3 个种子的均值
ax = axes[0]
lrs = ["0.005", "0.01", "0.02", "0.04"]
for i, init in enumerate(["xavier", "kaiming", "0.02", "0.08", "zero_proj"]):
    xs = [lr for lr in lrs if runs(init=init, lr=lr) and runs(init=init, lr=lr)[0] is not None] # 0.08 在学习率 0.04 时发散（梯度裁剪那组跑的）
    ys = [runs(init=init, lr=lr)[0] for lr in xs]
    ax.plot([float(x) for x in xs], ys, "o-", color=f"C{i}", label=init, alpha=0.6)
    best = min(xs, key=lambda lr: runs(init=init, lr=lr)[0])
    seeds = runs(init=init, lr=best)
    m = sum(seeds) / len(seeds)
    ax.errorbar([float(best)], [m], yerr=[[m - min(seeds)], [max(seeds) - m]], fmt="*", ms=14, color=f"C{i}", capsize=4)
    print(f"init {init:9s}: best lr {best}, val bpb {m:.4f} ({min(seeds):.4f}-{max(seeds):.4f}), lr sweep " + ", ".join(f"{lr}: {runs(init=init, lr=lr)[0]:.4f}" for lr in xs))
print(f"init zero     : val bpb {runs(init='zero')[0]:.4f}")
ax.set(xscale="log", xlabel="learning rate", ylabel="final val bits per byte", ylim=(1.165, 1.25), title="init (zero init: 3.16, off the chart)")
ax.set_xticks([float(lr) for lr in lrs], lrs, minor=False)
ax.set_xticks([], minor=True)
ax.legend()

# 二、weight decay
ax = axes[1]
for i, wd in enumerate(["none", "l2", "adamw", "adamw_noemb"]):
    done = point(ax, i, runs(wd=wd), f"C{i}")
    print(f"wd {wd:12s}: val bpb {sum(done) / len(done):.4f} ({min(done):.4f}-{max(done):.4f})")
ax.set_xticks(range(4), ["none", "L2 0.1", "AdamW 0.1", "AdamW 0.1\nno emb"])
ax.set(ylim=(1.165, top), title="weight decay (L2: 2.42, off the chart)")

# 三、梯度裁剪：学习率 0.01 和 0.04
ax = axes[2]
for i, (lr, clip) in enumerate([("0.01", "none"), ("0.01", "1.0"), ("0.04", "none"), ("0.04", "1.0")]):
    vals = runs(clip=clip, lr=lr)
    done = point(ax, i, vals, f"C{i}")
    print(f"lr {lr} clip {clip:4s}: " + (f"val bpb {sum(done) / len(done):.4f}, " if done else "") + f"{vals.count(None)} of {len(vals)} diverged")
ax.set_xticks(range(4), ["lr 0.01\nno clip", "lr 0.01\nclip 1.0", "lr 0.04\nno clip", "lr 0.04\nclip 1.0"])
ax.set(ylim=(1.165, top), title="gradient clipping (× diverged)")

# 四、dropout
ax = axes[3]
for i, dropout in enumerate(["0", "0.1", "0.1late"]):
    done = point(ax, i, runs(dropout=dropout), f"C{i}")
    print(f"dropout {dropout:7s}: val bpb {sum(done) / len(done):.4f} ({min(done):.4f}-{max(done):.4f})")
ax.set_xticks(range(3), ["0", "0.1", "0.1 from\nhalfway"])
ax.set(ylim=(1.165, top), title="dropout")
plt.tight_layout()
plt.savefig(os.path.join(out, "compare.png"), dpi=150)

# 曲线：梯度的总长度（学习率 0.01 和 0.04，不裁剪），dropout 的训练集和验证集 bpb
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for i, (lr, seed) in enumerate([("0.01", 42), ("0.04", 43)]):
    log = read(os.path.join(out, f"init0.08_wdnone_clipnone_drop0_lr{lr}_s{seed}", "log.csv"))
    axes[0].plot(log["step"][1:], log["grad_norm_mean"][1:], color=f"C{i}", label=f"lr {lr}, mean of 128 steps")
    axes[0].plot(log["step"][1:], log["grad_norm_max"][1:], color=f"C{i}", ls="--", label=f"lr {lr}, max of 128 steps")
axes[0].axhline(1.0, color="gray", ls=":", label="clip threshold 1.0")
axes[0].set(xlabel="step", ylabel="gradient norm", yscale="log", title="gradient norm, no clipping")
axes[0].legend(fontsize=8)
for i, dropout in enumerate(["0", "0.1", "0.1late"]):
    log = read(os.path.join(out, f"init0.08_wdnone_clipnone_drop{dropout}_lr0.01_s42", "log.csv"))
    axes[1].plot(log["step"], log["val_bpb"], color=f"C{i}", label=f"dropout {dropout}, val")
    axes[1].plot(log["step"], log["train_bpb"], color=f"C{i}", ls="--", label=f"dropout {dropout}, train")
axes[1].set(xlabel="step", ylabel="bits per byte", ylim=(1.15, 1.5), title="train (fixed windows, no dropout) vs val")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(out, "curves.png"), dpi=150)

# 过拟合：训练集只用 1/20。左：各做法训练集（空心）和验证集（实心）的最终 bpb；右：训练集和验证集的 bpb 曲线
configs = [("none", dict()), ("AdamW 0.1", dict(wd="adamw")), ("AdamW 0.1\nno emb", dict(wd="adamw_noemb")),
           ("clip 1.0", dict(clip="1.0")), ("dropout 0.1", dict(dropout="0.1")), ("dropout 0.1\nfrom halfway", dict(dropout="0.1late"))]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
ax = axes[0]
for i, (label, kw) in enumerate(configs):
    val, train = runs(suffix="_f20", **kw), runs(suffix="_f20", key="train_bpb", **kw)
    for vals, fmt in [(val, "o"), (train, "o")]:
        m = sum(vals) / len(vals)
        ax.errorbar([i], [m], yerr=[[m - min(vals)], [max(vals) - m]], fmt=fmt, color=f"C{i}", mfc=f"C{i}" if vals is val else "white", capsize=4)
    mv, mt = sum(val) / len(val), sum(train) / len(train)
    ax.plot([i, i], [mt, mv], color=f"C{i}", alpha=0.3)
    print(f"1/20 {label.replace(chr(10), ' '):24s}: train {mt:.4f}, val {mv:.4f} ({min(val):.4f}-{max(val):.4f}), gap {mv - mt:.3f}")
ax.axhline(sum(runs()) / 3, color="gray", ls=":", label="full data, none: val")
ax.set_xticks(range(len(configs)), [c[0] for c in configs])
ax.set(ylabel="final bits per byte", title="1/20 of training data (filled: val, hollow: train)")
ax.legend()
ax = axes[1]
for i, (label, kw) in [(0, configs[0]), (1, configs[1]), (4, configs[4])]:
    log = read(os.path.join(out, os.path.basename(glob.glob(os.path.join(out, f"init0.08_wd{kw.get('wd', 'none')}_clipnone_drop{kw.get('dropout', '0')}_lr0.01_s42_f20"))[0]), "log.csv"))
    ax.plot(log["step"], log["val_bpb"], color=f"C{i}", label=f"{label.replace(chr(10), ' ')}, val")
    ax.plot(log["step"], log["train_bpb"], color=f"C{i}", ls="--", label=f"{label.replace(chr(10), ' ')}, train")
ax.set(xlabel="step", ylabel="bits per byte", ylim=(1.0, 1.6), title="1/20 of training data, seed 42")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(out, "overfit.png"), dpi=150)
