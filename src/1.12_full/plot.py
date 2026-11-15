import os, csv
import matplotlib.pyplot as plt

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "out", "1.12_full")

def read(run):
    rows = list(csv.DictReader(open(os.path.join(out, run, "log.csv"))))
    return {k: [float(r[k]) if r[k] else None for r in rows] for k in rows[0]}

# 正式训练：Muon 不加 / 加 weight decay。左：验证集（实线）和训练集固定窗口（虚线）的 bpb；右：学习率倍数
runs = [(run, label) for run, label in [("lr0.01_wd0_n65536_s42", "Muon, no weight decay"), ("lr0.01_wd0.1_n65536_s42", "Muon, weight decay 0.1")]
        if os.path.exists(os.path.join(out, run, "log.csv"))]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [2, 1]})
for i, (run, label) in enumerate(runs):
    log = read(run)
    axes[0].plot(log["step"][1:], log["val_bpb"][1:], color=f"C{i}", label=f"{label}, val")
    axes[0].plot(log["step"][1:], log["train_bpb"][1:], color=f"C{i}", ls="--", label=f"{label}, train")
    best = min(range(1, len(log["val_bpb"])), key=lambda j: log["val_bpb"][j])
    print(f"{label:24s}: final val bpb {log['val_bpb'][-1]:.4f}, best {log['val_bpb'][best]:.4f} at step {log['step'][best]:.0f}, time {log['time'][-1]:.0f}s")
axes[0].set(xlabel="step", ylabel="bits per byte", ylim=(0.55, 1.2), title="TinyStories, 6 layers, width 256")
axes[0].legend(fontsize=8)
log = read(runs[0][0])
axes[1].plot(log["step"], log["lr"], color="gray")
axes[1].set(xlabel="step", ylabel="learning rate multiplier", title="schedule: 1% warmup, WSD")
plt.tight_layout()
plt.savefig(os.path.join(out, "curves.png"), dpi=150)

# 扫学习率（2048 步，不加 weight decay）
for lr in ["0.005", "0.01", "0.02", "0.04", "0.08"]:
    f = os.path.join(out, f"lr{lr}_wd0_n2048_s42", "final.csv")
    print(f"lr {lr:5s} (2048 steps): " + (f"val bpb {float(list(csv.DictReader(open(f)))[0]['val_bpb']):.4f}" if os.path.exists(f) else "diverged"))
