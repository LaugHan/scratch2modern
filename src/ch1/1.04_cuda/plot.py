import os, csv
import matplotlib.pyplot as plt

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "out")
batch_sizes = [1, 4, 16, 64]
runs = {"CPU": "1.03_batch/tinystories_b{}", "GPU": "1.04_cuda/b{}"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
for device, path in runs.items():
    speeds = []
    for i, b in enumerate(batch_sizes):
        rows = list(csv.DictReader(open(os.path.join(out, path.format(b), "log.csv"))))
        tokens = [int(r["tokens"]) for r in rows]
        time = [float(r["time"]) for r in rows]
        val = [float(r["val_loss"]) for r in rows]
        ax1.plot(time, val, color=f"C{i}", linestyle="-" if device == "GPU" else ":", label=f"{device} batch {b}")
        speeds.append(tokens[-1] / time[-1])
        print(f"{device} batch {b:2d}: val loss {val[-1]:.4f}, train time {time[-1]:.1f}s, tokens/s {speeds[-1]:.0f}")
    ax2.plot(batch_sizes, speeds, marker="o", label=device)

ax1.set(xlabel="train time (s)", ylabel="val loss", xscale="log")
ax2.set(xlabel="batch size", ylabel="tokens/s", xscale="log", yscale="log")
ax1.legend(fontsize=7)
ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(out, "1.04_cuda", "compare.png"), dpi=150)
