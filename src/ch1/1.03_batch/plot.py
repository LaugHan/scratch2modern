import os, csv
import matplotlib.pyplot as plt

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "out", "1.03_batch")
datasets = ["names", "tinystories"]
batch_sizes = [1, 4, 16, 64]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for row, dataset in zip(axes, datasets):
    for b in batch_sizes:
        rows = list(csv.DictReader(open(os.path.join(out, f"{dataset}_b{b}", "log.csv"))))
        tokens = [int(r["tokens"]) for r in rows]
        time = [float(r["time"]) for r in rows]
        val = [float(r["val_loss"]) for r in rows]
        row[0].plot(tokens, val, label=f"batch {b}")
        row[1].plot(time, val, label=f"batch {b}")
        print(f"{dataset} batch {b:2d}: val loss {val[-1]:.4f}, train time {time[-1]:.1f}s, tokens/s {tokens[-1] / time[-1]:.0f}")
    row[0].set(title=dataset, xlabel="tokens", ylabel="val loss")
    row[1].set(title=dataset, xlabel="train time (s)", ylabel="val loss", xscale="log")
    row[0].legend()

plt.tight_layout()
plt.savefig(os.path.join(out, "compare.png"), dpi=150)
