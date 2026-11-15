import os, csv
import matplotlib.pyplot as plt

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "out", "1.05_tokenizer")
tokenizers = ["char", "byte", "bpe512", "bpe1024", "bpe4096"]
groups = {"context": "same context: ~1KB of text", "params": "same params: ~138k"}

def read(path):
    rows = list(csv.DictReader(open(path)))
    return [int(r["step"]) for r in rows], [float(r["val_bpb"]) for r in rows]

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, (group, title) in zip(axes, groups.items()):
    for i, name in enumerate(tokenizers):
        step, bpb = read(os.path.join(out, name, "log.csv"))
        ax.plot(step, bpb, color=f"C{i}", ls="--", alpha=0.4)
        step, bpb = read(os.path.join(out, "analysis", group, name, "log.csv"))
        ax.plot(step, bpb, color=f"C{i}", label=name)
        print(f"{group:8s} {name:8s}: val bpb {bpb[-1]:.4f}")
    ax.set(title=title + " (dashed: 1.05)", xlabel="step", ylim=(1, 4))
axes[0].set_ylabel("val bits per byte")
axes[0].legend()
plt.tight_layout()
plt.savefig(os.path.join(out, "analysis", "compare.png"), dpi=150)
