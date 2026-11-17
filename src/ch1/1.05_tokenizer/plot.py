import os, csv
import matplotlib.pyplot as plt

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "out", "1.05_tokenizer")
tokenizers = ["char", "byte", "bpe512", "bpe1024", "bpe4096", "gpt2"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for name in tokenizers:
    rows = list(csv.DictReader(open(os.path.join(out, name, "log.csv"))))
    step = [int(r["step"]) for r in rows]
    text = [int(r["text_bytes"]) / 1e6 for r in rows]
    time = [float(r["time"]) for r in rows]
    bpb = [float(r["val_bpb"]) for r in rows]
    axes[0].plot(step, bpb, label=name)
    axes[1].plot(text, bpb, label=name)
    axes[2].plot(time, bpb, label=name)
    print(f"{name:8s}: val bpb {bpb[-1]:.4f}, text seen {text[-1]:.1f} MB, train time {time[-1]:.1f}s")

axes[0].set(xlabel="step (same compute per step)", ylabel="val bits per byte")
axes[1].set(xlabel="training text seen (MB)", ylabel="val bits per byte")
axes[2].set(xlabel="train time (s)", ylabel="val bits per byte", xscale="log")
for ax in axes:
    ax.set_ylim(top=4)
axes[0].legend()
plt.tight_layout()
plt.savefig(os.path.join(out, "compare.png"), dpi=150)
