import os, csv
import matplotlib.pyplot as plt

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "out")
runs = ["1.01_microgpt", "1.02_torch"]

def smooth(xs, k=50):
    return [sum(xs[max(0, i - k + 1):i + 1]) / len(xs[max(0, i - k + 1):i + 1]) for i in range(len(xs))]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
for run in runs:
    rows = list(csv.DictReader(open(os.path.join(out, run, "log.csv"))))
    step = [int(r["step"]) for r in rows]
    loss = [float(r["loss"]) for r in rows]
    step_time = [float(r["step_time"]) * 1000 for r in rows]
    ax1.plot(step, smooth(loss), label=run)
    ax2.plot(step, smooth(step_time), label=run)
    print(f"{run}: final loss (avg 50) {sum(loss[-50:]) / 50:.4f}, step time {sum(step_time) / len(step_time):.1f} ms")

ax1.set(xlabel="step", ylabel="loss (moving avg 50)")
ax2.set(xlabel="step", ylabel="step time ms (moving avg 50)", yscale="log")
ax1.legend()
ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(out, "1.02_torch", "compare.png"), dpi=150)
