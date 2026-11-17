import os, csv
import matplotlib.pyplot as plt

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "out", "1.01_microgpt")
rows = list(csv.DictReader(open(os.path.join(out_dir, "log.csv"))))
step = [int(r["step"]) for r in rows]
loss = [float(r["loss"]) for r in rows]

k = 50  # moving average window
smooth = [sum(loss[max(0, i - k + 1):i + 1]) / len(loss[max(0, i - k + 1):i + 1]) for i in range(len(loss))]

plt.plot(step, loss, alpha=0.3, label="loss")
plt.plot(step, smooth, label=f"moving avg ({k})")
plt.xlabel("step")
plt.ylabel("loss")
plt.legend()
plt.savefig(os.path.join(out_dir, "loss.png"), dpi=150)
