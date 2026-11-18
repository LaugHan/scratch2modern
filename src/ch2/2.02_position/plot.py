"""
画 2.02 的图，并打印 notes 里用到的每一个数。
用法：python plot.py     先跑完 sweep.sh 和 eval.py；写出 curves.png、position.png、extrapolation.png
"""

import os
import csv

import numpy as np
import matplotlib.pyplot as plt

ch2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'out', 'ch2')
out = os.path.join(ch2, '2.02_position')

# 颜色按组分：改注意力的、加到 embedding 上的、不加；learned 和 2.01 一样用 C0
styles = {
    'rope': dict(color='C3', lw=1.8), 'alibi': dict(color='C1', lw=1.8),
    'learned': dict(color='C0'), 'sincos': dict(color='C2'), 'sin': dict(color='C2', ls='--'),
    'binary': dict(color='C4'), 'frac': dict(color='C5'), 'int': dict(color='C7'),
    'none': dict(color='black', ls=':'),
}
names = list(styles)


def read(path):
    rows = list(csv.DictReader(open(path)))
    return {k: np.array([float(r[k] or 'nan') for r in rows]) for k in rows[0]}  # 第 0 步没有 batch_loss


def run_dir(pos):
    return os.path.join(ch2, '2.01_scale', 'default') if pos == 'learned' else os.path.join(out, f'pos{pos}')


def smooth(y, k):
    # 滑动平均，窗口 k；每个位置只有几百个样本，不平均的话曲线很毛
    return np.convolve(y, np.ones(k) / k, mode='valid')


# ---------------- 训练曲线 ----------------

logs = {pos: read(os.path.join(run_dir(pos), 'log.csv')) for pos in names}
fig, ax = plt.subplots(figsize=(8, 4.5))
for pos in names:
    ax.plot(logs[pos]['step'][1:], logs[pos]['val_loss'][1:], label=pos, **styles[pos])
ax.set(xlabel='step', ylabel='val loss', ylim=(3.6, 5.0))
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(out, 'curves.png'), dpi=150)

print("val loss during training:")
for pos in sorted(names, key=lambda p: logs[p]['val_loss'][-1]):
    log = logs[pos]
    at = {s: log['val_loss'][list(log['step']).index(s)] for s in [1024, 4096, 13056]}
    final = read(os.path.join(run_dir(pos), 'final.csv'))
    print(f"  {pos:8s} step 1024 {at[1024]:.4f}, 4096 {at[4096]:.4f}, 13056 {at[13056]:.4f}, "
          f"final {final['val_loss'][0]:.4f} (train {final['train_loss'][0]:.4f}), time {final['time'][0]:.0f}s")

# ---------------- 每个位置的 loss（训练长度 1024） ----------------

loss = {T: read(os.path.join(out, f'position_loss_{T}.csv')) for T in [1024, 2048, 4096]}
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for pos in names:
    y = loss[1024][pos]
    axes[0].plot(np.arange(1, 1025), y, label=pos, **styles[pos])  # 开头几个位置，对数横轴
    # 所有模型用的是同一批窗口，曲线的起伏来自文本本身，所以减去 learned 再看差别
    diff = smooth(y - loss[1024]['learned'], 32)
    axes[1].plot(np.arange(len(diff)) + 16, diff, **styles[pos])
axes[0].set(xscale='log', xlabel='position + 1', ylabel='val loss', ylim=(3.2, 6.2))
axes[0].legend(fontsize=8)
axes[1].set(xlabel='position', ylabel='val loss − learned', ylim=(-0.15, 0.15))  # int 约 +0.4，在图外
plt.tight_layout()
plt.savefig(os.path.join(out, 'position.png'), dpi=150)

print("\nper-position val loss (length 1024):")
for pos in names:
    y = loss[1024][pos]
    print(f"  {pos:8s} position 0 {y[0]:.3f}, 1 {y[1]:.3f}, 1-15 {y[1:16].mean():.3f}, "
          f"16-127 {y[16:128].mean():.3f}, 128-511 {y[128:512].mean():.3f}, 512-1023 {y[512:].mean():.3f}")

# ---------------- 外推：窗口 4096 ----------------

k = 64
fig, ax = plt.subplots(figsize=(8, 4.5))
for pos in names:
    if pos in loss[4096]:  # learned 只有 1024 个位置，跑不了
        y = smooth(loss[4096][pos], k)
        ax.plot(np.arange(len(y)) + k // 2, y, label=pos, **styles[pos])
ax.axvline(1024, color='gray', lw=0.8, ls=':')
ax.set(xlabel='position', ylabel=f'val loss (moving average over {k} positions)', ylim=(3.0, 9.0))
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(out, 'extrapolation.png'), dpi=150)

print("\nextrapolation (mean val loss over positions):")
for pos in names:
    row = f"  {pos:8s} length 1024: {loss[1024][pos].mean():.4f}"
    for T in [2048, 4096]:
        if pos in loss[T]:
            y = loss[T][pos]
            row += f" | length {T}: first 1024 {y[:1024].mean():.4f}, beyond {y[1024:].mean():.4f}"
    print(row)

# ---------------- 外推时注意力落在哪（attention.py） ----------------

rows = list(csv.DictReader(open(os.path.join(out, 'attention_far.csv'))))
# 对照：如果注意力在能看到的 key 上均匀分布，第 q 个 query 分到距离 >= 1024 的比例是 (q - 1023) / (q + 1)
q = np.arange(3072, 4096)
print(f"\nattention mass at distance >= 1024 (queries 3072-4095, window 4096); "
      f"uniform would give {((q - 1023) / (q + 1)).mean():.3f}")
for pos in ['alibi', 'rope']:
    mass = np.array([[float(r[f'head{h}']) for h in range(8)] for r in rows if r['pos'] == pos])
    print(f"  {pos:6s} mean over heads per layer: " + ' '.join(f"{m:.3f}" for m in mass.mean(1))
          + f" | overall {mass.mean():.3f}, max head {mass.max():.3f}, min head {mass.min():.3f}")
