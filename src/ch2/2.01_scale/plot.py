"""
画 2.01 的图，并打印 notes 里用到的每一个数。
用法：python plot.py     读 out/ch2/2.01_scale/，写出 curves.png、sweeps.png
"""

import os
import csv

import matplotlib.pyplot as plt

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'out', 'ch2', '2.01_scale')


def read(run, name='log.csv'):
    rows = list(csv.DictReader(open(os.path.join(out, run, name))))
    return {k: [float(r[k]) if r[k] else None for r in rows] for k in rows[0]}


def at(log, step, key='val_loss'):
    return log[key][log['step'].index(step)]


# ---------------- 速度：挑模型大小 ----------------

speed = read('.', 'speed.csv')
print("speed (batch 32 x 1024):")
for i in range(len(speed['n_layer'])):
    tokens_per_hour = speed['tokens_per_s'][i] * 3600
    print(f"  {speed['n_layer'][i]:.0f} layers, width {speed['n_embd'][i]:.0f}: "
          f"{speed['params'][i] / 1e6:.0f}M ({speed['block_params'][i] / 1e6:.1f}M in blocks), "
          f"{speed['step_time'][i] * 1e3:.0f} ms/step, {speed['tokens_per_s'][i] / 1e3:.0f}K tokens/s, "
          f"{tokens_per_hour / 1e6:.0f}M tokens/hour")

# ---------------- 正式训练：初始化 0.02（基线）对比 0.08 ----------------

runs = [('default', 'init 0.02 (baseline)'), ('init0.08', 'init 0.08 (chapter 1)')]
logs = {run: read(run) for run, _ in runs}
decay_start = int(0.8 * logs['default']['step'][-1])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={'width_ratios': [2, 1]})
for i, (run, label) in enumerate(runs):
    log = logs[run]
    axes[0].plot(log['step'][1:], log['val_loss'][1:], color=f'C{i}', label=f'{label}, val')
    axes[0].plot(log['step'][1:], log['train_loss'][1:], color=f'C{i}', ls='--', lw=1, label=f'{label}, train')
axes[0].axvline(decay_start, color='gray', lw=0.8, ls=':')
axes[0].text(decay_start, 4.9, ' decay starts', color='gray', fontsize=8)
axes[0].set(xlabel='step', ylabel='loss (nats per token)', ylim=(3.6, 5.0),
            title='FineWeb-Edu, 8 layers, width 512, context 1024')
axes[0].legend(fontsize=8)
axes[1].plot(logs['default']['step'], logs['default']['lr'], color='gray')
axes[1].set(xlabel='step', ylabel='learning rate multiplier', title='schedule: 1% warmup, WSD')
plt.tight_layout()
plt.savefig(os.path.join(out, 'curves.png'), dpi=150)

print("\nformal runs (val loss):")
for step in [0, 1024, 2048, 4096, 8192, 11264, 13056, 16384]:  # 13056 是衰减前最后一次评估
    print(f"  step {step:5d}: " + ', '.join(f"{run} {at(logs[run], step):.4f}" for run, _ in runs))
for run, _ in runs:
    final = read(run, 'final.csv')
    gain = at(logs[run], 13056) - final['val_loss'][0]
    print(f"  {run}: final val {final['val_loss'][0]:.4f}, train {final['train_loss'][0]:.4f}, "
          f"decay gain {gain:.4f}, time {final['time'][0]:.0f}s, peak {final['peak_memory'][0] / 1024:.1f} GB")

# ---------------- 短实验：扫学习率、扫初始化（各 2048 步） ----------------

lrs = [0.005, 0.01, 0.02, 0.04]
lr_runs = {lr: read(f'lr{lr:g}_steps2048') for lr in lrs}  # 这几个跑的时候初始化还是 0.08
inits = [0.01, 0.02, 0.044, 0.08]
init_runs = {init: read(f'init{init:g}_steps2048') for init in inits if init != 0.08}
init_runs[0.08] = lr_runs[0.01]  # lr 0.01、init 0.08 就是扫学习率时的那一个

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(lrs, [lr_runs[lr]['val_loss'][-1] for lr in lrs], 'o-', color='C0')
axes[0].set(xscale='log', xlabel='learning rate (Muon; Adam uses half)', ylabel='val loss after 2048 steps',
            title='learning rate (init 0.08)')
axes[0].set_xticks(lrs, [f'{lr:g}' for lr in lrs])
axes[0].set_xticks([], minor=True)
colors = {0.02: 'C0', 0.08: 'C1', 0.01: 'C2', 0.044: 'C3'}  # 0.02、0.08 和 curves.png 里同色
for init in inits:
    log = init_runs[init]
    axes[1].plot(log['step'][1:], log['val_loss'][1:], color=colors[init], label=f'init {init:g}')
axes[1].set(xlabel='step', ylabel='val loss', ylim=(4.0, 7.0), title='initialization (lr 0.01)')
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(out, 'sweeps.png'), dpi=150)

print("\nlearning rate sweep (2048 steps, init 0.08):")
for lr in lrs:
    print(f"  lr {lr:g}: val loss {lr_runs[lr]['val_loss'][-1]:.4f}")
print("initialization sweep (2048 steps, lr 0.01):")
for init in inits:
    log = init_runs[init]
    print(f"  init {init:g}: initial {log['val_loss'][0]:.2f}, step 224 {at(log, 224):.4f}, "
          f"step 992 {at(log, 992):.4f}, step 2048 {log['val_loss'][-1]:.4f}")
