"""
第二章共用的部分：配置、数据、评估、优化器、训练循环、生成。
第二章只改模型，训练方式沿用第一章的结论，所以这个文件整章不改。
每一节的 train.py 只写模型，然后调用这里的函数。
"""

import os
import sys
import math
import time
import dataclasses
from dataclasses import dataclass

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
device = 'cuda'


# ---------------- 配置 ----------------

@dataclass
class TrainConfig:
    # 训练方式，整章不变；每节的 Config 继承它，再加上模型的字段
    data: str = 'fineweb_edu'
    block_size: int = 1024
    batch_size: int = 32
    steps: int = 16384
    lr: float = 0.01   # Muon 的学习率，Adam 用它的一半（1.11）；2.01 扫过，0.01 最好
    wd: float = 0.1    # 只加在 Muon 上（1.12）
    seed: int = 42


def parse_config(Config):
    # 命令行用 key=value 覆盖默认值，run 的名字只由改过的字段拼成
    overrides = dict(arg.split('=', 1) for arg in sys.argv[1:])
    defaults = Config()
    cfg = Config(**{k: type(getattr(defaults, k))(v) for k, v in overrides.items()})
    name = '_'.join(f'{k}{v}' for k, v in overrides.items()) or 'default'
    return cfg, name


def out_dir(train_file, run_name):
    # 输出写到 out/ch2/<这一节>/<run>/
    section = os.path.basename(os.path.dirname(os.path.abspath(train_file)))
    path = os.path.join(ROOT, 'out', 'ch2', section, run_name)
    os.makedirs(path, exist_ok=True)
    return path


# ---------------- 数据 ----------------

# GPT-2 的分词器；词表 50257，补到 64 的倍数，矩阵乘法更快（1.09）
enc = tiktoken.get_encoding('gpt2')
EOT = enc.eot_token
vocab_size = 50304


def load_tokens(name, split):
    # prepare.py 写出的 uint16 token，转成 int32 整份放进显存
    path = os.path.join(ROOT, 'data', 'tok', f'{name}_{split}.bin')
    return torch.from_numpy(np.fromfile(path, dtype=np.uint16).astype(np.int32)).to(device)


def get_batch(data, batch_size, block_size):
    # 随机取 batch_size 个窗口；目标是同一个窗口往后错一位
    starts = torch.randint(len(data) - block_size, (batch_size,), device=device)
    offsets = starts[:, None] + torch.arange(block_size, device=device)
    return data[offsets].long(), data[offsets + 1].long()


def fixed_windows(data, n, block_size):
    # 评估用的固定窗口：从开头起连续切 n 个
    x = data[:n * block_size].view(n, block_size)
    y = data[1:n * block_size + 1].view(n, block_size)
    return x.long(), y.long()


# ---------------- 训练 ----------------

def autocast():
    return torch.autocast('cuda', dtype=torch.bfloat16)  # bf16 混合精度（1.08）


def loss_fn(model, x, y, reduction='mean'):
    with autocast():
        logits = model(x)
    return F.cross_entropy(logits   .float().view(-1, logits.size(-1)), y.view(-1), reduction=reduction)


@torch.no_grad()
def eval_loss(model, x, y, batch_size=16):
    # 固定窗口上每个 token 的平均 loss（nats）。分词器整章固定，所以直接比 loss，不换算成 bpb
    # 求和用 fp64，否则累加误差很大（1.12）
    nats = sum(loss_fn(model, x[i:i + batch_size], y[i:i + batch_size], 'sum').double()
               for i in range(0, len(x), batch_size))
    return nats.item() / y.numel()


def make_optimizers(model, cfg):
    # Transformer 内部的二维矩阵用 Muon，embedding、输出层和一维参数用 Adam（1.06、1.11）
    # Adam 不加 weight decay：torch.optim.Adam 的 weight decay 是 L2 的写法，会让训练失败（1.10）
    matrices = [p for n, p in model.named_parameters() if n.startswith('blocks.') and p.ndim == 2]
    others = [p for n, p in model.named_parameters() if not (n.startswith('blocks.') and p.ndim == 2)]
    return [torch.optim.Muon(matrices, lr=cfg.lr, weight_decay=cfg.wd),
            torch.optim.Adam(others, lr=cfg.lr / 2, betas=(0.85, 0.99), fused=True)]


def wsd(step, num_steps):
    # 前 1% 线性 warmup，之后保持峰值，最后 20% 线性降到 0（1.07、1.12）
    warmup = max(num_steps // 100, 1)
    return min((step + 1) / warmup, 1.0, (num_steps - step) / (0.2 * num_steps))


def train(model, cfg, path, num_evals=64, n_eval_windows=512):
    train_data, val_data = load_tokens(cfg.data, 'train'), load_tokens(cfg.data, 'val')
    val_x, val_y = fixed_windows(val_data, n_eval_windows, cfg.block_size)
    train_x, train_y = fixed_windows(train_data, n_eval_windows, cfg.block_size)  # 看有没有记住训练数据（1.10）
    print(f"train tokens: {len(train_data):,}, val tokens: {len(val_data):,}")

    optimizers = make_optimizers(model, cfg)
    schedulers = [torch.optim.lr_scheduler.LambdaLR(o, lambda s: wsd(s, cfg.steps)) for o in optimizers]
    compiled = torch.compile(model)  # 1.09

    log = open(os.path.join(path, 'log.csv'), 'w')
    # batch_loss：两次评估之间训练 batch 的平均；train_loss、val_loss：训练集、验证集固定窗口上的 loss
    log.write('step,tokens,time,lr,batch_loss,train_loss,val_loss\n')
    log.write(f"0,0,0,{wsd(0, cfg.steps):.6g},,"
              f"{eval_loss(compiled, train_x, train_y):.6f},{eval_loss(compiled, val_x, val_y):.6f}\n")

    # 计时只算训练，不算评估；读时间前先等 GPU 把排队的活干完（1.09）
    eval_every = cfg.steps // num_evals
    train_time, loss_sum = 0.0, torch.zeros((), device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(cfg.steps):
        x, y = get_batch(train_data, cfg.batch_size, cfg.block_size)
        loss = loss_fn(compiled, x, y)
        loss.backward()
        for opt, sched in zip(optimizers, schedulers):
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
        loss_sum += loss.detach()  # 留在 GPU 上累加，不同步（1.09）
        if step + 1 == int(0.8 * cfg.steps):
            # 衰减开始前存一份完整的 checkpoint：WSD 可以从这里接着训练得更长，再重新衰减（1.07）
            torch.save({'step': step + 1, 'model': model.state_dict(),
                        'optimizers': [o.state_dict() for o in optimizers]},
                       os.path.join(path, 'ckpt_before_decay.pt'))

        if (step + 1) % eval_every == 0:
            torch.cuda.synchronize()
            train_time += time.perf_counter() - t0
            batch_loss = loss_sum.item() / eval_every
            if not math.isfinite(batch_loss):
                print(f"\ndiverged before step {step + 1}")
                return None
            tokens = (step + 1) * cfg.batch_size * cfg.block_size
            train_loss, val_loss = eval_loss(compiled, train_x, train_y), eval_loss(compiled, val_x, val_y)
            log.write(f"{step + 1},{tokens},{train_time:.2f},{wsd(step, cfg.steps):.6g},"
                      f"{batch_loss:.6f},{train_loss:.6f},{val_loss:.6f}\n")
            log.flush()
            print(f"step {step + 1:6d} / {cfg.steps} | batch loss {batch_loss:.4f} | val loss {val_loss:.4f} "
                  f"| {train_time:.0f}s", flush=True)
            loss_sum.zero_()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
    log.close()

    peak_memory = torch.cuda.max_memory_allocated() / 2**20  # 包括显存里的训练数据
    print(f"val loss {val_loss:.4f}, time {train_time:.0f}s, peak memory {peak_memory:.0f} MB")
    with open(os.path.join(path, 'final.csv'), 'w') as f:
        f.write('train_loss,val_loss,time,peak_memory\n')
        f.write(f"{train_loss:.6f},{val_loss:.6f},{train_time:.2f},{peak_memory:.1f}\n")
    torch.save(model.state_dict(), os.path.join(path, 'model.pt'))
    return val_loss


# ---------------- 生成 ----------------

@torch.no_grad()
def generate(model, prompt, max_tokens=256, temperature=0.8):
    idx = torch.tensor([[EOT] + enc.encode_ordinary(prompt)], device=device)
    for _ in range(max_tokens):
        with autocast():
            logits = model(idx[:, -model.block_size:])[0, -1, :EOT + 1]  # 补出来的那些 id 不采样
        token = torch.multinomial(F.softmax(logits.float() / temperature, dim=-1), 1)
        if token.item() == EOT:
            break
        idx = torch.cat([idx, token[None]], dim=1)
    return enc.decode(idx[0, 1:].tolist())


def write_samples(model, path, prompts):
    with open(os.path.join(path, 'samples.txt'), 'w') as f:
        for prompt in prompts:
            sample = generate(model, prompt)
            print(f"--- {sample}\n")
            f.write(sample + '\n\n')
