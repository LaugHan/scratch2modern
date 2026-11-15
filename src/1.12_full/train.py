"""
microgpt in PyTorch (GPU)，第一章的收尾：用第一章的全部配置，在完整的 TinyStories 训练集上训练一个大一点的模型。
模型：6 层，宽度 256，4 个头，上下文 256，tokenizer 为 bpe1024。
训练：Muon（weight decay 0.1）+ Adam，前 1% warmup，WSD；bf16 混合精度，编译 + CUDA Graph；batch 64，65536 步。
先运行 prepare.py，它写出 data/tok/bpe1024_full.pt
用法：python train.py [lr=0.02] [wd=0.1] [steps=65536] [seed=42]     扫学习率时用较少的步数，例如 lr=0.04 steps=2048

Based on microgpt by @karpathy
"""

import os       # os.path.join
import sys      # sys.argv
import math     # math.log
import random   # random.seed, random.choices
import time     # time.perf_counter
import torch
import torch.nn.functional as F # F.scaled_dot_product_attention
args = dict(lr='0.02', wd='0.1', steps='65536', seed='42') # 默认值
args.update(a.split('=') for a in sys.argv[1:])
learning_rate = float(args['lr']) # Muon 的学习率；Adam 用它的一半（1.11 里 Muon 0.02、Adam 0.01）
weight_decay = float(args['wd']) # 只加在 Muon 上：不加的话，参数会越长越大，注意力的 logit 跟着变大，训练变得不稳定
num_steps = int(args['steps'])
seed = int(args['seed'])
random.seed(seed) # Let there be order among chaos
torch.manual_seed(seed)
autocast = lambda: torch.autocast('cuda', dtype=torch.bfloat16) # 1.08：bf16 混合精度
device = 'cuda:0'

# 项目路径：从 data/tok/ 读数据，日志写到 out/<这一节>/lr<lr>_wd<wd>_n<steps>_s<seed>/
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
run_name = f"lr{learning_rate:g}_wd{weight_decay:g}_n{num_steps}_s{seed}"
out_dir = os.path.join(root, 'out', os.path.basename(os.path.dirname(os.path.abspath(__file__))), run_name)
os.makedirs(out_dir, exist_ok=True)

# Let there be tokenized data: the whole TinyStories training set, and the same 500 validation stories as before
data = torch.load(os.path.join(root, 'data', 'tok', 'bpe1024_full.pt'))
train_data, val_data, vocab, BOS = data['train'].to(device), data['val'].to(device), data['vocab'], data['bos'] # 训练数据用 int16 存，约 1.2 GB，整份放进显存
vocab_size = len(vocab)
token_bytes = torch.tensor([len(b) for b in vocab], device=device) # to turn loss per token into bits per byte
train_bytes_per_token = sum(token_bytes[part.long()].sum().item() for part in train_data.split(2**26)) / len(train_data)
decode = lambda ids: b''.join(vocab[i] for i in ids).decode('utf-8', errors='replace')
print(f"train tokens: {len(train_data)}, val tokens: {len(val_data)}, bytes per token: {train_bytes_per_token:.2f}")

# Initialize the parameters, to store the knowledge of the model
n_layer = 6      # depth of the transformer neural network (number of layers)
n_embd = 256     # width of the network (embedding dimension)
block_size = 256 # maximum context length of the attention window
n_head = 4       # number of attention heads
head_dim = n_embd // n_head # derived dimension of each head：64，和主流模型一样
matrix = lambda nout, nin, std=0.08: (torch.randn(nout, nin, device=device) * std).requires_grad_() # 初始化 N(0, 0.08²)（1.10）
shapes = {'wte': (vocab_size, n_embd), 'wpe': (block_size, n_embd), 'lm_head': (vocab_size, n_embd)}
for i in range(n_layer):
    shapes[f'layer{i}.attn_wq'] = (n_embd, n_embd)
    shapes[f'layer{i}.attn_wk'] = (n_embd, n_embd)
    shapes[f'layer{i}.attn_wv'] = (n_embd, n_embd)
    shapes[f'layer{i}.attn_wo'] = (n_embd, n_embd)
    shapes[f'layer{i}.mlp_fc1'] = (4 * n_embd, n_embd)
    shapes[f'layer{i}.mlp_fc2'] = (n_embd, 4 * n_embd)
state_dict = {name: matrix(*shape) for name, shape in shapes.items()}
n_params = sum(p.numel() for p in state_dict.values())
n_vocab_params = state_dict['wte'].numel() + state_dict['lm_head'].numel()
print(f"num params: {n_params} ({n_vocab_params} in wte and lm_head)")

# Define the model architecture: a function mapping tokens and parameters to logits over what comes next
# Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
def linear(x, w):
    return x @ w.T

def softmax(logits):
    exps = (logits - logits.max(dim=-1, keepdim=True).values).exp()
    return exps / exps.sum(dim=-1, keepdim=True)

def rmsnorm(x):
    return x * (x.pow(2).mean(dim=-1, keepdim=True) + 1e-5) ** -0.5

def gpt(idx):
    B, T = idx.shape
    tok_emb = state_dict['wte'][idx] # token embedding                        (B, T, C)
    pos_emb = state_dict['wpe'][:T] # position embedding                      (T, C)
    x = tok_emb + pos_emb # joint token and position embedding
    x = rmsnorm(x) # note: not redundant due to backward pass via the residual connection

    for li in range(n_layer):
        # 1) Multi-head Attention block, all positions and all heads at once
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq']).view(B, T, n_head, head_dim).transpose(1, 2) # (B, H, T, D)
        k = linear(x, state_dict[f'layer{li}.attn_wk']).view(B, T, n_head, head_dim).transpose(1, 2) # (B, H, T, D)
        v = linear(x, state_dict[f'layer{li}.attn_wv']).view(B, T, n_head, head_dim).transpose(1, 2) # (B, H, T, D)
        x_attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)                             # (B, H, T, D)
        x_attn = x_attn.transpose(1, 2).reshape(B, T, n_embd)                                         # (B, T, C)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = x + x_residual
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = x.relu()
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = x + x_residual

    logits = linear(x, state_dict['lm_head']) # (B, T, vocab_size)
    return logits

def token_losses(x, y):
    with autocast():
        probs = softmax(gpt(x))
        return -probs.gather(-1, y.unsqueeze(-1)).squeeze(-1).log() # loss of every position, in nats

# A batch is batch_size random windows of block_size tokens; the target is the same window shifted by one
def get_batch(data, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,)).pin_memory().to(device, non_blocking=True)
    offsets = ix[:, None] + torch.arange(block_size, device=device)
    return data[offsets].long(), data[offsets + 1].long() # 存的是 int16，取出来以后再转成 embedding 要的 int64

# Validation: the same 500 stories as in every section, measured in bits per byte
# 训练集上也取同样多的固定窗口，用同样的方式评估，看有没有死记训练数据（1.10）
windows = lambda data, n: (torch.stack([data[i:i + block_size] for i in range(0, n * block_size, block_size)]).long(),
                           torch.stack([data[i + 1:i + block_size + 1] for i in range(0, n * block_size, block_size)]).long())
n_windows = (len(val_data) - 1) // block_size
val_x, val_y = windows(val_data, n_windows)
train_x, train_y = windows(train_data, n_windows)

@torch.no_grad()
def bpb(xs, ys):
    nats = sum(token_losses(xs[i:i + 8], ys[i:i + 8]).double().sum() for i in range(0, len(xs), 8)).item() # 求和用 fp64，否则累加误差很大
    return nats / math.log(2) / token_bytes[ys].sum().item()

# 优化器：Muon 管 Transformer 内部的矩阵，wte、wpe、lm_head 用 Adam（1.06、1.11），都用 PyTorch 自带的实现
matrices = [p for k, p in state_dict.items() if k.startswith('layer')]
others = [p for k, p in state_dict.items() if not k.startswith('layer')]
# Adam 不加 weight decay：torch.optim.Adam 的 weight decay 是 L2 的写法，1.10 里会让训练失败
optimizers = [torch.optim.Muon(matrices, lr=learning_rate, weight_decay=weight_decay), # 每步把参数乘上 (1 - lr·wd)，和 AdamW 一样与梯度分开做
              torch.optim.Adam(others, lr=learning_rate / 2, betas=(0.85, 0.99), eps=1e-8, fused=True)]
# 学习率调度：前 1% 线性 warmup，之后保持峰值，最后 20% 线性降到 0（WSD，1.07）
warmup = num_steps // 100
lr_mult = lambda step: min((step + 1) / warmup, 1.0, (num_steps - step) / (0.2 * num_steps))
schedulers = [torch.optim.lr_scheduler.LambdaLR(o, lr_mult) for o in optimizers]
print(f"run: {run_name}")

# 训练一步的前向：交给 torch.compile 编译，并用 CUDA Graph（1.09）
@torch.compile(mode='reduce-overhead')
def batch_loss(x, y):
    return token_losses(x, y).mean()

batch_size = 64
num_evals = 64
log = open(os.path.join(out_dir, 'log.csv'), 'w')
log.write('step,tokens,text_bytes,time,lr,train_loss,train_bpb,val_bpb\n')
log.write(f"0,0,0,0,{lr_mult(0):.6g},,{bpb(train_x, train_y):.6f},{bpb(val_x, val_y):.6f}\n")
# 计时：每隔 eval_every 步评估一次，计时只算两次评估之间的训练；读时间前先等 GPU 把排队的活干完
eval_every = num_steps // num_evals
train_time, loss_sum, diverged = 0.0, torch.zeros((), device=device), False
torch.cuda.synchronize()
t0 = time.perf_counter()
for step in range(num_steps):
    x, y = get_batch(train_data, batch_size)
    torch.compiler.cudagraph_mark_step_begin() # 告诉 CUDA Graph：新的一步开始了，上一步的输出可以覆盖
    loss = batch_loss(x, y)
    loss.backward()
    for o, sch in zip(optimizers, schedulers):
        o.step()
        o.zero_grad(set_to_none=True)
        sch.step()

    loss_sum += loss.detach() # 留在 GPU 上累加，不同步（1.09）
    if (step + 1) % eval_every == 0:
        torch.cuda.synchronize()
        train_time += time.perf_counter() - t0
        train_loss = loss_sum.item() / eval_every
        if not math.isfinite(train_loss): # 学习率太大，训练发散了
            print(f"\ndiverged before step {step+1}")
            diverged = True
            break
        tokens = (step + 1) * batch_size * block_size
        train_bpb, val_bpb = bpb(train_x, train_y), bpb(val_x, val_y)
        log.write(f"{step+1},{tokens},{tokens * train_bytes_per_token:.0f},{train_time:.4f},{lr_mult(step):.6g},{train_loss:.6f},{train_bpb:.6f},{val_bpb:.6f}\n")
        log.flush()
        print(f"step {step+1:5d} / {num_steps:5d} | train loss {train_loss:.4f} | val bpb {val_bpb:.4f} | {train_time:.0f}s", end='\r')
        loss_sum.zero_()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
log.close()
if diverged:
    sys.exit()

peak_memory = torch.cuda.max_memory_allocated() / 2**20 # 训练过程中显存占用的最高点（MB），包括显存里的训练数据
print(f"\ntrain bpb {train_bpb:.4f}, val bpb {val_bpb:.4f}, time {train_time:.1f}s, peak memory {peak_memory:.0f} MB")
with open(os.path.join(out_dir, 'final.csv'), 'w') as f:
    f.write('train_bpb,val_bpb,time,peak_memory\n')
    f.write(f"{train_bpb:.6f},{val_bpb:.6f},{train_time:.4f},{peak_memory:.1f}\n")
torch.save(state_dict, os.path.join(out_dir, 'model.pt')) # 存下参数，之后可以单独拿来生成或分析

# Inference: may the model babble back to us
temperature = 0.5 # in (0, 1], control the "creativity" of generated text, low to high
samples = open(os.path.join(out_dir, 'samples.txt'), 'w')
print("\n--- inference (new, hallucinated stories) ---")
with torch.no_grad():
    for sample_idx in range(3):
        idx = [BOS]
        for _ in range(block_size - 1):
            logits = gpt(torch.tensor([idx], device=device))[0, -1]
            probs = softmax(logits / temperature)
            token_id = random.choices(range(vocab_size), weights=probs.tolist())[0]
            if token_id == BOS:
                break
            idx.append(token_id)
        sample = decode(idx[1:])
        print(f"sample {sample_idx+1}: {sample}\n")
        samples.write(sample + '\n\n')
samples.close()
