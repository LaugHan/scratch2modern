"""
microgpt in PyTorch (GPU)，在 TinyStories 上训练，tokenizer 为 bpe1024，宽度 48，bf16 混合精度，打开 1.09 的全部加速。
第一章的最终配置。另外可以换上 1.06、1.07 里效果更好、但当时没有采用的两项：Muon 优化器、WSD 学习率调度。
用法：python train.py [opt=adam] [sched=linear] [seed=42]     例如 opt=muon sched=wsd

Based on microgpt by @karpathy
"""

import os       # os.path.join
import sys      # sys.argv
import math     # math.log
import random   # random.seed, random.choices, random.gauss
import time     # time.perf_counter
import torch
import torch.nn.functional as F # F.scaled_dot_product_attention
args = dict(opt='adam', sched='linear', seed='42') # 默认值
args.update(a.split('=') for a in sys.argv[1:])
opt   = args['opt']   # 优化器：adam（1.06 起一直用的）、muon（1.06 里最好的）
sched = args['sched'] # 学习率调度：linear（一直用的）、wsd（1.07 里最好的）
seed  = int(args['seed'])
learning_rate = 0.01 # Adam 的学习率（1.06）；Muon 只用在 Transformer 内部的矩阵上，学习率 0.02（1.06），其余参数仍用 Adam 0.01
num_steps = 4096
random.seed(seed) # Let there be order among chaos
torch.manual_seed(seed)
autocast = lambda: torch.autocast('cuda', dtype=torch.bfloat16) # 1.08 已经对比过：bf16 混合精度
torch.set_num_threads(1)
device = 'cuda:0'

# 项目路径：从 data/tok/ 读数据，日志写到 out/<这一节>/<opt>_<sched>_s<seed>/
tokenizer = 'bpe1024'
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
run_name = f"{opt}_{sched}_s{seed}"
out_dir = os.path.join(root, 'out', os.path.basename(os.path.dirname(os.path.abspath(__file__))), run_name)
os.makedirs(out_dir, exist_ok=True)

# Let there be tokenized data: two streams of token ids, and the bytes each token stands for
data = torch.load(os.path.join(root, 'data', 'tok', f'{tokenizer}.pt'))
train_data, val_data, vocab, BOS = data['train'], data['val'], data['vocab'], data['bos']
vocab_size = len(vocab)
token_bytes = torch.tensor([len(b) for b in vocab], device=device) # to turn loss per token into bits per byte
decode = lambda ids: b''.join(vocab[i] for i in ids).decode('utf-8', errors='replace')
print(f"tokenizer: {tokenizer}, vocab size: {vocab_size}, train tokens: {len(train_data)}, val tokens: {len(val_data)}")

# Initialize the parameters, to store the knowledge of the model
n_layer = 1     # depth of the transformer neural network (number of layers)
n_embd = 48     # width of the network (embedding dimension)
block_size = 256 # maximum context length of the attention window
n_head = 4      # number of attention heads
head_dim = n_embd // n_head # derived dimension of each head
matrix = lambda nout, nin, std=0.08: torch.tensor([[random.gauss(0, std) for _ in range(nin)] for _ in range(nout)], device=device, requires_grad=True) # 初始化 N(0, 0.08²)（1.10）
shapes = {'wte': (vocab_size, n_embd), 'wpe': (block_size, n_embd), 'lm_head': (vocab_size, n_embd)}
for i in range(n_layer):
    shapes[f'layer{i}.attn_wq'] = (n_embd, n_embd)
    shapes[f'layer{i}.attn_wk'] = (n_embd, n_embd)
    shapes[f'layer{i}.attn_wv'] = (n_embd, n_embd)
    shapes[f'layer{i}.attn_wo'] = (n_embd, n_embd)
    shapes[f'layer{i}.mlp_fc1'] = (4 * n_embd, n_embd)
    shapes[f'layer{i}.mlp_fc2'] = (n_embd, 4 * n_embd)
state_dict = {name: matrix(*shape) for name, shape in shapes.items()}
params = list(state_dict.values())
n_params = sum(p.numel() for p in params)
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
        x_attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)          # (B, H, T, D)
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
# 训练数据整份放在 GPU 上，只把 16 个起点传过去，在 GPU 上切出窗口（1.09）
def get_batch(data, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,)).pin_memory().to(device, non_blocking=True)
    offsets = ix[:, None] + torch.arange(block_size, device=device)
    return data[offsets], data[offsets + 1]

# Validation: the same 500 stories for every tokenizer, measured in bits per byte so tokenizers can be compared
val_x = torch.stack([val_data[i:i + block_size] for i in range(0, len(val_data) - block_size - 1, block_size)]).to(device)
val_y = torch.stack([val_data[i + 1:i + block_size + 1] for i in range(0, len(val_data) - block_size - 1, block_size)]).to(device)

@torch.no_grad()
def bpb(xs, ys):
    nats = sum(token_losses(xs[i:i + 8], ys[i:i + 8]).double().sum() for i in range(0, len(xs), 8)).item() # 求和用 fp64，否则累加误差很大
    return nats / math.log(2) / token_bytes[ys].sum().item()

# 优化器：Adam（1.06），fused 实现（1.09）；或者 Muon（1.06），用 PyTorch 自带的实现
if opt == 'adam':
    optimizers = [torch.optim.Adam(params, lr=learning_rate, betas=(0.85, 0.99), eps=1e-8, fused=True)]
elif opt == 'muon': # Muon 只管 Transformer 内部的矩阵；wte、wpe、lm_head 仍然用 Adam
    matrices = [p for k, p in state_dict.items() if k.startswith('layer')]
    others = [p for k, p in state_dict.items() if not k.startswith('layer')]
    optimizers = [torch.optim.Muon(matrices, lr=0.02, weight_decay=0.0), # PyTorch 的 Muon 默认带 weight decay 0.1，这里关掉，和 1.06 一致
                  torch.optim.Adam(others, lr=learning_rate, betas=(0.85, 0.99), eps=1e-8, fused=True)]
# 学习率调度：线性衰减到 0（microgpt 原本的做法），或者 WSD：前 80% 保持峰值，最后 20% 线性降到 0（1.07）
if sched == 'linear':
    lr_mult = lambda step: 1 - step / num_steps
elif sched == 'wsd':
    lr_mult = lambda step: min(1.0, (num_steps - step) / (0.2 * num_steps))
schedulers = [torch.optim.lr_scheduler.LambdaLR(o, lr_mult) for o in optimizers]
print(f"run: {run_name}")

# 训练一步的前向：交给 torch.compile 编译，并用 CUDA Graph（1.09）
@torch.compile(mode='reduce-overhead')
def batch_loss(x, y):
    return token_losses(x, y).mean()

batch_size = 16
num_evals = 32
train_bytes_per_token = token_bytes[train_data.to(device)].double().mean().item()
print(f"bytes per token: {train_bytes_per_token:.2f}")
train_data = train_data.to(device) # 整份训练数据只有约 48 MB，直接放进显存

log = open(os.path.join(out_dir, 'log.csv'), 'w')
log.write('step,tokens,text_bytes,time,lr,train_loss,val_bpb\n')
log.write(f"0,0,0,0,{lr_mult(0):.6g},,{bpb(val_x, val_y):.6f}\n")
# 计时：每 128 步评估一次，计时只算两次评估之间的训练；读时间前先等 GPU 把排队的活干完
eval_every = num_steps // num_evals
zero = lambda: torch.zeros((), device=device)
train_time, loss_sum, diverged = 0.0, zero(), False
torch.cuda.synchronize()
t0 = time.perf_counter()
for step in range(num_steps):
    x, y = get_batch(train_data, batch_size)
    torch.compiler.cudagraph_mark_step_begin() # 告诉 CUDA Graph：新的一步开始了，上一步的输出可以覆盖
    loss = batch_loss(x, y)
    loss.backward()
    for o in optimizers:
        o.step()
        o.zero_grad(set_to_none=True)
    for sch in schedulers:
        sch.step()

    # 都留在 GPU 上累加，不同步（1.09）
    loss_sum += loss.detach()
    if (step + 1) % eval_every == 0:
        torch.cuda.synchronize()
        train_time += time.perf_counter() - t0
        train_loss = loss_sum.item() / eval_every
        if not math.isfinite(train_loss): # 学习率太大，训练发散了
            print(f"\ndiverged before step {step+1}")
            diverged = True
            break
        tokens = (step + 1) * batch_size * block_size
        val_bpb = bpb(val_x, val_y)
        log.write(f"{step+1},{tokens},{tokens * train_bytes_per_token:.0f},{train_time:.4f},{lr_mult(step):.6g},{train_loss:.6f},{val_bpb:.6f}\n")
        print(f"step {step+1:5d} / {num_steps:5d} | train loss {train_loss:.4f} | val bpb {val_bpb:.4f}", end='\r')
        loss_sum = zero()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
log.close()
if diverged:
    sys.exit()

peak_memory = torch.cuda.max_memory_allocated() / 2**20 # 训练过程中显存占用的最高点（MB）
print(f"\nval bpb {val_bpb:.4f}, time {train_time:.1f}s, peak memory {peak_memory:.0f} MB")
with open(os.path.join(out_dir, 'final.csv'), 'w') as f:
    f.write('val_bpb,time,peak_memory\n')
    f.write(f"{val_bpb:.6f},{train_time:.4f},{peak_memory:.1f}\n")

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
