"""
microgpt in PyTorch (GPU)，在 TinyStories 上训练，tokenizer 为 bpe1024，宽度 48，bf16 混合精度。
数学不变，只追求快：工程上的加速一项一项打开，看每一项快了多少。
用法：python train.py <level> [seed]     level 是下面 levels 里的一个名字，打开它和它之前的所有加速

Based on microgpt by @karpathy
"""

import os       # os.path.join
import sys      # sys.argv
import math     # math.log
import random   # random.seed, random.choices, random.gauss
import time     # time.perf_counter
import torch
import torch.nn.functional as F # F.scaled_dot_product_attention
learning_rate, num_steps = 0.01, 4096
levels = ['baseline', 'no_sync', 'fused_adam', 'sdpa', 'compile', 'cudagraph']
level_name = sys.argv[1]
level = levels.index(level_name)
no_sync     = level >= 1 # 减少 CPU 和 GPU 之间的同步：loss 留在 GPU 上累加，数据提前放到 GPU 上
fused_adam  = level >= 2 # Adam 所有参数的更新合并成一个 kernel
sdpa        = level >= 3 # 注意力用 PyTorch 内置的 scaled_dot_product_attention
use_compile = level >= 4 # torch.compile：把一串逐元素运算融合成少数几个 kernel
cudagraph   = level >= 5 # CUDA Graph：把一整步前向和反向的 kernel 录下来，之后一次性重放
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
random.seed(seed) # Let there be order among chaos
torch.manual_seed(seed)
autocast = lambda: torch.autocast('cuda', dtype=torch.bfloat16) # 1.08 已经对比过：bf16 混合精度
torch.set_num_threads(1)
device = 'cuda:0'

# 项目路径：从 data/tok/ 读数据，日志写到 out/<这一节>/<level>/s<seed>/
tokenizer = 'bpe1024'
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
out_dir = os.path.join(root, 'out', os.path.basename(os.path.dirname(os.path.abspath(__file__))), level_name, f's{seed}')
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
matrix = lambda nout, nin, std=0.08: torch.tensor([[random.gauss(0, std) for _ in range(nin)] for _ in range(nout)], device=device, requires_grad=True)
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
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

causal_mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool, device=device)) # position t sees positions <= t

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
        if sdpa: # 和下面四行算的是同一个东西，PyTorch 会自动挑一个更快的实现
            x_attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)                         # (B, H, T, D)
        else:
            attn_logits = q @ k.transpose(-2, -1) / head_dim**0.5                                     # (B, H, T, T)
            attn_logits = attn_logits.masked_fill(~causal_mask[:T, :T], float('-inf'))
            attn_weights = softmax(attn_logits)
            x_attn = attn_weights @ v                                                                 # (B, H, T, D)
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
    ix = torch.randint(len(data) - block_size, (batch_size,))
    if no_sync: # 数据已经整份放在 GPU 上，只把 16 个起点传过去，在 GPU 上切出窗口
        ix = ix.pin_memory().to(device, non_blocking=True)
        offsets = ix[:, None] + torch.arange(block_size, device=device)
        return data[offsets], data[offsets + 1]
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# Validation: the same 500 stories for every tokenizer, measured in bits per byte so tokenizers can be compared
val_x = torch.stack([val_data[i:i + block_size] for i in range(0, len(val_data) - block_size - 1, block_size)]).to(device)
val_y = torch.stack([val_data[i + 1:i + block_size + 1] for i in range(0, len(val_data) - block_size - 1, block_size)]).to(device)
val_bytes = token_bytes[val_y].sum().item()

@torch.no_grad()
def val_bpb():
    nats = sum(token_losses(val_x[i:i + 8], val_y[i:i + 8]).double().sum() for i in range(0, len(val_x), 8)).item() # 求和用 fp64，否则纯 bf16 下累加误差很大
    return nats / math.log(2) / val_bytes

# 优化器和学习率调度在 1.06、1.07 已经对比过，这里直接用 PyTorch 自带的：Adam（β 沿用 microgpt 的设置），学习率线性衰减到 0
# fused=False 时，PyTorch 默认把所有参数打包、每种运算一个 kernel（foreach）；fused=True 时，整个 Adam 更新只用一个 kernel
optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(0.85, 0.99), eps=1e-8, fused=fused_adam)
lr_mult = lambda step: 1 - step / num_steps
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_mult)
print(f"level: {level_name} ({', '.join(levels[1:level + 1]) or 'nothing'} on), adam lr {learning_rate:g} linear decay, {num_steps} steps")

# 训练一步的前向：算出这个 batch 的平均 loss。打开 compile 时，交给 torch.compile 编译（第一次调用时编译，要花一些时间）
def batch_loss(x, y):
    return token_losses(x, y).mean()
if use_compile:
    batch_loss = torch.compile(batch_loss, mode='reduce-overhead' if cudagraph else 'default')

batch_size = 16
num_evals = 32
train_bytes_per_token = token_bytes[train_data.to(device)].double().mean().item()
print(f"bytes per token: {train_bytes_per_token:.2f}")
if no_sync:
    train_data = train_data.to(device) # 整份训练数据只有约 48 MB，直接放进显存

log = open(os.path.join(out_dir, 'log.csv'), 'w')
log.write('step,tokens,text_bytes,time,lr,train_loss,val_bpb\n')
log.write(f"0,0,0,0,{learning_rate * lr_mult(0):.6g},,{val_bpb():.6f}\n")
# 计时：每 128 步评估一次，计时只算两次评估之间的训练；读时间前先等 GPU 把排队的活干完
eval_every = num_steps // num_evals
train_time, loss_sum, train_losses, diverged = 0.0, 0.0, [], False
first_interval = None # 前 128 步的时间，打开 compile 时包含编译的时间
torch.cuda.synchronize()
t0 = time.perf_counter()
for step in range(num_steps):
    x, y = get_batch(train_data, batch_size)
    if cudagraph:
        torch.compiler.cudagraph_mark_step_begin() # 告诉 CUDA Graph：新的一步开始了，上一步的输出可以覆盖
    loss = batch_loss(x, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()

    if no_sync:
        loss_sum = loss_sum + loss.detach() # 留在 GPU 上累加，CPU 不用等 GPU，可以接着发下一步的指令
    else:
        train_losses.append(loss.item()) # .item() 要等 GPU 把这一步算完，CPU 才能接着往下走
        if not math.isfinite(train_losses[-1]): # 学习率太大，训练发散了
            print(f"\ndiverged at step {step+1}")
            diverged = True
            break
    if (step + 1) % eval_every == 0:
        torch.cuda.synchronize()
        train_time += time.perf_counter() - t0
        first_interval = first_interval or train_time
        train_loss = (loss_sum.item() if no_sync else sum(train_losses)) / eval_every
        if not math.isfinite(train_loss):
            print(f"\ndiverged before step {step+1}")
            diverged = True
            break
        tokens = (step + 1) * batch_size * block_size
        bpb = val_bpb()
        log.write(f"{step+1},{tokens},{tokens * train_bytes_per_token:.0f},{train_time:.4f},{learning_rate * lr_mult(step):.6g},{train_loss:.6f},{bpb:.6f}\n")
        print(f"step {step+1:5d} / {num_steps:5d} | train loss {train_loss:.4f} | val bpb {bpb:.4f}", end='\r')
        loss_sum, train_losses = 0.0, []
        torch.cuda.synchronize()
        t0 = time.perf_counter()
log.close()
if diverged:
    sys.exit()

# 每步耗时：去掉前 128 步（编译、预热），只看稳定以后的速度
ms_per_step = (train_time - first_interval) / (num_steps - eval_every) * 1e3
peak_memory = torch.cuda.max_memory_allocated() / 2**20
print(f"\nval bpb {bpb:.4f}, time {train_time:.1f}s (first {eval_every} steps {first_interval:.1f}s), {ms_per_step:.2f} ms/step, peak memory {peak_memory:.0f} MB")
with open(os.path.join(out_dir, 'final.csv'), 'w') as f:
    f.write('val_bpb,time,first_interval,ms_per_step,peak_memory_mb\n')
    f.write(f"{bpb:.6f},{train_time:.4f},{first_interval:.4f},{ms_per_step:.4f},{peak_memory:.1f}\n")

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
