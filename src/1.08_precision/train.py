"""
microgpt in PyTorch (GPU)，在 TinyStories 上训练，tokenizer 为 bpe1024，宽度 48。
优化器固定为 Adam，学习率 0.01 线性衰减。对比不同的数据精度，每种精度在 precisions/ 里一个文件。
用法：python train.py <precision> [seed]     例如 bf16_autocast

Based on microgpt by @karpathy
"""

import os       # os.path.join
import sys      # sys.argv
import math     # math.log
import random   # random.seed, random.choices, random.gauss
import time     # time.perf_counter
import importlib # 按名字加载 precisions/ 里的文件
import contextlib # contextlib.nullcontext
import torch
learning_rate, num_steps = 0.01, 4096
precision_name = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
random.seed(seed) # Let there be order among chaos
torch.manual_seed(seed)
# 数据精度：参数用什么格式存、矩阵乘法用什么格式算、要不要放大 loss，定义在 precisions/<name>.py
precision = importlib.import_module(f'precisions.{precision_name}')
torch.backends.cuda.matmul.allow_tf32 = precision.tf32
autocast = lambda: torch.autocast('cuda', dtype=precision.autocast) if precision.autocast else contextlib.nullcontext()
torch.set_num_threads(1)
device = 'cuda:0'

# 项目路径：从 data/tok/ 读数据，日志写到 out/<这一节>/<precision>/s<seed>/
tokenizer = 'bpe1024'
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
out_dir = os.path.join(root, 'out', os.path.basename(os.path.dirname(os.path.abspath(__file__))), precision_name, f's{seed}')
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
matrix = lambda nout, nin, std=0.08: torch.tensor([[random.gauss(0, std) for _ in range(nin)] for _ in range(nout)], dtype=precision.dtype, device=device, requires_grad=True)
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
block_linear = precision.linear or linear # Transformer 层内的线性层（fp8 时换成 fp8 矩阵乘法）

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
        q = block_linear(x, state_dict[f'layer{li}.attn_wq']).view(B, T, n_head, head_dim).transpose(1, 2) # (B, H, T, D)
        k = block_linear(x, state_dict[f'layer{li}.attn_wk']).view(B, T, n_head, head_dim).transpose(1, 2) # (B, H, T, D)
        v = block_linear(x, state_dict[f'layer{li}.attn_wv']).view(B, T, n_head, head_dim).transpose(1, 2) # (B, H, T, D)
        attn_logits = q @ k.transpose(-2, -1) / head_dim**0.5                                         # (B, H, T, T)
        attn_logits = attn_logits.masked_fill(~causal_mask[:T, :T], float('-inf'))
        attn_weights = softmax(attn_logits)
        x_attn = (attn_weights @ v).transpose(1, 2).reshape(B, T, n_embd)                             # (B, T, C)
        x = block_linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = x + x_residual
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = block_linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = x.relu()
        x = block_linear(x, state_dict[f'layer{li}.mlp_fc2'])
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
optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(0.85, 0.99), eps=1e-8)
lr_mult = lambda step: 1 - step / num_steps
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_mult)
# loss scaling：反向传播前把 loss 放大，免得小梯度在 fp16 里下溢成 0；更新前再缩回去。
# 倍数动态调整：梯度溢出成 inf 就跳过这一步、倍数减半，连续 2000 步没溢出就翻倍。不需要时 enabled=False，什么都不做
scaler = torch.amp.GradScaler('cuda', init_scale=precision.loss_scale or 1.0, growth_interval=2000, enabled=precision.loss_scale is not None)
print(f"precision: {precision_name}, adam lr {learning_rate:g} linear decay, {num_steps} steps")
matrix_keys = [k for k in state_dict if k.startswith('layer') or k == 'lm_head'] # 统计梯度里有多少恰好是 0（fp16 下溢的迹象）
skipped_steps = 0

batch_size = 16
num_evals = 32
train_bytes_per_token = token_bytes[train_data.to(device)].double().mean().item()
print(f"bytes per token: {train_bytes_per_token:.2f}")

log = open(os.path.join(out_dir, 'log.csv'), 'w')
log.write('step,tokens,text_bytes,time,lr,train_loss,val_bpb,grad_zero,loss_scale\n')
log.write(f"0,0,0,0,{learning_rate * lr_mult(0):.6g},,{val_bpb():.6f},,{scaler.get_scale():g}\n")
train_time, train_losses, diverged = 0.0, [], False
for step in range(num_steps):
    t0 = time.perf_counter()

    x, y = get_batch(train_data, batch_size)
    loss = token_losses(x, y).mean()
    scaler.scale(loss).backward() # 不需要 loss scaling 时就是 loss.backward()
    if (step + 1) % (num_steps // num_evals) == 0:
        grad_zero = sum((state_dict[k].grad == 0).sum().item() for k in matrix_keys) / sum(state_dict[k].numel() for k in matrix_keys)

    scale = scaler.get_scale()
    scaler.step(optimizer) # 先把梯度缩回原来的大小；如果有 inf，就跳过这一步
    scaler.update()        # 调整倍数
    skipped_steps += scaler.get_scale() < scale # 倍数变小，说明这一步溢出、被跳过了
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()

    train_losses.append(loss.item()) # .item() waits for the GPU to finish this step
    train_time += time.perf_counter() - t0
    if not math.isfinite(train_losses[-1]): # 学习率太大，训练发散了
        print(f"\ndiverged at step {step+1}")
        diverged = True
        break
    if (step + 1) % (num_steps // num_evals) == 0:
        tokens = (step + 1) * batch_size * block_size
        train_loss = sum(train_losses) / len(train_losses)
        bpb = val_bpb()
        log.write(f"{step+1},{tokens},{tokens * train_bytes_per_token:.0f},{train_time:.4f},{learning_rate * lr_mult(step):.6g},{train_loss:.6f},{bpb:.6f},{grad_zero:.6f},{scaler.get_scale():g}\n")
        print(f"step {step+1:5d} / {num_steps:5d} | train loss {train_loss:.4f} | val bpb {bpb:.4f}", end='\r')
        train_losses = []
log.close()
if diverged:
    sys.exit()

# 最后把参数转成 fp64、去掉 autocast 和 fp8 再评估一次：只比较训练出来的权重，不掺进评估时的精度损失
peak_memory = torch.cuda.max_memory_allocated() / 2**20
for k in state_dict:
    state_dict[k] = state_dict[k].detach().double()
autocast, block_linear = contextlib.nullcontext, linear
bpb_fp64 = val_bpb()
print(f"\nval bpb {bpb:.4f} (fp64: {bpb_fp64:.4f}), time {train_time:.1f}s, peak memory {peak_memory:.0f} MB, skipped steps {skipped_steps}")
with open(os.path.join(out_dir, 'final.csv'), 'w') as f:
    f.write('val_bpb,val_bpb_fp64,time,peak_memory_mb,skipped_steps\n')
    f.write(f"{bpb:.6f},{bpb_fp64:.6f},{train_time:.4f},{peak_memory:.1f},{skipped_steps}\n")

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
