"""
和 ../train.py 相同，只是上下文长度和宽度可以从命令行设置，
用来在"上下文字节数相同"或"参数量相同"的条件下对比各个 tokenizer。
用法：python train.py <tokenizer> <block_size> <n_embd> <group>     例如 char 1015 16 context

Based on microgpt by @karpathy
"""

import os       # os.path.join
import sys      # sys.argv
import math     # math.log
import random   # random.seed, random.choices, random.gauss
import time     # time.perf_counter
import torch
random.seed(42) # Let there be order among chaos
torch.manual_seed(42)
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
device = 'cuda:0'

# 项目路径：从 data/tok/ 读数据，日志写到 out/1.05_tokenizer/analysis/<group>/<tokenizer>/
tokenizer, group = sys.argv[1], sys.argv[4]
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
out_dir = os.path.join(root, 'out', '1.05_tokenizer', 'analysis', group, tokenizer)
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
n_embd = int(sys.argv[3]) # 网络宽度（embedding 维度）
block_size = int(sys.argv[2]) # 注意力窗口的最大上下文长度（token 数）
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
        attn_logits = q @ k.transpose(-2, -1) / head_dim**0.5                                         # (B, H, T, T)
        attn_logits = attn_logits.masked_fill(~causal_mask[:T, :T], float('-inf'))
        attn_weights = softmax(attn_logits)
        x_attn = (attn_weights @ v).transpose(1, 2).reshape(B, T, n_embd)                             # (B, T, C)
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
    nats = sum(token_losses(val_x[i:i + 8], val_y[i:i + 8]).sum() for i in range(0, len(val_x), 8)).item()
    return nats / math.log(2) / val_bytes

# Let there be Adam, the blessed optimizer and its buffers
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [torch.zeros_like(p) for p in params] # first moment buffer
v = [torch.zeros_like(p) for p in params] # second moment buffer

# Same batch, same context, same number of steps for every tokenizer
batch_size = 16
num_steps = 4096
num_evals = 32
train_bytes_per_token = token_bytes[train_data.to(device)].double().mean().item()
print(f"bytes per token: {train_bytes_per_token:.2f}, context: {block_size} tokens = {block_size * train_bytes_per_token:.0f} bytes")

log = open(os.path.join(out_dir, 'log.csv'), 'w')
log.write('step,tokens,text_bytes,time,train_loss,val_bpb\n')
log.write(f"0,0,0,0,,{val_bpb():.6f}\n")
train_time, train_losses = 0.0, []
for step in range(num_steps):
    t0 = time.perf_counter()

    x, y = get_batch(train_data, batch_size)
    loss = token_losses(x, y).mean()
    loss.backward()

    # Adam optimizer update: update the model parameters based on the corresponding gradients
    lr_t = learning_rate * (1 - step / num_steps) # linear learning rate decay
    with torch.no_grad():
        for i, p in enumerate(params):
            m[i] = beta1 * m[i] + (1 - beta1) * p.grad
            v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
            m_hat = m[i] / (1 - beta1 ** (step + 1))
            v_hat = v[i] / (1 - beta2 ** (step + 1))
            p -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
            p.grad = None

    train_losses.append(loss.item()) # .item() waits for the GPU to finish this step
    train_time += time.perf_counter() - t0
    if (step + 1) % (num_steps // num_evals) == 0:
        tokens = (step + 1) * batch_size * block_size
        train_loss = sum(train_losses) / len(train_losses)
        bpb = val_bpb()
        log.write(f"{step+1},{tokens},{tokens * train_bytes_per_token:.0f},{train_time:.4f},{train_loss:.6f},{bpb:.6f}\n")
        print(f"step {step+1:5d} / {num_steps:5d} | train loss {train_loss:.4f} | val bpb {bpb:.4f}", end='\r')
        train_losses = []
log.close()

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
