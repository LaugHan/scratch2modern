"""
microgpt in PyTorch (CPU), trained on batches. Dataset: TinyStories.
Usage: python train_tinystories.py <batch_size>

Based on microgpt by @karpathy
"""

import os       # os.path.exists
import sys      # sys.argv
import random   # random.seed, random.choices, random.gauss, random.shuffle
import time     # time.perf_counter
import torch
random.seed(42) # Let there be order among chaos
torch.manual_seed(42)
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

# Project paths: read data from data/, write logs to out/<this section>/<run>/
batch_size = int(sys.argv[1])
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
data_path = os.path.join(root, 'data', 'tinystories_valid.txt')
out_dir = os.path.join(root, 'out', os.path.basename(os.path.dirname(os.path.abspath(__file__))), f'tinystories_b{batch_size}')
os.makedirs(out_dir, exist_ok=True)

# Let there be a Dataset `docs`: list[str] of documents (short stories)
docs = [doc.strip() for doc in open(data_path).read().split('<|endoftext|>') if doc.strip()]
docs = docs[1:] # the first story in the file is cut off
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars) # token id for a special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")

# One long stream of tokens: BOS story BOS story BOS ... ; 90% for training, 10% for validation
stoi = {ch: i for i, ch in enumerate(uchars)}
def encode(docs):
    return torch.tensor([t for doc in docs for t in [BOS] + [stoi[ch] for ch in doc]] + [BOS])
n_train = int(0.9 * len(docs))
train_data, val_data = encode(docs[:n_train]), encode(docs[n_train:])
print(f"train tokens: {len(train_data)}, val tokens: {len(val_data)}")

# Initialize the parameters, to store the knowledge of the model
n_layer = 1     # depth of the transformer neural network (number of layers)
n_embd = 16     # width of the network (embedding dimension)
block_size = 256 # maximum context length of the attention window
n_head = 4      # number of attention heads
head_dim = n_embd // n_head # derived dimension of each head
matrix = lambda nout, nin, std=0.08: torch.tensor([[random.gauss(0, std) for _ in range(nin)] for _ in range(nout)], requires_grad=True)
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = list(state_dict.values())
print(f"num params: {sum(p.numel() for p in params)}")

# Define the model architecture: a function mapping tokens and parameters to logits over what comes next
# Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
def linear(x, w):
    return x @ w.T

def softmax(logits):
    exps = (logits - logits.max(dim=-1, keepdim=True).values).exp()
    return exps / exps.sum(dim=-1, keepdim=True)

def rmsnorm(x):
    return x * (x.pow(2).mean(dim=-1, keepdim=True) + 1e-5) ** -0.5

causal_mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)) # position t sees positions <= t

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

def loss_fn(x, y):
    probs = softmax(gpt(x))
    return -probs.gather(-1, y.unsqueeze(-1)).log().mean()

# A batch is batch_size random windows of block_size tokens; the target is the same window shifted by one
def get_batch(data, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y

# Validation: a fixed set of windows, the same for every run (the first 2**16 val tokens)
val_x = torch.stack([val_data[i:i + block_size] for i in range(0, 2**16, block_size)])
val_y = torch.stack([val_data[i + 1:i + block_size + 1] for i in range(0, 2**16, block_size)])

@torch.no_grad()
def val_loss():
    return sum(loss_fn(val_x[i:i + 32], val_y[i:i + 32]) * len(val_x[i:i + 32]) for i in range(0, len(val_x), 32)).item() / len(val_x)

# Let there be Adam, the blessed optimizer and its buffers
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [torch.zeros_like(p) for p in params] # first moment buffer
v = [torch.zeros_like(p) for p in params] # second moment buffer

# Same token budget for every batch size: bigger batches take fewer steps
token_budget = 2**22
num_steps = token_budget // (batch_size * block_size)
num_evals = 32
print(f"batch size: {batch_size}, steps: {num_steps}")

log = open(os.path.join(out_dir, 'log.csv'), 'w')
log.write('step,tokens,time,train_loss,val_loss\n')
log.write(f"0,0,0,,{val_loss():.6f}\n")
train_time, train_losses = 0.0, []
for step in range(num_steps):
    t0 = time.perf_counter()

    x, y = get_batch(train_data, batch_size)
    loss = loss_fn(x, y)
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

    train_time += time.perf_counter() - t0
    train_losses.append(loss.item())
    if (step + 1) % max(1, num_steps // num_evals) == 0 or step == num_steps - 1:
        tokens = (step + 1) * batch_size * block_size
        train_loss = sum(train_losses) / len(train_losses)
        log.write(f"{step+1},{tokens},{train_time:.4f},{train_loss:.6f},{val_loss():.6f}\n")
        print(f"step {step+1:5d} / {num_steps:5d} | train loss {train_loss:.4f}", end='\r')
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
            logits = gpt(torch.tensor([idx]))[0, -1]
            probs = softmax(logits / temperature)
            token_id = random.choices(range(vocab_size), weights=probs.tolist())[0]
            if token_id == BOS:
                break
            idx.append(token_id)
        sample = ''.join(uchars[t] for t in idx[1:])
        print(f"sample {sample_idx+1}: {sample}\n")
        samples.write(sample + '\n\n')
samples.close()
